import argparse
import os
import sys

import numpy as np
import torch
import wandb
from diffsynth import ModelManager, WanVideoPipeline, save_video
from diffsynth.models.camera import CamVidEncoder

from .model_setup import add_lora_to_model, setup_ic_light_pipeline
from .pipeline_relight import WanVideoPipelineWithRelight
from .video_io import load_mask_frames, load_video_frames


light_a_video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Light-A-Video'))
if light_a_video_path in sys.path:
    sys.path.remove(light_a_video_path)
sys.path.insert(0, light_a_video_path)
from src.ic_light import BGSource, Relighter


def main():
    parser = argparse.ArgumentParser(description='EX-4D with Relighting')
    
    # Video paths
    parser.add_argument('--color_video', type=str, required=True, help='Path to input color video')
    parser.add_argument('--mask_video', type=str, required=True, help='Path to mask video')
    parser.add_argument('--output_video', type=str, required=True, help='Path to output video')
    
    # Video parameters
    parser.add_argument('--height', type=int, default=512, help='Output height')
    parser.add_argument('--width', type=int, default=512, help='Output width')
    parser.add_argument('--num_frames', type=int, default=49, help='Number of frames to process')
    
    # EX-4D model paths
    parser.add_argument('--ex4d_path', type=str, default='models/EX-4D/ex4d.ckpt', help='Path to EX-4D model')
    parser.add_argument('--text_encoder_path', type=str, 
                       default='models/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth',
                       help='Path to text encoder')
    parser.add_argument('--vae_path', type=str,
                       default='models/Wan-AI/Wan2.1_VAE.pth',
                       help='Path to VAE model')
    parser.add_argument('--clip_path', type=str,
                       default='models/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
                       help='Path to CLIP model')
    parser.add_argument('--dit_dir', type=str,
                       default='models/Wan-AI/',
                       help='Directory containing DiT model files')
    
    # IC-Light model paths
    parser.add_argument('--sd_model', type=str, 
                       default='stablediffusionapi/realistic-vision-v51',
                       help='Path to Stable Diffusion model for IC-Light')
    parser.add_argument('--ic_light_model', type=str, 
                       default='./models/iclight_sd15_fc.safetensors',
                       help='Path to IC-Light model weights')
    
    # Relighting switch - Main control for enabling/disabling IC-Light
    parser.add_argument('--enable_relight', action='store_true', default=False,
                       help='Enable IC-Light relighting. If not set, runs without relighting (faster, no light effects)')
    
    # Relighting parameters (only used when --enable_relight is set)
    parser.add_argument('--light_direction', type=str, default='TOP',
                       choices=['NONE', 'LEFT', 'RIGHT', 'TOP', 'BOTTOM'],
                       help='Light source direction (only used with --enable_relight)')
    parser.add_argument('--relight_prompt', type=str, 
                       default='natural lighting, soft light, realistic',
                       help='Prompt for relighting style (only used with --enable_relight)')
    parser.add_argument('--gamma', type=float, default=0.7,
                       help='CLA gamma parameter for temporal consistency (only used with --enable_relight)')
    parser.add_argument('--lowres_denoise', type=float, default=0.9,
                       help='Relighting denoising strength (only used with --enable_relight)')
    parser.add_argument('--relight_steps', type=int, default=15,
                       help='Number of IC-Light denoising steps (only used with --enable_relight)')
    parser.add_argument('--relight_cfg', type=float, default=2.0,
                       help='IC-Light CFG scale (only used with --enable_relight)')
    
    # Inference parameters
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps')
    
    # Progressive fusion parameters (only used when --enable_relight is set)
    parser.add_argument('--fusion_k', type=float, default=1.0,
                       help='Lambda decay factor for progressive fusion (only used with --enable_relight)')
    parser.add_argument('--fusion_threshold', type=float, default=0.2,
                       help='Stop fusion when lambda < threshold (only used with --enable_relight)')
    parser.add_argument('--save_intermediate_steps', action='store_true', default=False,
                       help='Save intermediate relighting results for debugging')
    parser.add_argument('--intermediate_dir', type=str, default='outputs/intermediate',
                       help='Directory to save intermediate results')
    parser.add_argument('--use_progressive_fusion', action='store_true', default=False,
                       help='Use progressive fusion (only used with --enable_relight)')
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='4D-relight',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity name (optional)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Custom wandb run name (auto-generated if not provided)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None,
                       help='Tags for wandb run')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        # Generate custom run name if not provided
        run_name = args.wandb_run_name
        if run_name is None:
            # Create a descriptive run name based on parameters
            video_name = os.path.splitext(os.path.basename(args.color_video))[0]
            light_dir = args.light_direction if args.enable_relight else "no-relight"
            run_name = f"{video_name}_{light_dir}_steps{args.num_inference_steps}"
        
        # Initialize wandb
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.wandb_tags,
            config={
                "color_video": args.color_video,
                "mask_video": args.mask_video,
                "output_video": args.output_video,
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "seed": args.seed,
                "num_inference_steps": args.num_inference_steps,
                "enable_relight": args.enable_relight,
                "light_direction": args.light_direction if args.enable_relight else "N/A",
                "relight_prompt": args.relight_prompt if args.enable_relight else "N/A",
                "gamma": args.gamma if args.enable_relight else "N/A",
                "lowres_denoise": args.lowres_denoise if args.enable_relight else "N/A",
                "relight_steps": args.relight_steps if args.enable_relight else "N/A",
                "relight_cfg": args.relight_cfg if args.enable_relight else "N/A",
                "fusion_k": args.fusion_k if args.enable_relight else "N/A",
                "fusion_threshold": args.fusion_threshold if args.enable_relight else "N/A",
            }
        )
        print(f"Initialized wandb run: {wandb.run.name}")
        print(f"Wandb URL: {wandb.run.url}")
    
    # Load videos
    print(f"Loading color video: {args.color_video}")
    color_tensor = load_video_frames(args.color_video, args.num_frames, args.height, args.width)
    
    print(f"Loading mask video: {args.mask_video}")
    mask_tensor = load_mask_frames(args.mask_video, args.num_frames, args.height, args.width)
    
    # Apply mask to color video
    color_tensor = (color_tensor * mask_tensor).to(torch.bfloat16) * 2 - 1
    mask_tensor = mask_tensor.to(torch.bfloat16) * 2 - 1
    
    # Clear GPU cache before loading models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Print execution mode
    print("\n" + "="*60)
    if args.enable_relight:
        print("🌟 Running with IC-Light RELIGHTING enabled")
        print(f"   Light direction: {args.light_direction}")
        print(f"   Relight steps: {args.relight_steps}")
        print(f"   Fusion parameters: k={args.fusion_k}, threshold={args.fusion_threshold}")
    else:
        print("⚡ Running WITHOUT relighting (standard EX-4D mode)")
        print("   Faster processing, no light effects")
    print("="*60 + "\n")
    
    # Load EX-4D models
    print("Loading EX-4D models...")
    dit_paths = [
        os.path.join(args.dit_dir, f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors")
        for i in range(1, 8)
    ]
    
    # Use more memory-efficient settings
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=args.device)
    
    # Check available memory before loading
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"Available GPU memory before loading: {free_mem / 1024**3:.2f} GB")
    
    try:
        model_manager.load_models([dit_paths, args.text_encoder_path, args.vae_path, args.clip_path])
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory: {e}")
        print("Trying to free memory and retry with smaller batch size...")
        torch.cuda.empty_cache()
        # Try loading with more aggressive memory management
        model_manager.load_models([dit_paths, args.text_encoder_path, args.vae_path, args.clip_path])
    
    # Use appropriate pipeline based on mode
    if args.enable_relight:
        print("Initializing WanVideoPipelineWithRelight (with IC-Light support)...")
        pipe = WanVideoPipelineWithRelight(device=args.device, torch_dtype=torch.bfloat16)
        pipe.fetch_models(model_manager)
    else:
        print("Initializing standard WanVideoPipeline (without IC-Light)...")
        pipe = WanVideoPipeline.from_model_manager(model_manager, device=args.device)
    
    pipe.camera_encoder = CamVidEncoder(16, 1024, 5120).to(args.device, dtype=torch.bfloat16)
    
    # Add LoRA
    print("Loading EX-4D LoRA...")
    add_lora_to_model(pipe.denoising_model(), pretrained_path=args.ex4d_path, lora_rank=16, lora_alpha=16.0)
    pipe.load_cam(args.ex4d_path)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.camera_encoder = pipe.camera_encoder.to(args.device, dtype=torch.bfloat16)
    pipe.camera_encoder.eval()
    
    # Setup IC-Light Relighter for progressive fusion
    relighter = None
    if args.enable_relight:
        print("\n=== Setting up IC-Light for Progressive Relighting ===")
        
        # Setup IC-Light pipeline with CLA
        ic_light_pipe = setup_ic_light_pipeline(
            args.sd_model, 
            args.ic_light_model, 
            args.device, 
            torch.float16,  # IC-Light uses float16
            gamma=args.gamma
        )
        
        # Get light source
        bg_source = BGSource[args.light_direction]
        print(f"Creating IC-Light Relighter with {bg_source.value}")
        
        # Create Relighter object using existing ic_light.py module
        generator = torch.manual_seed(args.seed)
        relighter = Relighter(
            pipeline=ic_light_pipe,
            relight_prompt=args.relight_prompt,
            num_frames=args.num_frames,
            image_width=args.width,
            image_height=args.height,
            num_samples=1,
            steps=args.relight_steps,
            cfg=args.relight_cfg,
            lowres_denoise=args.lowres_denoise,
            bg_source=bg_source,
            generator=generator
        )
        print(f"Relighting denoising strength: {args.lowres_denoise}")
        # Set relighter to pipeline
        pipe.set_relighter(relighter)
        
        print("IC-Light Relighter ready for progressive fusion")
    
    # Set prompts
    base_prompt = "4K ultra HD, surround motion, realistic tone, panoramic shot, wide-angle view, cinematic quality"
    if args.enable_relight and args.relight_prompt:
        # prompt = f"{base_prompt}, {args.relight_prompt}"
        prompt = f"{base_prompt}"
    else:
        prompt = base_prompt
    
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    # Run EX-4D inference with or without progressive relighting
    print("\n=== Running EX-4D inference ===")
    if args.enable_relight:
        print(f"Mode: WITH IC-Light progressive fusion")
        print(f"  Light direction: {args.light_direction}")
        print(f"  Fusion parameters: k={args.fusion_k}, threshold={args.fusion_threshold}")
    else:
        print(f"Mode: Standard EX-4D (NO relighting)")
    
    input_cond = color_tensor.to(args.device)[None]  # Add batch dimension
    input_mask = mask_tensor.to(args.device)[None]
    
    # Prepare pipeline arguments
    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "video": input_cond,
        "mask": input_mask,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "tiled": False,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
    }
    
    # Add relighting-specific arguments only when enabled
    if args.enable_relight:
        pipe_kwargs.update({
            "use_progressive_fusion": args.use_progressive_fusion,
            "fusion_k": args.fusion_k,
            "fusion_threshold": args.fusion_threshold,
            "save_intermediate_steps": args.save_intermediate_steps,
            "intermediate_dir": args.intermediate_dir,
        })
    with torch.no_grad(), torch.amp.autocast(dtype=torch.bfloat16, device_type=args.device):
        output_video = pipe(**pipe_kwargs)
    
    # Save output video
    print(f"\nSaving output video: {args.output_video}")
    save_video(output_video, args.output_video, fps=15, quality=8)
    print("Done!")
    
    # Upload to wandb if enabled
    if args.use_wandb:
        print("Uploading video to wandb...")
        # Create a custom name for the video in wandb
        video_name = os.path.basename(args.output_video)
        
        # Log the video to wandb
        wandb.log({
            "output_video": wandb.Video(args.output_video, caption=video_name, format="mp4"),
        })
        
        # Also log the input videos for comparison
        wandb.log({
            "input_color_video": wandb.Video(args.color_video, caption=f"Input: {os.path.basename(args.color_video)}", format="mp4"),
        })
        
        if args.mask_video:
            wandb.log({
                "input_mask_video": wandb.Video(args.mask_video, caption=f"Mask: {os.path.basename(args.mask_video)}", format="mp4"),
            })
        
        print(f"Video uploaded to wandb: {wandb.run.url}")
        
        # Finish the wandb run
        wandb.finish()
    
    if args.enable_relight:
        print(f"\n Relighting applied with {args.light_direction} light source")
        print(f"   Relight prompt: {args.relight_prompt}")
        print(f"   CLA gamma: {args.gamma}")
    else:
        print(f"\n Standard EX-4D processing completed (no relighting)")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared after processing")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared after error")
        raise
