import os

import numpy as np
import torch
from diffsynth import WanVideoPipeline
from einops import rearrange
from tqdm import tqdm

from .fusion import flow_matching_fusion_step
from .temporal import global_histogram_matching, temporal_smooth_lighting


class WanVideoPipelineWithRelight(WanVideoPipeline):
    """Extended WanVideoPipeline with progressive IC-Light relighting"""
    
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        self.relighter = None
        self.reference_frames = None  # Reference frames captured at the first fusion step.
        self.fixed_detail_diff = None  # Deprecated; dynamic detail difference is used.
    
    def set_relighter(self, relighter):
        """Set the IC-Light relighter for progressive fusion"""
        self.relighter = relighter
    
    def get_fusion_weight(self, current_step, total_steps, start_ratio=0.6, end_ratio=0.8):
        """
        Time-aware fusion schedule.

        Uses a progressive weight to avoid abrupt trajectory changes.
        
        Args:
            current_step: current diffusion step
            total_steps: total number of steps
            start_ratio: ratio to start fusion (default 0.6)
            end_ratio: ratio to reach full ramp (default 0.8)
        
        Returns:
            weight in [0, 1]
        """
        current_ratio = current_step / max(total_steps - 1, 1)
        
        if current_ratio < start_ratio:
            return 0.0
        # elif current_ratio < end_ratio:
        #     # Mid stage: linear ramp-up.
        #     return (current_ratio - start_ratio) / (end_ratio - start_ratio) * 0.2
        # else:
        #     # Late stage: full fusion.
        #     return 1.0
        else:
            return (current_ratio - start_ratio) / (end_ratio - start_ratio) * 0.5
    
    def get_adaptive_fusion_weight(self, current_step, total_steps, 
                                   geometry_ratio=0.7, 
                                   ramp_end_ratio=0.85,
                                   stable_ratio=0.95,
                                   max_weight=0.6,
                                   final_weight=0.4):

        progress = current_step / max(total_steps - 1, 1)
        print(f"max_weight: {max_weight}")
        if progress < geometry_ratio:
            # Stage 1: geometry completion only.
            return 0.0
            
        elif progress < ramp_end_ratio:
            # Stage 2: fast ramp-up with square-root growth.
            local_progress = (progress - geometry_ratio) / (ramp_end_ratio - geometry_ratio)
            return max_weight * (local_progress ** 0.5)
            
        elif progress < stable_ratio:
            # Stage 3: hold maximum weight for stabilization.
            return max_weight
            
        else:
            # Stage 4: fade down for final convergence.
            fade_progress = (progress - stable_ratio) / (1.0 - stable_ratio)
            current_weight = max_weight - (max_weight - final_weight) * fade_progress
            return current_weight
    

    def _save_frames(self, frames, save_path):
        """
        Save video frames as image grid.
        Args:
            frames: [F, C, H, W] tensor in range [0, 1]
            save_path: path to save the image
        """
        import torchvision
        # Take middle frame or create a grid of multiple frames
        num_frames = frames.shape[0]
        if num_frames <= 8:
            # Save all frames as grid
            grid = torchvision.utils.make_grid(frames, nrow=min(num_frames, 4), padding=2)
        else:
            # Sample 8 frames evenly
            indices = torch.linspace(0, num_frames - 1, 8).long()
            selected_frames = frames[indices]
            grid = torchvision.utils.make_grid(selected_frames, nrow=4, padding=2)
        
        # Save image
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(grid_np).save(save_path)
    
    def _save_intermediate_grids(self, intermediate_frames, save_dir, num_steps):
        """
        Save intermediate results as 2 rows × 10 columns grids.
        Args:
            intermediate_frames: dict with keys like 'consist', 'relight', 'fusion', 'latents', 'no_relight'
            save_dir: directory to save grids
            num_steps: total number of inference steps
        """
        import torchvision
        from PIL import Image, ImageDraw, ImageFont
        
        # Iterate over the keys that actually exist in intermediate_frames
        for key in intermediate_frames.keys():
            if len(intermediate_frames[key]) == 0:
                continue
                
            # Special handling for latents: decode to RGB images
            # 'no_relight' also contains latents that need decoding
            if key in ['latents', 'no_relight']:
                print(f"  Decoding {len(intermediate_frames[key])} latent steps to RGB...")
                decoded_frames = []
                
                for latent in intermediate_frames[key]:
                    # latent shape: [B, C, F, H, W]
                    # Apply normalization (same as in main decoding)
                    vae_dtype = next(iter(self.vae.parameters())).dtype
                    latent = latent.to(self.device).to(vae_dtype)
                    
                    # Decode to video using decode_video method
                    self.load_models_to_device(['vae'])
                    with torch.no_grad():
                        decoded = self.decode_video(latent, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
                    
                    # Use tensor2video to convert [B, C, F, H, W] -> List of PIL Images
                    # This ensures the same processing as final output
                    video_frames = self.tensor2video(decoded[0])  # decoded[0]: [C, F, H, W] -> List[PIL Image]
                    
                    # Take middle frame from the video
                    mid_frame_idx = len(video_frames) // 2
                    pil_frame = video_frames[mid_frame_idx]
                    
                    # Convert PIL Image back to tensor [C, H, W] for grid creation
                    # Note: tensor2video already normalized to [0, 255], convert to [0, 1]
                    frame_array = np.array(pil_frame).astype(np.float32) / 255.0  # [H, W, C]
                    frame = torch.from_numpy(frame_array).permute(2, 0, 1)  # [C, H, W]
                    decoded_frames.append(frame.cpu())
                
                frames = torch.stack(decoded_frames, dim=0)  # [N, C, H, W]
                num_collected = frames.shape[0]
            else:
                # For consist/relight/fusion: these are in [-1, 1] range from decode_video
                # Need to normalize to [0, 1] range (same as tensor2video does)
                raw_frames = torch.stack(intermediate_frames[key], dim=0)  # [N, C, H, W]
                
                # Apply same normalization as tensor2video: [-1, 1] -> [0, 1]
                # tensor2video does: ((frames + 1) * 127.5).clip(0, 255) -> [0, 255]
                # We want [0, 1], so: ((frames + 1) / 2).clip(0, 1)
                frames = ((raw_frames + 1.0) / 2.0).clamp(0.0, 1.0)
                num_collected = frames.shape[0]
            
            # Pad or sample to 20 frames (2 rows × 10 columns)
            target_frames = 24
            if num_collected < target_frames:
                # Pad with last frame
                padding = [frames[-1:]] * (target_frames - num_collected)
                frames = torch.cat([frames] + padding, dim=0)
            elif num_collected > target_frames:
                # Sample evenly
                indices = torch.linspace(0, num_collected - 1, target_frames).long()
                frames = frames[indices]
            
            # Create grid: 2 rows × 10 columns
            grid = torchvision.utils.make_grid(frames, nrow=10, padding=4, pad_value=1.0)
            
            # Convert to PIL Image (frames are now in [0, 1] range)
            grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(grid_np)
            
            # Add text labels for each step
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Calculate position for each frame in grid
            frame_height = frames.shape[2]
            frame_width = frames.shape[3]
            padding = 4
            
            for i in range(min(target_frames, num_collected)):
                row = i // 10
                col = i % 10
                x = col * (frame_width + padding) + 5
                y = row * (frame_height + padding) + 5
                
                # Draw step number
                step_num = i * (num_steps // target_frames) if num_collected >= target_frames else i
                text = f"Step {step_num}"
                draw.text((x, y), text, fill=(255, 255, 0), font=font)
            
            # Save grid image
            save_path = os.path.join(save_dir, f"grid_{key}_2x10.png")
            img.save(save_path)
            print(f"  Saved {key} grid: {save_path}")

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        video,
        mask,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        # IC-Light progressive fusion parameters
        use_progressive_fusion=False,
        fusion_k=1,
        fusion_threshold=0.2,
        # Intermediate results saving
        save_intermediate_steps=False,
        intermediate_dir="outputs/intermediate",
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32).to(self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=noise.dtype, device=noise.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if video is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_images(video[:, :, 0], num_frames, height, width)
        else:
            image_emb = {}
        ray_latent = self.encode_rays(video, mask)
        image_emb["ray_latent"] = ray_latent
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # Initialize optional mask-based residual compensation state.
        if use_progressive_fusion and self.relighter is not None:
            org_target = video.clone()
            # Convert mask from [-1, 1] to [0, 1] for weighting.
            org_mask = ((mask.clone() + 1.0) / 2.0).clamp(0, 1)
            # Reset references for each pipeline call.
            self.reference_frames = None
            self.fixed_detail_diff = None
        else:
            org_target = None
            org_mask = None
        
        # Create intermediate directory if saving
        intermediate_frames = {'consist': [], 'relight': [], 'fusion': [], 'latents': [], 'no_relight': []}
        # if save_intermediate_steps and use_progressive_fusion and self.relighter is not None:
        #     os.makedirs(intermediate_dir, exist_ok=True)
        #     print(f"Saving intermediate results to: {intermediate_dir}")

        # Denoise with progressive IC-Light fusion
        self.load_models_to_device(["dit"])
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep_tensor = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)

                # Inference
                noise_pred_posi = self.dit(latents, timestep=timestep_tensor, **prompt_emb_posi, **image_emb, **extra_input)
                if cfg_scale != 1.0:
                    noise_pred_nega = self.dit(latents, timestep=timestep_tensor, **prompt_emb_nega, **image_emb, **extra_input)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi
                
                
                # Progressive IC-Light fusion
                if use_progressive_fusion and self.relighter is not None:
                    # Four-stage adaptive fusion schedule.
                    fusion_weight = self.get_adaptive_fusion_weight(
                        current_step=progress_id,
                        total_steps=num_inference_steps,
                        geometry_ratio=0.75,    # Geometry-first phase.
                        ramp_end_ratio=0.85,   # Reach max fusion weight.
                        stable_ratio=0.90,     # Start final fade.
                        max_weight=0.5,        # Peak fusion weight.
                        final_weight=0.2       # Final convergence weight.
                    )
                    if fusion_weight > 0 :
                        # Get predicted x0
                        sigma = self.scheduler.sigmas[progress_id] if hasattr(self.scheduler, 'sigmas') else 1.0
                        pred_x0_latent = latents - sigma * noise_pred
                        
                        # Decode to video frames
                        self.load_models_to_device(['vae'])
                        consist_target = self.decode_video(pred_x0_latent, **tiler_kwargs)

                        # Apply IC-Light relighting using existing Relighter
                        # consist_target: [B, C, F, H, W], reshape to [F, C, H, W] for relighter
                        consist_target_frames = rearrange(consist_target, "1 c f h w -> f c h w")
                        
                        # Store the first fused prediction as geometry reference.
                        if self.reference_frames is None:
                            print(f"[INFO] Saving reference frames at step {progress_id} (fusion_weight={fusion_weight:.3f})")
                            self.reference_frames = consist_target_frames.detach().clone()
                        
                        # Geometry-only compensation relative to reference frames.
                        detail_diff_frames = self.reference_frames - consist_target_frames
                        
                        # Keep compensation small to avoid suppressing relighting.
                        detail_compensation_strength = 0.03 * fusion_weight
                        consist_target_frames = consist_target_frames + detail_compensation_strength * detail_diff_frames
                        consist_target_frames = consist_target_frames.clamp(-1, 1)
                        
                        print(f"[INFO] 🔧 Detail compensation: strength={detail_compensation_strength:.4f}, "
                              f"diff_range=[{detail_diff_frames.min().item():.3f}, {detail_diff_frames.max().item():.3f}]")
                        # Use shared noise across all frames for temporal consistency.
                        
                        # Generate one noise sample and repeat it over frames.
                        seed_generator = torch.Generator(device=self.device).manual_seed(seed if seed else 42)
                        one_frame_noise = torch.randn(
                            (1, 4, height // 8, width // 8), 
                            generator=seed_generator, 
                            device=self.device, 
                            dtype=self.relighter.vae.dtype
                        )
                        
                        # Repeat for all frames.
                        F = consist_target_frames.shape[0]
                        fixed_latents_for_relight = one_frame_noise.repeat(F, 1, 1, 1)
                        
                        print(f"[INFO] 🎲 Using shared noise for all {F} frames (shape: {fixed_latents_for_relight.shape})")
                        
                        # Pass shared noise via `latents` (not `init_latent`).
                        relight_target = self.relighter(
                            consist_target_frames, 
                            latents=fixed_latents_for_relight
                        )
                        relight_target = relight_target.clamp(-1, 1)
                        # relight_target = apply_global_light_consistency(relight_target)
                        # Global statistics matching reduces brightness pumping.
                        relight_target = global_histogram_matching(relight_target)
                        # Temporal smoothing on relighted frames.
                        # relight_target = adaptive_temporal_smooth(
                        #     relight_target, 
                        #     window_size=9,
                        #     sigma=18,
                        #     motion_thresh=0.3,
                        #     texture_smooth_ratio=0.3
                        # )
                        # relight_target = temporal_smooth_lighting(relight_target, window_size=7, sigma=20)
                        relight_target = temporal_smooth_lighting(relight_target, window_size=9, sigma=25)
                        relight_target = relight_target.clamp(-1, 1)
                        
                        # Collect intermediate results
                        if save_intermediate_steps:
                            # Take middle frame for grid visualization (convert to float32 for numpy compatibility)
                            mid_idx = consist_target_frames.shape[0] // 2
                            intermediate_frames['consist'].append(consist_target_frames[mid_idx].float().cpu())
                            intermediate_frames['relight'].append(relight_target[mid_idx].float().cpu())
                        
                        # Progressive fusion: consist + weight * (relight - consist)
                        progress_percent = progress_id / (num_inference_steps - 1) * 100
                        print(f"[DEBUG] Step {progress_id}/{num_inference_steps-1} ({progress_percent:.1f}%): fusion_weight={fusion_weight:.3f}")
                        print(f"[DEBUG] consist_target_frames range: min={consist_target_frames.min().item():.3f}, max={consist_target_frames.max().item():.3f}")
                        print(f"[DEBUG] relight_target       range: min={relight_target.min().item():.3f}, max={relight_target.max().item():.3f}")
                        
                        # Fusion equation.
                        fusion_target_frames = consist_target_frames + fusion_weight * (relight_target - consist_target_frames)
                        fusion_target = rearrange(fusion_target_frames, "f c h w -> () c f h w")
                        
                        # Collect fusion result
                        if save_intermediate_steps:
                            mid_idx = fusion_target_frames.shape[0] // 2
                            intermediate_frames['fusion'].append(fusion_target_frames[mid_idx].float().cpu())
                        
                        # Encode fusion target back to latent
                        fusion_latent = self.encode_video(fusion_target, **tiler_kwargs)
                        
                        # # Custom step with fusion target
                        # self.load_models_to_device(["dit"])
                        # latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
                        # latents = latents + lbd * 0.1 * (fusion_latent - latents)  # Blend with small weight

                        # Custom step with fusion target using flow matching
                        self.load_models_to_device(["dit"])
                        timestep = self.scheduler.timesteps[progress_id]
                        latents = flow_matching_fusion_step(
                            self.scheduler, 
                            noise_pred, 
                            timestep, 
                            latents, 
                            fusion_latent
                        )
                    else:
                        # Normal scheduler step
                        latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
                else:
                    # Normal scheduler step (no relighting)
                    latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
                
                # # Save latents for this step
                # if save_intermediate_steps:
                #     print("--------------------------------Save latents--------------------------------")
                #     intermediate_frames['no_relight'].append(latents.float().cpu().clone())
        '''        
        # Save intermediate results as grid (2 rows × 10 columns)
        if save_intermediate_steps and use_progressive_fusion and self.relighter is not None:
            if len(intermediate_frames['consist']) > 0:
                print(f"\nGenerating intermediate result grids...")
                self._save_intermediate_grids(intermediate_frames, intermediate_dir, num_inference_steps)
        
        # Save no-relight intermediate results for comparison
        if save_intermediate_steps and len(intermediate_frames['no_relight']) > 0:
            print(f"\nGenerating no-relight intermediate result grid...")
            self._save_intermediate_grids({'no_relight': intermediate_frames['no_relight']}, intermediate_dir, num_inference_steps)
        '''
        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
