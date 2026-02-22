import os
import sys
from types import MethodType

import safetensors.torch as sf
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from peft import LoraConfig, inject_adapter_in_model
from torch.hub import download_url_to_file
from transformers import CLIPTextModel, CLIPTokenizer

from diffsynth import load_state_dict


light_a_video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Light-A-Video'))
if light_a_video_path in sys.path:
    sys.path.remove(light_a_video_path)
sys.path.insert(0, light_a_video_path)
from src.ic_light_pipe import StableDiffusionImg2ImgPipeline


def add_lora_to_model(model, lora_rank=16, lora_alpha=16.0, lora_target_modules="q,k,v,o,ffn.0,ffn.2", 
                     init_lora_weights="kaiming", pretrained_path=None, state_dict_converter=None):
    """Add LoRA to model"""
    if init_lora_weights == "kaiming":
        init_lora_weights = True
        
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )
    model = inject_adapter_in_model(lora_config, model)
    
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    if pretrained_path is not None:
        state_dict = load_state_dict(pretrained_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(f"LORA: {num_updated_keys} parameters loaded from {pretrained_path}. {num_unexpected_keys} unexpected.")

def setup_ic_light_pipeline(sd_model_path, ic_light_model_path, device, dtype, gamma=0.7):
    """Setup IC-Light pipeline with CLA (Consistent Light Attention)"""
    print("Loading IC-Light components...")
    
    # Load base SD components
    tokenizer = CLIPTokenizer.from_pretrained(sd_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(sd_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd_model_path, subfolder="unet")
    
    # Modify UNet conv_in to accept 8 channels (4 latent + 4 light condition)
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, 
                                      unet.conv_in.kernel_size, 
                                      unet.conv_in.stride, 
                                      unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in
    
    # Hook UNet forward to concatenate light conditions
    unet_original_forward = unet.forward
    
    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
    
    unet.forward = hooked_unet_forward
    
    # Load IC-Light weights
    if not os.path.exists(ic_light_model_path):
        print("Downloading IC-Light model...")
        os.makedirs(os.path.dirname(ic_light_model_path), exist_ok=True)
        download_url_to_file(
            url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors',
            dst=ic_light_model_path
        )
    
    sd_offset = sf.load_file(ic_light_model_path)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged
    
    # Move to device
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    vae = vae.to(device=device, dtype=dtype)
    unet = unet.to(device=device, dtype=dtype)
    
    # Set attention processors
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())
    
    # Add Consistent Light Attention (CLA) with Local Temporal Smoothing
    print(f"Adding Consistent Light Attention (Local Temporal Smoothing) with gamma={gamma}")
    

    @torch.inference_mode()
    def custom_forward_CLA(self, 
                        hidden_states, 
                        encoder_hidden_states=None,
                        attention_mask=None, 
                        cross_attention_kwargs=None
                        ):
        """
        Consistent Light Attention with Local Temporal Smoothing (3-frame window).
        
        Key improvements over global average:
        1. Uses sliding window (prev, curr, next frames) instead of global mean
        2. Preserves temporal dynamics while reducing high-frequency flicker
        3. Allows gradual lighting changes over time
        
        Weights: w_curr=0.6, w_neighbor=0.2 each
        - Adjustable: higher w_curr = more responsive, lower = smoother
        

        """
        
        batch_size, sequence_length, channel = hidden_states.shape
        
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        # Compute Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)   
        value = self.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        
        hidden_states_orig = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        
        n_groups = 2  # CFG dimension
        if batch_size % n_groups != 0:
            n_groups = 1
        
        n_frames = batch_size // n_groups
        
        b, h, s, d = key.shape
        key_video = key.view(n_groups, n_frames, h, s, d)
        value_video = value.view(n_groups, n_frames, h, s, d)
        
        k_prev = torch.roll(key_video, shifts=1, dims=1)
        k_next = torch.roll(key_video, shifts=-1, dims=1)
        k_prev[:, 0] = key_video[:, 0]
        k_next[:, -1] = key_video[:, -1]
        
        v_prev = torch.roll(value_video, shifts=1, dims=1)
        v_next = torch.roll(value_video, shifts=-1, dims=1)
        v_prev[:, 0] = value_video[:, 0]
        v_next[:, -1] = value_video[:, -1]
        
        w_curr = 0.6
        w_neighbor = (1 - w_curr) / 2
        
        smooth_key = w_curr * key_video + w_neighbor * (k_prev + k_next)
        smooth_value = w_curr * value_video + w_neighbor * (v_prev + v_next)
        
        smooth_key = smooth_key.view(b, h, s, d)
        smooth_value = smooth_value.view(b, h, s, d)
        
        hidden_states_consist = F.scaled_dot_product_attention(
            query, smooth_key, smooth_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = (1 - gamma) * hidden_states_orig + gamma * hidden_states_consist
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if self.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / self.rescale_output_factor
        
        return hidden_states
 
    # Apply CLA to specific attention layers
    num_cla_applied = 0
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        name_split_list = name.split(".")
        
        cond_1 = name_split_list[0] in "up_blocks"
        cond_2 = name_split_list[-1] in ('attn1')
        
        if "Attention" in module_name and cond_1 and cond_2:
            cond_3 = name_split_list[1] 
            if cond_3 not in "3":
                module.forward = MethodType(custom_forward_CLA, module)
                num_cla_applied += 1
    
    print(f"Added CLA to {num_cla_applied} attention layers")
    
    # Create IC-Light scheduler and pipeline
    ic_light_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1
    )
    
    ic_light_pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=ic_light_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )
    
    ic_light_pipe = ic_light_pipe.to(device=device, dtype=dtype)
    ic_light_pipe.vae.requires_grad_(False)
    ic_light_pipe.unet.requires_grad_(False)
    
    return ic_light_pipe
