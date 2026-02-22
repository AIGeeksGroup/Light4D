import torch
import torchvision.transforms.functional as TF


def temporal_smooth(video_tensor, window_size=3):
    """
    Apply temporal smoothing to video tensor using sliding window average.
    
    Args:
        video_tensor: [F, C, H, W] tensor
        window_size: size of temporal window (larger = smoother but slower response to dynamic lighting)
    
    Returns:
        smoothed tensor with same shape
    """
    f, c, h, w = video_tensor.shape
    half_window = window_size // 2
    
    # Replicate boundary frames to keep output length unchanged.
    pad_start = video_tensor[0:1].repeat(half_window, 1, 1, 1)
    pad_end = video_tensor[-1:].repeat(half_window, 1, 1, 1)
    padded = torch.cat([pad_start, video_tensor, pad_end], dim=0)
    
    smoothed = torch.zeros_like(video_tensor)
    
    for i in range(f):
        window = padded[i : i + window_size]
        smoothed[i] = torch.mean(window, dim=0)
    
    return smoothed

def apply_global_light_consistency(relight_target, reference_brightness=None):
    """
    Align overall brightness across frames by shifting per-frame mean.

    Args:
        relight_target: [F, C, H, W] in range [-1, 1]
        reference_brightness: target brightness; uses median if None

    Returns:
        adjusted tensor with consistent global brightness
    """
    F, C, H, W = relight_target.shape
    
    frame_brightness = relight_target.mean(dim=[1, 2, 3])
    
    if reference_brightness is None:
        reference_brightness = frame_brightness.median()
    
    adjusted = relight_target.clone()
    for f in range(F):
        brightness_diff = reference_brightness - frame_brightness[f]
        adjusted[f] = adjusted[f] + brightness_diff
    
    return adjusted

def temporal_smooth_lighting(video_tensor, window_size=5, sigma=15.0):
    """
    Temporally smooth lighting while preserving texture detail.

    Method:
        1) Extract a low-frequency lighting layer with Gaussian blur.
        2) Compute a high-frequency texture layer as residual.
        3) Smooth only the lighting layer over time.
        4) Recombine smoothed lighting with original texture.

    Args:
        video_tensor: [F, C, H, W] tensor in range [-1, 1]
        window_size: temporal window size (odd); typical values 3, 5, 7
        sigma: Gaussian blur radius for lighting extraction

    Returns:
        smoothed tensor with same shape
    """
    f, c, h, w = video_tensor.shape
    device = video_tensor.device
    dtype = video_tensor.dtype
    
    # Use a kernel large enough to cover the Gaussian support.
    kernel_size = int(sigma * 4) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    lighting_layer = torch.zeros_like(video_tensor)
    for i in range(f):
        lighting_layer[i] = TF.gaussian_blur(video_tensor[i:i+1], kernel_size=kernel_size, sigma=sigma)[0]
    
    texture_layer = video_tensor - lighting_layer
    
    half_window = window_size // 2
    pad_start = lighting_layer[0:1].repeat(half_window, 1, 1, 1)
    pad_end = lighting_layer[-1:].repeat(half_window, 1, 1, 1)
    padded_lighting = torch.cat([pad_start, lighting_layer, pad_end], dim=0)
    
    smoothed_lighting = torch.zeros_like(lighting_layer)
    
    for i in range(f):
        window = padded_lighting[i : i + window_size]
        smoothed_lighting[i] = torch.mean(window, dim=0)
    
    result = smoothed_lighting + texture_layer
    
    return result

def adaptive_temporal_smooth(video_tensor, window_size=5, sigma=20.0, motion_thresh=0.22, texture_smooth_ratio=0.2):
    """
    Adaptive temporal smoothing with motion-aware weights.

    Args:
        video_tensor: [F, C, H, W] tensor in range [-1, 1]
        window_size: temporal window size (odd), typically 5-9
        sigma: Gaussian blur radius for lighting extraction
        motion_thresh: sensitivity for motion-aware weighting
        texture_smooth_ratio: blend factor for optional texture smoothing

    Returns:
        smoothed tensor [F, C, H, W]
    """
    f, c, h, w = video_tensor.shape
    device = video_tensor.device
    dtype = video_tensor.dtype
    
    kernel_size = int(sigma * 4) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    lighting_layer = torch.zeros_like(video_tensor)
    for i in range(f):
        lighting_layer[i] = TF.gaussian_blur(video_tensor[i:i+1], kernel_size=kernel_size, sigma=sigma)[0]
    
    texture_layer = video_tensor - lighting_layer
    
    half_window = window_size // 2
    
    pad_start = lighting_layer[0:1].repeat(half_window, 1, 1, 1)
    pad_end = lighting_layer[-1:].repeat(half_window, 1, 1, 1)
    padded_lighting = torch.cat([pad_start, lighting_layer, pad_end], dim=0)
    
    smoothed_lighting = torch.zeros_like(lighting_layer)
    
    for i in range(f):
        window_frames = padded_lighting[i : i + window_size]
        center_frame = window_frames[half_window:half_window+1]
        
        diff = torch.abs(window_frames - center_frame).mean(dim=1, keepdim=True)
        
        weights = torch.exp(- (diff ** 2) / (2 * motion_thresh ** 2))
        
        weights_sum = weights.sum(dim=0, keepdim=True) + 1e-8
        normalized_weights = weights / weights_sum
        
        smoothed_frame = (window_frames * normalized_weights).sum(dim=0)
        smoothed_lighting[i] = smoothed_frame
    
    if texture_smooth_ratio > 0:
        pad_start_tex = texture_layer[0:1].repeat(half_window, 1, 1, 1)
        pad_end_tex = texture_layer[-1:].repeat(half_window, 1, 1, 1)
        padded_texture = torch.cat([pad_start_tex, texture_layer, pad_end_tex], dim=0)
        
        smoothed_texture = torch.zeros_like(texture_layer)
        
        for i in range(f):
            window_tex = padded_texture[i : i + window_size]
            center_tex = window_tex[half_window:half_window+1]
            
            diff_tex = torch.abs(window_tex - center_tex).mean(dim=1, keepdim=True)
            weights_tex = torch.exp(- (diff_tex ** 2) / (2 * (motion_thresh * 0.5) ** 2))
            
            weights_tex_sum = weights_tex.sum(dim=0, keepdim=True) + 1e-8
            normalized_weights_tex = weights_tex / weights_tex_sum
            
            smoothed_tex_frame = (window_tex * normalized_weights_tex).sum(dim=0)
            
            smoothed_texture[i] = (1 - texture_smooth_ratio) * texture_layer[i] + texture_smooth_ratio * smoothed_tex_frame
        
        texture_layer = smoothed_texture
    
    result = smoothed_lighting + texture_layer
    
    return result

def global_histogram_matching(video_tensor, reference_tensor=None):
    """
    Align per-frame statistics to a reference frame distribution.

    Args:
        video_tensor: [F, C, H, W] in range [-1, 1] or [0, 1]
    """
    if reference_tensor is None:
        reference_tensor = video_tensor.mean(dim=0, keepdim=True)
    
    f, c, h, w = video_tensor.shape
    matched_video = torch.zeros_like(video_tensor)
    
    # AdaIN-style mean/std matching for stability and speed.
    ref_mean = reference_tensor.view(1, c, -1).mean(dim=2, keepdim=True)
    ref_std = reference_tensor.view(1, c, -1).std(dim=2, keepdim=True) + 1e-6
    
    for i in range(f):
        frame = video_tensor[i]
        frame_flat = frame.view(c, -1)
        
        curr_mean = frame_flat.mean(dim=1, keepdim=True)
        curr_std = frame_flat.std(dim=1, keepdim=True) + 1e-6
        
        normalized = (frame_flat - curr_mean) / curr_std
        matched = normalized * ref_std + ref_mean
        
        matched_video[i] = matched.view(c, h, w)
        
    return matched_video
