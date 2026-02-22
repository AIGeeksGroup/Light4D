import torch


def flow_matching_fusion_step(scheduler, noise_pred, timestep, sample, fusion_sample):
    """
    Flow matching step adapted for EX-4D's FlowMatchScheduler.
    Replaces standard denoising with fusion-guided flow.
    
    Args:
        scheduler: FlowMatchScheduler instance
        noise_pred: model prediction (not used in flow matching)
        timestep: current timestep
        sample: current latent sample
        fusion_sample: target fusion latent
    
    Returns:
        prev_sample: updated latent
    """
    # Find timestep index
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.cpu()
    timestep_id = torch.argmin((scheduler.timesteps - timestep).abs())
    
    # Get current and next sigma
    sigma = scheduler.sigmas[timestep_id]
    if timestep_id + 1 >= len(scheduler.timesteps):
        sigma_next = torch.tensor(0.0)
    else:
        sigma_next = scheduler.sigmas[timestep_id + 1]
    
    # Compute fusion direction: from current sample to fusion target
    # Upcast to avoid precision issues
    sample = sample.to(torch.float32)
    fusion_sample = fusion_sample.to(torch.float32)
    
    fusion_vector = (sample - fusion_sample) / (sigma + 1e-8)
    prev_sample = sample + (sigma_next - sigma) * fusion_vector
    
    # Cast back to original dtype
    prev_sample = prev_sample.to(sample.dtype)
    
    return prev_sample
