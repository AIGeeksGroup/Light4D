import cv2
import numpy as np
import torch


def load_video_frames(video_path, num_frames, height, width):
    """Load and process video frames"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < num_frames:
        print(f"Warning: Video has only {frame_count} frames, but {num_frames} required")
    
    # Get frame indices
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        indices = np.arange(frame_count)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    # Pad or trim frames to match num_frames
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    else:
        frames = frames[:num_frames]
    
    frames = np.array(frames)
    video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    return video_tensor

def load_mask_frames(mask_path, num_frames, height, width):
    """Load and process mask frames"""
    cap = cv2.VideoCapture(mask_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < num_frames:
        print(f"Warning: Mask video has only {frame_count} frames, but {num_frames} required")
    
    # Get frame indices
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        indices = np.arange(frame_count)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            # Apply random erosion
            import random
            kernel = random.randint(2, 8)
            frame = cv2.erode(frame, np.ones((kernel, kernel), np.uint8), iterations=1)
            frames.append(frame)
    
    cap.release()
    
    # Pad or trim frames to match num_frames
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    else:
        frames = frames[:num_frames]
    
    frames = np.array(frames)
    mask_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    mask_tensor = (mask_tensor / 255.0 > 0.5).float()  # Binarize mask
    return mask_tensor
