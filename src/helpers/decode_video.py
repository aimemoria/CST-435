import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import torch
import cv2

def get_video_info(path: str) -> Tuple[float, float, Tuple[int,int]]:
    """Return (duration_sec, fps, (width, height))."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return float(duration), float(fps), (width, height)

def sample_frames(path: str, fps: int = 4, resize: int = 224) -> List[torch.Tensor]:
    """Sample RGB frames at a lower FPS and return normalized torch tensors [3,H,W]."""
    cap = cv2.VideoCapture(path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(round(video_fps / fps)), 1)
    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).convert("RGB")
            if resize:
                img = img.resize((resize, resize))
            arr = np.asarray(img).astype("float32") / 255.0
            # Normalize roughly like ImageNet (mean/std). Keep it simple here.
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            arr = np.transpose(arr, (2,0,1))  # [3,H,W]
            frames.append(torch.from_numpy(arr))
        idx += 1

    cap.release()
    return frames

def frame_diff_stack(frames: List[torch.Tensor], k: int = 3) -> torch.Tensor:
    """Return a single tensor stacking k consecutive frame differences along channel dim.
    If not enough frames, we pad by repeating the last diff.
    Output shape: [3*k, H, W]
    """
    if len(frames) < 2:
        x = frames[0] if frames else torch.zeros(3,224,224)
        return x.repeat(k,1,1)

    diffs = []
    for i in range(1, len(frames)):
        diffs.append(frames[i] - frames[i-1])
    # pick k most recent diffs
    if len(diffs) < k:
        diffs += [diffs[-1]] * (k - len(diffs))
    else:
        diffs = diffs[-k:]
    return torch.cat(diffs, dim=0)  # [3*k,H,W]
