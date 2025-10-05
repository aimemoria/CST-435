import numpy as np
import torch
from typing import Tuple
import librosa

# MoviePy-based audio extraction
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

def extract_audio_array(path: str, sr: int = 16000) -> np.ndarray:
    """Return mono audio as float32 numpy array at target sr using MoviePy."""
    clip = VideoFileClip(path)
    # MoviePy returns audio at its native sampling rate; resample with librosa
    audioclip = clip.audio
    if audioclip is None:
        return np.zeros(1, dtype=np.float32)
    # to_soundarray can be large; for short clips (classroom) it's okay
    audio = audioclip.to_soundarray(fps=sr)  # shape [T, channels] at target fps
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)

def log_mel_spectrogram(audio: np.ndarray, sr: int = 16000, n_mels: int = 64,
                        win_length: float = 0.025, hop_length: float = 0.010) -> torch.Tensor:
    """Compute log-mel spectrogram [1, Mels, T]."""
    n_fft = int(sr * win_length)
    hop = int(sr * hop_length)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    logS = librosa.power_to_db(S + 1e-10)
    logS = torch.from_numpy(logS).unsqueeze(0).float()   # [1,M,T]
    return logS
