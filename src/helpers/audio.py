import numpy as np
import torch
from typing import Tuple
import librosa
import av
import io

def extract_audio_array(path: str, sr: int = 16000) -> np.ndarray:
    """Return mono audio as float32 numpy array at target sr using PyAV."""
    try:
        container = av.open(path)
        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

        if audio_stream is None:
            container.close()
            return np.zeros(1, dtype=np.float32)

        audio_frames = []
        for frame in container.decode(audio_stream):
            array = frame.to_ndarray()
            audio_frames.append(array)

        container.close()

        if not audio_frames:
            return np.zeros(1, dtype=np.float32)

        # Concatenate all frames
        audio = np.concatenate(audio_frames, axis=1)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0)
        else:
            audio = audio[0]

        # Resample to target sr using librosa
        original_sr = audio_stream.rate
        if original_sr != sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)

        return audio.astype(np.float32)
    except Exception as e:
        # If audio extraction fails, return zeros
        return np.zeros(1, dtype=np.float32)

def log_mel_spectrogram(audio: np.ndarray, sr: int = 16000, n_mels: int = 64,
                        win_length: float = 0.025, hop_length: float = 0.010) -> torch.Tensor:
    """Compute log-mel spectrogram [1, Mels, T]."""
    n_fft = int(sr * win_length)
    hop = int(sr * hop_length)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    logS = librosa.power_to_db(S + 1e-10)
    logS = torch.from_numpy(logS).unsqueeze(0).float()   # [1,M,T]
    return logS
