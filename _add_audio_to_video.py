"""Add audio track to the test video using MoviePy."""
import numpy as np
try:
    from moviepy import VideoFileClip, AudioClip
except ImportError:
    from moviepy.editor import VideoFileClip, AudioClip

def make_audio(t):
    """Create audio with different frequencies for explosion vs explanation."""
    # t can be scalar or array
    t_arr = np.atleast_1d(t)
    result = np.zeros_like(t_arr)

    # Explanation segments: calm low frequency
    mask_explanation = ((0 <= t_arr) & (t_arr < 5)) | ((15 <= t_arr) & (t_arr < 20)) | ((30 <= t_arr) & (t_arr < 35))
    result[mask_explanation] = 0.1 * np.sin(2 * np.pi * 100 * t_arr[mask_explanation])

    # Explosion segments: high frequency
    mask_explosion = ((5 <= t_arr) & (t_arr < 8)) | ((20 <= t_arr) & (t_arr < 25)) | ((35 <= t_arr) & (t_arr < 40))
    result[mask_explosion] = 0.3 * np.sin(2 * np.pi * 800 * t_arr[mask_explosion])

    # Transition/other: low ambience
    mask_other = ~(mask_explanation | mask_explosion)
    result[mask_other] = 0.05 * np.sin(2 * np.pi * 50 * t_arr[mask_other])

    return result

# Load the video
print("Loading video...")
video = VideoFileClip("videos/test_science_demo.mp4")

# Create audio clip with same duration
print("Creating audio...")
audio = AudioClip(make_audio, duration=video.duration, fps=44100)

# Set audio to video
print("Combining...")
final = video.with_audio(audio)

# Write output
print("Writing final video with audio...")
final.write_videofile("videos/test_science_demo_with_audio.mp4",
                     codec='libx264', audio_codec='aac')

print("Done! Video saved as videos/test_science_demo_with_audio.mp4")

# Rename to original name
import os
os.remove("videos/test_science_demo.mp4")
os.rename("videos/test_science_demo_with_audio.mp4", "videos/test_science_demo.mp4")
print("Renamed to test_science_demo.mp4")
