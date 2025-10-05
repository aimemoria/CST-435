"""
Create a simple test video with visual and audio patterns
that can be annotated as 'explosion' (fast changes) or 'explanation' (calm).
"""
import numpy as np
import cv2
import soundfile as sf

def create_test_video():
    """Create a test video with explosion and explanation segments."""

    # Video parameters
    width, height = 320, 240
    fps = 24
    duration = 40  # seconds
    total_frames = fps * duration

    # Output path
    output_video = "videos/test_science_demo_temp.mp4"
    output_audio = "videos/test_audio.wav"
    output_final = "videos/test_science_demo.mp4"

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print("Creating video frames...")

    # Create video frames
    for frame_num in range(total_frames):
        t = frame_num / fps  # Current time in seconds

        # Explanation segments: calm blue gradient (0-5s, 15-20s, 30-35s)
        # Explosion segments: rapid red/orange flashing (5-8s, 20-25s, 35-40s)

        if (0 <= t < 5) or (15 <= t < 20) or (30 <= t < 35):
            # Calm explanation - slow blue gradient
            intensity = 100 + 50 * np.sin(t * 0.5)
            frame = np.ones((height, width, 3), dtype=np.uint8)
            frame[:, :, 2] = 50  # Low red (BGR format)
            frame[:, :, 1] = 100  # Medium green
            frame[:, :, 0] = int(intensity)  # Varying blue
        elif (5 <= t < 8) or (20 <= t < 25) or (35 <= t < 40):
            # Explosion - rapid flashing red/orange
            intensity = 200 + 55 * np.sin(t * 20)  # Fast oscillation
            frame = np.ones((height, width, 3), dtype=np.uint8)
            frame[:, :, 2] = int(intensity)  # High red (BGR)
            frame[:, :, 1] = int(intensity * 0.5)  # Medium orange
            frame[:, :, 0] = 30  # Low blue
        else:
            # Transition segments - calm green
            frame = np.ones((height, width, 3), dtype=np.uint8)
            frame[:, :] = [100, 120, 80]  # BGR

        out.write(frame)

    out.release()
    print(f"Video frames created: {output_video}")

    # Create audio
    print("Creating audio...")
    sample_rate = 44100
    audio_duration = duration
    t_audio = np.linspace(0, audio_duration, int(sample_rate * audio_duration))
    audio = np.zeros_like(t_audio)

    for i, t in enumerate(t_audio):
        if (0 <= t < 5) or (15 <= t < 20) or (30 <= t < 35):
            # Calm explanation - low frequency
            audio[i] = 0.1 * np.sin(2 * np.pi * 100 * t)
        elif (5 <= t < 8) or (20 <= t < 25) or (35 <= t < 40):
            # Explosion - high frequency
            audio[i] = 0.3 * np.sin(2 * np.pi * 800 * t)
        else:
            # Low ambience
            audio[i] = 0.05 * np.sin(2 * np.pi * 50 * t)

    sf.write(output_audio, audio, sample_rate)
    print(f"Audio created: {output_audio}")

    # Combine video and audio using ffmpeg
    print("Combining video and audio with ffmpeg...")
    import subprocess
    cmd = f'ffmpeg -y -i {output_video} -i {output_audio} -c:v libx264 -c:a aac -strict experimental {output_final}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Final video created successfully: {output_final}")
        # Clean up temp files
        import os
        os.remove(output_video)
        os.remove(output_audio)
    else:
        print("FFmpeg not available. Using video without audio.")
        import os
        os.rename(output_video, output_final)

    print("Done!")

if __name__ == "__main__":
    create_test_video()
