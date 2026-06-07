import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.video_detection import detect_falls_from_video

if __name__ == "__main__":
    detect_falls_from_video(
        video_path="data/video/input/fall_video.mp4",
        output_path="data/video/results.json",
        save_frames=True
        
    )