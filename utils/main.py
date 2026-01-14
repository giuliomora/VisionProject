import sys
import os
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_utils import read_video, save_video

def main():
    try:
        print("Inizio lettura video...")
        video_frames= read_video("input_videos/video_1.mp4")
        print(f"Video letto con successo: {len(video_frames)} frame")
        
        print("Inizio salvataggio video...")
        save_video(video_frames, "output_videos/saved_video.avi")
        print("Video salvato con successo!")
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()