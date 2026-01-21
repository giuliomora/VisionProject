import sys
import os

from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer

def main():
        
        #input
        video_frames = read_video("input_videos/video_1.mp4")

        #initialize and run tracker
        player_tracker = PlayerTracker(model_path="models/player_detector.pt")
        ball_tracker = BallTracker(model_path="models/ball_detector_model.pt")


        player_tracks = player_tracker.get_object_tracks(video_frames,
                                                         read_from_stub=True,
                                                         stub_path="stubs/player_tracks.stub.pkl"
                                                         )
        ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                        read_from_stub=True,
                                                        stub_path="stubs/ball_tracks.stub.pkl"
                                                        )

        # draw output
        player_tracks_drawer = PlayerTracksDrawer()
        ball_tracker_drawer = BallTracksDrawer()

        output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
        output_video_frames = ball_tracker_drawer.draw(output_video_frames, ball_tracks)

        #save
        save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()