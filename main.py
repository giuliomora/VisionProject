import os
import sys
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import (PlayerTracksDrawer,
                    BallTracksDrawer,
                    PassInterceptionDrawer,
                    TeamBallControlDrawer
                    )
from team_assigner import TeamAssigner
from ball_acquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector

def main():
        
        #input
        video_frames = read_video("input_videos/video_2.mp4")

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

        #remove wrong ball detections
        ball_tracks = ball_tracker.remuve_wrong_detections(ball_tracks)
        
        #interpolation
        ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
        
        # Team assignment using color-based clustering (no CLIP model)
        team_assigner = TeamAssigner()
        player_assignement = team_assigner.get_player_teams_across_frames(video_frames,
                                                                    player_tracks,
                                                                 read_from_stub=True,
                                                                    stub_path="stubs/player_assignement_stub.pkl"
                                                                    )
        
        #ball acquisition
        ball_aquisition_detector = BallAquisitionDetector()
        ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

        #detect passes and interceptions
        pass_and_interception_detector = PassAndInterceptionDetector()
        passes = pass_and_interception_detector.detect_passes(ball_aquisition, player_assignement)
        interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition, player_assignement)
        # draw output
        player_tracks_drawer = PlayerTracksDrawer()
        ball_tracker_drawer = BallTracksDrawer()
        pass_and_interceptions_drawer = PassInterceptionDrawer()

        output_video_frames = player_tracks_drawer.draw(video_frames, 
                                                        player_tracks, 
                                                        player_assignement,
                                                        ball_aquisition
                                                        )
        
        output_video_frames = ball_tracker_drawer.draw(output_video_frames, ball_tracks)

        # team ball control drawer
        team_ball_control_drawer = TeamBallControlDrawer()
        output_video_frames = team_ball_control_drawer.draw(output_video_frames, 
                                                                          player_assignement, 
                                                                          ball_aquisition)
        
        # pass and interception drawer
        output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
                                                                 passes,
                                                                 interceptions)

        #save
        save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()