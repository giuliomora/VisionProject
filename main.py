import os
import argparse
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from shooting_detector import ShootingDetector
from assist_detector import AssistDetector
from drawers import (
    PlayerTracksDrawer, 
    BallTracksDrawer,
    CourtKeypointDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    HoopDrawer,
    ShootingDrawer,
    ShootingPositionsDrawer,
    AssistDrawer
)
from configs import(
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('--input_video', type=str, default='input_videos/video_1.mp4',
                        help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Read video
    video_frames = read_video(args.input_video)
    
    # Initialize trackers
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    # Track players
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=os.path.join(args.stub_path, 'player_track_stubs.pkl')
                                      )
    
    # Track ball
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                 read_from_stub=True,
                                                 stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl')
                                                )
    
    # Track hoops
    hoop_tracks = ball_tracker.get_hoop_tracks(video_frames,
                                                read_from_stub=args.stub_path,
                                                stub_path=os.path.join(args.stub_path, "hoop_detections.pkl"))
    
    # Detect court keypoints
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                    read_from_stub=True,
                                                                    stub_path=os.path.join(args.stub_path, 'court_key_points_stub.pkl')
                                                                    )

    # Clean ball tracks
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                    player_tracks,
                                                                    read_from_stub=True,
                                                                    stub_path=os.path.join(args.stub_path, 'player_assignment_stub.pkl')
                                                                    )

    # Detect ball possession
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    # Detect passes and interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition, player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition, player_assignment)

    # Convert to tactical view
    tactical_view_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")
    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints_per_frame, player_tracks)

    # Detect shots
    shooting_detector = ShootingDetector(
        hoop_proximity_threshold=120,
        made_shot_threshold=80,
        min_frames_between_shots=60
    )
    shots = shooting_detector.detect_shots(
        ball_tracks,
        hoop_tracks,
        player_tracks,
        player_assignment,
        ball_aquisition
    )
    
    # Export shooting stats
    shooting_detector.export_to_tsv(os.path.join(args.stub_path, 'team_shooting_stats.tsv'))
    shooting_detector.export_shots_to_tsv(os.path.join(args.stub_path, 'all_shots.tsv'))
    
    # Generate shooting positions PDF
    shooting_positions_drawer = ShootingPositionsDrawer(court_image_path="./images/basketball_court.png")
    shooting_positions_drawer.draw_shooting_positions(
        shots=shots,
        tactical_player_positions=tactical_player_positions,
        output_path="./images/shooting_positions.pdf"
    )
    
    # Print shooting stats
    print(f"\nShooting Stats:")
    for team_id, stats in shooting_detector.get_team_stats().items():
        print(f"  Team {team_id}: {stats['made']}/{stats['attempts']} "
              f"({stats['made']/stats['attempts']*100:.1f}% FG)" if stats['attempts'] > 0 
              else f"  Team {team_id}: No shots")
    
    # Detect assists
    assist_detector = AssistDetector(max_frames_pass_to_shot=150)
    assists = assist_detector.detect_assists(
        passes=passes,
        shots=shots,
        ball_acquisition=ball_aquisition,
        player_assignment=player_assignment
    )
    
    # Export game stats
    assist_detector.export_to_csv(
        output_path=os.path.join(args.stub_path, 'game_stats.csv'),
        shots=shots,
        passes=passes,
        interceptions=interceptions,
        player_assignment=player_assignment,
        ball_acquisition=ball_aquisition
    )
    
    # Print assist stats
    print(f"\nAssist Stats:")
    for team_id, stats in assist_detector.get_team_stats().items():
        print(f"  Team {team_id}: {stats['total_assists']} assists")
        for player_id, count in stats['assists_by_player'].items():
            print(f"    - Player {player_id}: {count} assist(s)")
    
    # Initialize drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    hoop_drawer = HoopDrawer()
    shooting_drawer = ShootingDrawer()
    assist_drawer = AssistDrawer()

    # Draw player tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, 
                                                    player_tracks,
                                                    player_assignment,
                                                    ball_aquisition)
    
    # Draw ball tracks
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    # Draw court keypoints
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints_per_frame)

    # Draw frame number
    output_video_frames = frame_number_drawer.draw(output_video_frames)

    # Draw passes, interceptions, assists, shots
    output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
                                                             passes,
                                                             interceptions,
                                                             assists,
                                                             shots)
    
    # Draw tactical view
    output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                    tactical_view_converter.court_image_path,
                                                    tactical_view_converter.width,
                                                    tactical_view_converter.height,
                                                    tactical_view_converter.key_points,
                                                    tactical_player_positions,
                                                    player_assignment,
                                                    ball_aquisition,
                                                    )
    
    # Draw hoops
    output_video_frames = hoop_drawer.draw(output_video_frames, hoop_tracks)
    
    # Draw shots
    output_video_frames = shooting_drawer.draw(output_video_frames, shots)
    
    # Draw assists
    output_video_frames = assist_drawer.draw(output_video_frames, assists)

    # Save video
    save_video(output_video_frames, args.output_video)

if __name__ == '__main__':
    main()
    