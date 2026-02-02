import cv2
import numpy as np

class PassInterceptionDrawer:
    """Draws passes and interceptions on video frames."""
    
    def __init__(self):
        pass

    def get_stats(self, passes, interceptions):
        """Calculate passes and interceptions for both teams."""
        team1_passes = []
        team2_passes = []
        team1_interceptions = []
        team2_interceptions = []

        for frame_num, (pass_frame, interception_frame) in enumerate(zip(passes, interceptions)):
            if pass_frame == 1:
                team1_passes.append(frame_num)
            elif pass_frame == 2:
                team2_passes.append(frame_num)
                
            if interception_frame == 1:
                team1_interceptions.append(frame_num)
            elif interception_frame == 2:
                team2_interceptions.append(frame_num)
                
        return len(team1_passes), len(team2_passes), len(team1_interceptions), len(team2_interceptions)

    def draw(self, video_frames, passes, interceptions, assists=None, shots=None):
        """Draw stats on frames."""
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue
            
            frame_drawn = self.draw_frame(frame, frame_num, passes, interceptions, assists, shots)
            output_video_frames.append(frame_drawn)
        return output_video_frames
    
    def draw_frame(self, frame, frame_num, passes, interceptions, assists=None, shots=None):
        """Draw interception, assist and shot stats on frame."""
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        font_scale = 0.45
        font_thickness = 1

        # Overlay Position - basso a sinistra, piccolo (solo Team 1)
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = 10
        rect_y1 = frame_height - 55
        rect_x2 = 220
        rect_y2 = frame_height - 10
        
        # Text positions
        text_x = 15
        text_y_start = frame_height - 38

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Get stats until current frame
        interceptions_till_frame = interceptions[:frame_num+1]
        
        _, _, team1_interceptions, _ = self.get_stats(
            [0] * len(interceptions_till_frame),
            interceptions_till_frame
        )
        
        # Count assists per team
        team1_assists = 0
        if assists is not None:
            team1_assists = sum(1 for a in assists if a.frame <= frame_num and a.team_id == 1)
        
        # Count made/missed shots for team 1
        team1_made = 0
        team1_missed = 0
        if shots is not None:
            team1_made = sum(1 for s in shots if s.frame_start <= frame_num and s.team_id == 1 and s.made)
            team1_missed = sum(1 for s in shots if s.frame_start <= frame_num and s.team_id == 1 and not s.made)

        # Title
        cv2.putText(
            frame, 
            "TEAM 1 STATS",
            (text_x + 45, text_y_start), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.45, 
            (0, 0, 0), 
            1
        )
        
        # Team 1: I (intercetti), A (assist), Done (fatti), Missed (sbagliati)
        cv2.putText(
            frame, 
            f"I:{team1_interceptions} A:{team1_assists} Done:{team1_made} Missed:{team1_missed}",
            (text_x, text_y_start + 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 150, 0),  # Blu
            font_thickness
        )

        return frame