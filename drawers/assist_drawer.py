import cv2
import numpy as np
from typing import List

class AssistDrawer:
    """Draws assists on video frames."""
    
    def __init__(self, display_duration: int = 60):
        """Initialize with display duration in frames."""
        self.display_duration = display_duration
    
    def draw(self, video_frames: List, assists: List) -> List:
        """Draw assists on video frames."""
        output_video_frames = []
        
        # Create display schedule for assists
        assist_display = {}
        for assist in assists:
            for frame in range(assist.frame, min(assist.frame + self.display_duration, len(video_frames))):
                if frame not in assist_display:
                    assist_display[frame] = []
                assist_display[frame].append(assist)
        
        for frame_num, frame in enumerate(video_frames):
            frame_drawn = frame.copy()
            
            # Draw cumulative assist stats
            frame_drawn = self.draw_assist_stats(frame_drawn, frame_num, assists)
            
            # Draw active assist notifications
            if frame_num in assist_display:
                for assist in assist_display[frame_num]:
                    frame_drawn = self.draw_assist_notification(
                        frame_drawn, 
                        assist, 
                        frame_num
                    )
            
            output_video_frames.append(frame_drawn)
        
        return output_video_frames
    
    def draw_assist_notification(self, frame, assist, current_frame: int):
        """Draw assist notification on frame."""
        overlay = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate fade out opacity
        frames_since_pass = current_frame - assist.frame
        alpha = max(0.3, 1.0 - (frames_since_pass / self.display_duration) * 0.7)
        
        # Notification box centered at top
        box_width = 350
        box_height = 80
        box_x1 = (frame_width - box_width) // 2
        box_y1 = 20
        box_x2 = box_x1 + box_width
        box_y2 = box_y1 + box_height
        
        # Team color
        if assist.team_id == 1:
            color = (255, 150, 0)
            text_color = (255, 255, 255)
        else:
            color = (0, 100, 255)
            text_color = (255, 255, 255)
        
        # Draw box with border
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), color, -1)
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)
        
        # Text
        cv2.putText(
            overlay,
            "ASSIST!",
            (box_x1 + 100, box_y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            text_color,
            2
        )
        
        cv2.putText(
            overlay,
            f"Player {assist.passer_id} -> Player {assist.scorer_id}",
            (box_x1 + 50, box_y1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )
        
        # Apply overlay with transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def draw_assist_stats(self, frame, frame_num: int, assists: List):
        """Assist stats are drawn in PassInterceptionDrawer unified box."""
        return frame
