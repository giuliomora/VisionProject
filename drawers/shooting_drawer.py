import cv2
import numpy as np
from typing import List, Dict, Optional

class ShootingDrawer:
    """Draws shooting events on frames."""
    
    def __init__(self, 
                 made_color=(0, 255, 0),
                 missed_color=(0, 0, 255),
                 display_frames: int = 60):
        self.made_color = tuple(int(c) for c in made_color)
        self.missed_color = tuple(int(c) for c in missed_color)
        self.display_frames = display_frames
    
    def draw(self, frames: List[np.ndarray], shots: List) -> List[np.ndarray]:
        """Draw shooting events on frames."""
        output_frames = []
        
        # Create lookup dict: frame_idx -> active shots
        active_shots = {}
        for shot in shots:
            for frame_idx in range(shot.frame_start, min(shot.frame_start + self.display_frames, len(frames))):
                if frame_idx not in active_shots:
                    active_shots[frame_idx] = []
                active_shots[frame_idx].append(shot)
        
        for frame_idx, frame in enumerate(frames):
            annotated_frame = frame.copy()
            
            if frame_idx in active_shots:
                for shot in active_shots[frame_idx]:
                    self._draw_shot_event(annotated_frame, shot, frame_idx)
            
            output_frames.append(annotated_frame)
        
        return output_frames
    
    def _draw_shot_event(self, frame: np.ndarray, shot, current_frame: int):
        """Draw single shot event on frame."""
        color = self.made_color if shot.made else self.missed_color
        result_text = "MADE!" if shot.made else "MISSED"
        
        # Calculate opacity (fade out)
        frames_since_shot = current_frame - shot.frame_start
        opacity = max(0.3, 1.0 - (frames_since_shot / self.display_frames) * 0.7)
        
        # Top banner
        banner_height = 68
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), color, -1)
        cv2.addWeighted(overlay, opacity * 0.7, frame, 1 - opacity * 0.7, 0, frame)
        
        # Main text
        main_text = f"SHOT - Team {shot.team_id}"
        if shot.player_id != -1:
            main_text += f" (Player {shot.player_id})"
        
        cv2.putText(frame, main_text,
                    (17, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)
        
        cv2.putText(frame, result_text,
                    (17, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (255, 255, 255), 2)
        
        # Draw circle at shot position
        if shot.position and shot.position != (0, 0):
            pos_x, pos_y = int(shot.position[0]), int(shot.position[1])
            
            # Pulsating circle
            pulse = int(10 + 5 * np.sin(frames_since_shot * 0.3))
            cv2.circle(frame, (pos_x, pos_y), pulse + 20, color, 3)
            cv2.circle(frame, (pos_x, pos_y), 8, color, -1)
            
            # Arrow pointing up (shot direction)
            cv2.arrowedLine(frame, 
                           (pos_x, pos_y), 
                           (pos_x, pos_y - 50),
                           color, 2, tipLength=0.3)