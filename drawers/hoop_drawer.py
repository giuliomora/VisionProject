import cv2
import numpy as np
from typing import List, Dict

class HoopDrawer:
    """Draws detected hoops on frames."""
    
    def __init__(self, color=(0, 165, 255), thickness=3):
        self.color = color
        self.thickness = thickness
    
    def draw(self, frames: List[np.ndarray], hoop_tracks: List[Dict]) -> List[np.ndarray]:
        """Draw hoops on frames."""
        output_frames = []
        
        for frame_idx, frame in enumerate(frames):
            annotated_frame = frame.copy()
            
            if frame_idx < len(hoop_tracks):
                hoops = hoop_tracks[frame_idx]
                
                for hoop_id, hoop_data in hoops.items():
                    bbox = hoop_data.get('bbox')
                    if bbox is None:
                        continue
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.color, self.thickness)
                    
                    # Draw label
                    label = f"Hoop {hoop_id}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Text background
                    cv2.rectangle(annotated_frame, 
                                  (x1, y1 - label_size[1] - 10), 
                                  (x1 + label_size[0] + 5, y1), 
                                  self.color, -1)
                    
                    # Text
                    cv2.putText(annotated_frame, label,
                                (x1 + 2, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                    
                    # Draw hoop center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            output_frames.append(annotated_frame)
        
        return output_frames