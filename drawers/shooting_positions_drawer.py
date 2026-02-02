import cv2
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont


class ShootingPositionsDrawer:
    """Generates court image with shooting positions (accumulated across videos)."""
    
    def __init__(self, court_image_path: str, tactical_width: int = 300, tactical_height: int = 161):
        self.court_image_path = court_image_path
        self.tactical_width = tactical_width
        self.tactical_height = tactical_height
        self.accumulated_shots_path = "stubs/accumulated_shots.pkl"
        
    def _load_accumulated_shots(self) -> List[Dict]:
        """Load accumulated shots from pickle file."""
        if os.path.exists(self.accumulated_shots_path):
            try:
                with open(self.accumulated_shots_path, 'rb') as f:
                    shots = pickle.load(f)
                    print(f"Loaded {len(shots)} accumulated shots from {self.accumulated_shots_path}")
                    return shots
            except Exception as e:
                print(f"Error loading accumulated shots: {e}")
        return []
    
    def _save_accumulated_shots(self, shots: List[Dict]):
        """Save accumulated shots to pickle file."""
        with open(self.accumulated_shots_path, 'wb') as f:
            pickle.dump(shots, f)
        print(f"Saved {len(shots)} accumulated shots to {self.accumulated_shots_path}")
        
    def draw_shooting_positions(self, 
                                  shots: List,
                                  tactical_player_positions: List[Dict],
                                  output_path: str = "images/shooting_positions.pdf"):
        """Draw shooting positions on court (accumulated across runs)."""
        # Load court image
        court_img = cv2.imread(self.court_image_path)
        if court_img is None:
            raise FileNotFoundError(f"Cannot load court image from {self.court_image_path}")
        
        # Convert to RGB for PIL
        court_img = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)
        
        # Create PIL object for drawing
        pil_img = Image.fromarray(court_img)
        draw = ImageDraw.Draw(pil_img)
        
        # Calculate scale factors
        img_width, img_height = pil_img.size
        scale_x = img_width / self.tactical_width
        scale_y = img_height / self.tactical_height
        
        print(f"Court image size: {img_width}x{img_height}")
        print(f"Tactical system: {self.tactical_width}x{self.tactical_height}")
        print(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
        
        # Colors
        made_color = (0, 200, 0)
        missed_color = (200, 0, 0)
        
        # Load previous accumulated shots
        accumulated_shots = self._load_accumulated_shots()
        
        # Process new shots and add to accumulated
        for shot in shots:
            # Filter only Team 1 shots
            if shot.team_id != 1:
                continue
                
            # Find tactical position at shot frame
            frame_idx = shot.frame_start
            player_id = shot.player_id
            
            tactical_pos = None
            
            # Search for tactical position in shot frame or nearby frames
            for offset in range(0, 30):
                check_frame = frame_idx - offset
                if check_frame >= 0 and check_frame < len(tactical_player_positions):
                    frame_positions = tactical_player_positions[check_frame]
                    if player_id in frame_positions:
                        tactical_pos = frame_positions[player_id]
                        break
            
            if tactical_pos is None:
                print(f"Warning: No tactical position found for shot at frame {frame_idx}, player {player_id}")
                continue
            
            # Add to accumulated shots
            accumulated_shots.append({
                'tactical_x': tactical_pos[0],
                'tactical_y': tactical_pos[1],
                'made': shot.made,
                'team_id': shot.team_id,
                'player_id': player_id
            })
        
        # Save accumulated shots
        self._save_accumulated_shots(accumulated_shots)
        
        # Draw ALL accumulated shots
        team_stats = {'made': 0, 'missed': 0}
        
        for shot_data in accumulated_shots:
            # Scale tactical coordinates to image dimensions
            x = int(shot_data['tactical_x'] * scale_x)
            y = int(shot_data['tactical_y'] * scale_y)
            
            # Update stats
            if shot_data['made']:
                team_stats['made'] += 1
            else:
                team_stats['missed'] += 1
            
            # Draw symbol (scaled)
            symbol_size = int(8 * min(scale_x, scale_y))
            line_width = max(2, int(3 * min(scale_x, scale_y)))
            
            if shot_data['made']:
                # Green circle for made shots
                draw.ellipse(
                    [x - symbol_size, y - symbol_size, x + symbol_size, y + symbol_size],
                    outline=made_color,
                    width=line_width
                )
            else:
                # Red X for missed shots
                draw.line(
                    [x - symbol_size, y - symbol_size, x + symbol_size, y + symbol_size],
                    fill=missed_color,
                    width=line_width
                )
                draw.line(
                    [x - symbol_size, y + symbol_size, x + symbol_size, y - symbol_size],
                    fill=missed_color,
                    width=line_width
                )
        
        # Save as PDF
        if output_path.endswith('.pdf'):
            pil_img.save(output_path, "PDF", resolution=100.0)
        else:
            pil_img.save(output_path)
        
        print(f"Shooting positions exported to {output_path}")
        print(f"  Team 1 (cumulative): {team_stats['made']} made, {team_stats['missed']} missed")
        print(f"  Total accumulated shots: {len(accumulated_shots)}")
        
        return pil_img
        
        return pil_img
