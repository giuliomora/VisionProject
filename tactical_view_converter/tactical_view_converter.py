import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import get_foot_position,measure_distance

class TacticalViewConverter:
    """Converts video coordinates to tactical court view using homography."""
    
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300
        self.height= 161

        self.actual_width_in_meters=28
        self.actual_height_in_meters=15 

        # Court keypoints in tactical view coordinates
        self.key_points = [
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height)),
            (0,int((5.18/self.actual_height_in_meters)*self.height)),
            (0,int((10/self.actual_height_in_meters)*self.height)),
            (0,int((14.1/self.actual_height_in_meters)*self.height)),
            (0,int(self.height)),

            (int(self.width/2),self.height),
            (int(self.width/2),0),
            
            (int((5.79/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),

            (self.width,int(self.height)),
            (self.width,int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width,int((10/self.actual_height_in_meters)*self.height)),
            (self.width,int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width,int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width,0),

            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),
        ]

    def validate_keypoints(self, keypoints_list):
        """Validate keypoints by comparing proportions and removing overlaps."""
        keypoints_list = deepcopy(keypoints_list)
        
        # Mirrored keypoints between court halves
        mirrored_keypoints = [
            (0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (8, 16), (9, 17),
        ]
        
        # Court half indices
        left_half_indices = [0, 1, 2, 3, 4, 5, 8, 9]
        right_half_indices = [10, 11, 12, 13, 14, 15, 16, 17]
        
        overlap_threshold = 50

        # Phase 1: Count keypoints per court half
        frame_votes = []
        
        for frame_keypoints in keypoints_list:
            frame_keypoints_xy = frame_keypoints.xy.tolist()[0]
            
            left_count = sum(1 for i in left_half_indices 
                            if i < len(frame_keypoints_xy) and 
                            frame_keypoints_xy[i][0] > 0 and frame_keypoints_xy[i][1] > 0)
            right_count = sum(1 for i in right_half_indices 
                             if i < len(frame_keypoints_xy) and 
                             frame_keypoints_xy[i][0] > 0 and frame_keypoints_xy[i][1] > 0)
            
            frame_votes.append((left_count, right_count))
        
        # Phase 2: Determine stable court half with persistent state
        min_frames_to_switch = 30
        
        stable_halves = []
        current_stable_half = 'none'
        consecutive_different = 0
        pending_half = 'none'
        
        for frame_idx, (left_count, right_count) in enumerate(frame_votes):
            # Determine frame half
            if left_count > right_count:
                frame_half = 'left'
            elif right_count > left_count:
                frame_half = 'right'
            elif left_count > 0:
                frame_half = 'both'
            else:
                frame_half = 'none'
            
            # Initialize stable half
            if current_stable_half == 'none':
                current_stable_half = frame_half
                stable_halves.append(current_stable_half)
                continue
            
            # Reset counter if same as stable half
            if frame_half == current_stable_half or frame_half == 'none' or frame_half == 'both':
                consecutive_different = 0
                pending_half = 'none'
                stable_halves.append(current_stable_half)
            else:
                # Track consecutive different frames
                if pending_half == frame_half:
                    consecutive_different += 1
                else:
                    pending_half = frame_half
                    consecutive_different = 1
                
                # Switch half if enough consecutive frames
                if consecutive_different >= min_frames_to_switch:
                    current_stable_half = pending_half
                    consecutive_different = 0
                    pending_half = 'none'
                
                stable_halves.append(current_stable_half)
        
        # Phase 3: Apply decision and clean keypoints
        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            frame_keypoints_xy = frame_keypoints.xy.tolist()[0]
            stable_half = stable_halves[frame_idx]
            
            if stable_half == 'left':
                # Remove right half keypoints
                for i in right_half_indices:
                    if i < len(frame_keypoints_xy):
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
            elif stable_half == 'right':
                # Remove left half keypoints
                for i in left_half_indices:
                    if i < len(frame_keypoints_xy):
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
            else:
                # Check overlaps only
                for left_idx, right_idx in mirrored_keypoints:
                    if left_idx >= len(frame_keypoints_xy) or right_idx >= len(frame_keypoints_xy):
                        continue
                    left_kp = frame_keypoints_xy[left_idx]
                    right_kp = frame_keypoints_xy[right_idx]
                    
                    if left_kp[0] <= 0 or left_kp[1] <= 0:
                        continue
                    if right_kp[0] <= 0 or right_kp[1] <= 0:
                        continue
                    
                    dist = measure_distance(left_kp, right_kp)
                    
                    if dist < overlap_threshold:
                        left_count, right_count = frame_votes[frame_idx]
                        if left_count <= right_count:
                            keypoints_list[frame_idx].xy[0][left_idx] *= 0
                            keypoints_list[frame_idx].xyn[0][left_idx] *= 0
                        else:
                            keypoints_list[frame_idx].xy[0][right_idx] *= 0
                            keypoints_list[frame_idx].xyn[0][right_idx] *= 0
            
            # Phase 4: Proportion-based validation
            frame_keypoints_xy = keypoints_list[frame_idx].xy.tolist()[0]
            detected_indices = [i for i, kp in enumerate(frame_keypoints_xy) if kp[0] > 0 and kp[1] > 0]
            
            if len(detected_indices) < 3:
                continue
            
            invalid_keypoints = []
            for i in detected_indices:
                if frame_keypoints_xy[i][0] == 0 and frame_keypoints_xy[i][1] == 0:
                    continue

                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                j, k = other_indices[0], other_indices[1]

                d_ij = measure_distance(frame_keypoints_xy[i], frame_keypoints_xy[j])
                d_ik = measure_distance(frame_keypoints_xy[i], frame_keypoints_xy[k])
                
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = (prop_detected - prop_tactical) / prop_tactical
                    error = abs(error)

                    if error > 0.8:
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)
            
        return keypoints_list

    def _correct_position_by_nearest_keypoint(self, player_pos, detected_keypoints, valid_indices, tactical_position):
        """Correct tactical position based on nearest keypoints."""
        if len(valid_indices) == 0:
            return tactical_position
        
        corrected_x = tactical_position[0]
        corrected_y = tactical_position[1]
        
        # Y correction: find nearest vertical keypoint
        min_y_diff = float('inf')
        nearest_y_idx = None
        
        for i in valid_indices:
            kp_video = detected_keypoints[i]
            y_diff = abs(player_pos[1] - kp_video[1])
            if y_diff < min_y_diff:
                min_y_diff = y_diff
                nearest_y_idx = i
        
        # Apply Y correction if close to keypoint
        if nearest_y_idx is not None and min_y_diff < 30:
            nearest_tactical_y = self.key_points[nearest_y_idx][1]
            weight_y = 1.0 - (min_y_diff / 30.0)
            corrected_y = tactical_position[1] * (1 - weight_y) + nearest_tactical_y * weight_y
        
        # X correction: find nearest horizontal keypoint
        min_x_diff = float('inf')
        nearest_x_idx = None
        
        for i in valid_indices:
            kp_video = detected_keypoints[i]
            x_diff = abs(player_pos[0] - kp_video[0])
            if x_diff < min_x_diff:
                min_x_diff = x_diff
                nearest_x_idx = i
        
        # Apply X correction if close to keypoint
        if nearest_x_idx is not None and min_x_diff < 30:
            nearest_tactical_x = self.key_points[nearest_x_idx][0]
            weight_x = 1.0 - (min_x_diff / 30.0)
            corrected_x = tactical_position[0] * (1 - weight_x) + nearest_tactical_x * weight_x
        
        return [corrected_x, corrected_y]

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """Transform player positions from video to tactical court coordinates."""
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            tactical_positions = {}

            frame_keypoints = frame_keypoints.xy.tolist()[0]

            if frame_keypoints is None or len(frame_keypoints) == 0:
                tactical_player_positions.append(tactical_positions)
                continue
            
            detected_keypoints = frame_keypoints
            
            valid_indices = [i for i, kp in enumerate(detected_keypoints) if kp[0] > 0 and kp[1] > 0]
            
            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue
            
            source_points = np.array([detected_keypoints[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)
            
            try:
                homography = Homography(source_points, target_points)
                
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    player_position = np.array([get_foot_position(bbox)])
                    tactical_position = homography.transform_points(player_position)

                    # Skip if out of bounds
                    if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height:
                        continue

                    # Correct position based on nearest keypoints
                    corrected_position = self._correct_position_by_nearest_keypoint(
                        get_foot_position(bbox),
                        detected_keypoints,
                        valid_indices,
                        tactical_position[0].tolist()
                    )
                    
                    tactical_positions[player_id] = corrected_position
                    
            except (ValueError, cv2.error):
                pass
            
            tactical_player_positions.append(tactical_positions)
        
        return tactical_player_positions

