import numpy as np
import csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import sys
sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, measure_distance


@dataclass
class Shot:
    """Shot attempt data."""
    frame_start: int
    frame_end: int
    team_id: int
    player_id: int
    made: bool
    position: Tuple[float, float]
    hoop_id: int


class ShootingDetector:
    """Detects shots and determines if they are made."""
    
    def __init__(self, 
                 hoop_proximity_threshold: float = 250,
                 ball_rising_frames: int = 3,
                 made_shot_threshold: float = 100,
                 min_frames_between_shots: int = 90,
                 cooldown_after_made_shot: int = 150,
                 debug: bool = False):
        self.hoop_proximity_threshold = hoop_proximity_threshold
        self.ball_rising_frames = ball_rising_frames
        self.made_shot_threshold = made_shot_threshold
        self.min_frames_between_shots = min_frames_between_shots
        self.cooldown_after_made_shot = cooldown_after_made_shot
        self.debug = debug
        self.shots: List[Shot] = []

    def _get_ball_center(self, ball_track: Dict) -> Optional[Tuple[float, float]]:
        """Extract ball bbox center."""
        if not ball_track or 1 not in ball_track:
            return None
        bbox = ball_track[1].get('bbox')
        if bbox is None:
            return None
        return get_center_of_bbox(bbox)

    def _get_hoop_centers(self, hoop_track: Dict) -> Dict[int, Tuple[float, float]]:
        """Extract hoop centers from tracking data."""
        hoops = {}
        if not hoop_track:
            return hoops
        for hoop_id, hoop_data in hoop_track.items():
            if isinstance(hoop_data, dict):
                bbox = hoop_data.get('bbox')
                if bbox:
                    hoops[hoop_id] = get_center_of_bbox(bbox)
        return hoops

    def _get_player_with_ball(self, frame_idx: int, ball_acquisition: List[int]) -> Optional[int]:
        """Find player with ball at given frame."""
        if frame_idx >= len(ball_acquisition) or frame_idx < 0:
            return None
        player_id = ball_acquisition[frame_idx]
        if player_id != -1:
            return player_id
        return None

    def _is_ball_approaching_hoop(self, ball_tracks: List[Dict], 
                                   hoop_tracks: List[Dict],
                                   frame_idx: int,
                                   look_back: int = 10) -> Tuple[bool, Optional[int]]:
        """Check if ball is approaching a hoop. Returns (is_approaching, hoop_id)."""
        if frame_idx < look_back:
            return False, None
        
        current_ball = self._get_ball_center(ball_tracks[frame_idx])
        if current_ball is None:
            return False, None
        
        current_hoops = self._get_hoop_centers(hoop_tracks[frame_idx]) if frame_idx < len(hoop_tracks) else {}
        if not current_hoops:
            return False, None
        
        # Calculate current distances
        current_distances = {hoop_id: measure_distance(current_ball, pos) 
                            for hoop_id, pos in current_hoops.items()}
        
        # Calculate past distances
        past_frame = frame_idx - look_back
        past_ball = self._get_ball_center(ball_tracks[past_frame])
        if past_ball is None:
            return False, None
        
        past_hoops = self._get_hoop_centers(hoop_tracks[past_frame]) if past_frame < len(hoop_tracks) else {}
        
        for hoop_id, current_dist in current_distances.items():
            if hoop_id in past_hoops:
                past_dist = measure_distance(past_ball, past_hoops[hoop_id])
                if past_dist - current_dist > 50:
                    return True, hoop_id
        
        return False, None

    def _is_ball_near_hoop(self, ball_center: Tuple[float, float], 
                           hoop_track: Dict,
                           threshold: float) -> Tuple[bool, Optional[int], float]:
        """Check if ball is near a hoop. Returns (is_near, hoop_id, distance)."""
        hoops = self._get_hoop_centers(hoop_track)
        for hoop_id, hoop_center in hoops.items():
            dist = measure_distance(ball_center, hoop_center)
            if dist < threshold:
                return True, hoop_id, dist
        return False, None, float('inf')

    def _get_hoop_bbox(self, hoop_track: Dict, hoop_id: int) -> Optional[Tuple[float, float, float, float]]:
        """Extract hoop bounding box."""
        if not hoop_track or hoop_id not in hoop_track:
            return None
        hoop_data = hoop_track[hoop_id]
        if isinstance(hoop_data, dict):
            return hoop_data.get('bbox')
        return None

    def _detect_made_shot(self, ball_tracks: List[Dict], 
                          hoop_tracks: List[Dict],
                          frame_start: int, 
                          hoop_id: int,
                          look_ahead_frames: int = 60,
                          horizontal_tolerance: float = 1.2,
                          direction_change_threshold: float = 25.0) -> bool:
        """Detect if shot was made by checking ball trajectory through hoop."""
        end_frame = min(frame_start + look_ahead_frames, len(ball_tracks))
        
        # Raccogli le posizioni della palla e del canestro
        ball_positions = []  # Lista di (frame, ball_x, ball_y)
        hoop_info = None     # (hoop_center_x, hoop_top_y, hoop_bottom_y, hoop_width)
        
        for i in range(frame_start, end_frame):
            ball_center = self._get_ball_center(ball_tracks[i])
            if ball_center is None:
                continue
            
            # Get hoop bbox
            if i < len(hoop_tracks):
                hoop_bbox = self._get_hoop_bbox(hoop_tracks[i], hoop_id)
                if hoop_bbox:
                    hoop_x1, hoop_y1, hoop_x2, hoop_y2 = hoop_bbox
                    hoop_center_x = (hoop_x1 + hoop_x2) / 2
                    hoop_width = hoop_x2 - hoop_x1
                    hoop_top_y = hoop_y1
                    hoop_bottom_y = hoop_y2
                    hoop_info = (hoop_center_x, hoop_top_y, hoop_bottom_y, hoop_width)
            
            ball_positions.append((i, ball_center[0], ball_center[1]))
        
        if not ball_positions or hoop_info is None:
            return False
        
        hoop_center_x, hoop_top_y, hoop_bottom_y, hoop_width = hoop_info
        
        # Define pass-through zone
        half_tolerance = (hoop_width * horizontal_tolerance) / 2
        pass_zone_left = hoop_center_x - half_tolerance
        pass_zone_right = hoop_center_x + half_tolerance
        
        # Step 1: Find frames where ball is above or inside hoop
        above_frames = []
        for frame_idx, ball_x, ball_y in ball_positions:
            is_above_or_inside = ball_y < hoop_bottom_y
            if is_above_or_inside:
                above_frames.append((frame_idx, ball_x, ball_y))
        
        if not above_frames:
            return False
        
        # Step 2: Find frames where ball is below hoop and aligned
        last_above_frame = above_frames[-1][0]
        
        below_and_aligned = False
        transition_frame = None
        
        for frame_idx, ball_x, ball_y in ball_positions:
            if frame_idx <= last_above_frame:
                continue
            
            is_below = ball_y > hoop_bottom_y
            is_aligned = pass_zone_left <= ball_x <= pass_zone_right
            
            if is_below and is_aligned:
                below_and_aligned = True
                transition_frame = frame_idx
                break
        
        if not below_and_aligned:
            return False
        
        # Step 3: Check for bounce (direction change)
        critical_frames = []
        for frame_idx, ball_x, ball_y in ball_positions:
            if (hoop_top_y - 20) <= ball_y <= (hoop_bottom_y + 20):
                critical_frames.append((frame_idx, ball_x, ball_y))
        
        if len(critical_frames) >= 3:
            # Calculate X velocities
            velocities_x = []
            for i in range(1, len(critical_frames)):
                prev_frame, prev_x, _ = critical_frames[i-1]
                curr_frame, curr_x, _ = critical_frames[i]
                
                frame_diff = curr_frame - prev_frame
                if frame_diff > 0:
                    velocity_x = (curr_x - prev_x) / frame_diff
                    velocities_x.append((curr_frame, velocity_x))
            
            # Detect direction reversal (bounce)
            for i in range(1, len(velocities_x)):
                prev_vel = velocities_x[i-1][1]
                curr_vel = velocities_x[i][1]
                
                if prev_vel * curr_vel < 0:
                    magnitude_change = abs(prev_vel) + abs(curr_vel)
                    if magnitude_change > direction_change_threshold:
                        return False
        
        return True

    def _find_shooter(self, 
                      frame_idx: int,
                      ball_acquisition: List[int],
                      player_assignment: List[Dict],
                      player_tracks: List[Dict],
                      look_back_frames: int = 60) -> Tuple[Optional[int], Optional[int], Optional[Tuple[float, float]]]:
        """Find shooter by searching backwards in time."""
        for look_back in range(min(look_back_frames, frame_idx)):
            check_frame = frame_idx - look_back
            player_id = self._get_player_with_ball(check_frame, ball_acquisition)
            
            if player_id is not None:
                team_id = 1
                if check_frame < len(player_assignment):
                    team_id = player_assignment[check_frame].get(player_id, 1)
                
                position = None
                if check_frame < len(player_tracks) and player_id in player_tracks[check_frame]:
                    bbox = player_tracks[check_frame][player_id].get('bbox')
                    if bbox:
                        position = get_center_of_bbox(bbox)
                
                return player_id, team_id, position
        
        return None, None, None

    def detect_shots(self,
                     ball_tracks: List[Dict],
                     hoop_tracks: List[Dict],
                     player_tracks: List[Dict],
                     player_assignment: List[Dict],
                     ball_acquisition: List[int]) -> List[Shot]:
        """Detect all shots in video."""
        self.shots = []
        last_shot_frame = -self.min_frames_between_shots
        last_shot_was_made = False
        
        if self.debug:
            print("\n=== SHOOTING DETECTION ===")
            print(f"Threshold proximity: {self.hoop_proximity_threshold}")
            print(f"Threshold made: {self.made_shot_threshold}")
            print(f"Cooldown after made shot: {self.cooldown_after_made_shot} frames")
            print(f"Min frames between missed shots: {self.min_frames_between_shots} frames")
        
        for frame_idx in range(len(ball_tracks)):
            # Determine cooldown based on last shot result
            if last_shot_was_made:
                current_cooldown = self.cooldown_after_made_shot
            else:
                current_cooldown = self.min_frames_between_shots
            
            # Skip if too close to last shot
            if frame_idx - last_shot_frame < current_cooldown:
                continue
            
            ball_center = self._get_ball_center(ball_tracks[frame_idx])
            if ball_center is None:
                continue
            
            if frame_idx >= len(hoop_tracks):
                continue
                
            # Check if ball is near hoop
            is_near, hoop_id, dist = self._is_ball_near_hoop(
                ball_center, hoop_tracks[frame_idx], self.hoop_proximity_threshold
            )
            
            if is_near:
                # Check if ball is approaching
                approaching, _ = self._is_ball_approaching_hoop(
                    ball_tracks, hoop_tracks, frame_idx
                )
                
                if approaching or dist < self.made_shot_threshold * 1.5:
                    if self.debug:
                        print(f"\nPotential shot at frame {frame_idx}, dist: {dist:.1f}, hoop: {hoop_id}")
                    
                    # Determine if made
                    made = self._detect_made_shot(
                        ball_tracks, hoop_tracks, 
                        frame_idx, hoop_id
                    )
                    
                    # Find shooter
                    player_id, team_id, position = self._find_shooter(
                        frame_idx,
                        ball_acquisition,
                        player_assignment,
                        player_tracks
                    )
                    
                    shot = Shot(
                        frame_start=frame_idx,
                        frame_end=frame_idx + 20,
                        team_id=team_id or 1,
                        player_id=player_id or -1,
                        made=made,
                        position=position or (0, 0),
                        hoop_id=hoop_id
                    )
                    self.shots.append(shot)
                    
                    last_shot_frame = frame_idx
                    last_shot_was_made = made
                    
                    cooldown_info = f"(cooldown: {self.cooldown_after_made_shot}f)" if made else f"(cooldown: {self.min_frames_between_shots}f)"
                    print(f"Shot detected: Frame {frame_idx}, "
                          f"Team {team_id}, Player {player_id}, "
                          f"Made: {made}, Dist: {dist:.1f} {cooldown_info}")
        
        return self.shots

    def get_team_stats(self) -> Dict[int, Dict[str, int]]:
        """Calculate team statistics."""
        stats = {
            1: {'attempts': 0, 'made': 0, 'missed': 0},
            2: {'attempts': 0, 'made': 0, 'missed': 0}
        }
        
        for shot in self.shots:
            team = shot.team_id
            if team in stats:
                stats[team]['attempts'] += 1
                if shot.made:
                    stats[team]['made'] += 1
                else:
                    stats[team]['missed'] += 1
        
        return stats

    def export_to_tsv(self, output_path: str):
        """Export team statistics to TSV file."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['team_id', 'attempts', 'made', 'missed', 'percentage'])
            
            stats = self.get_team_stats()
            for team_id, team_stats in stats.items():
                attempts = team_stats['attempts']
                made = team_stats['made']
                missed = team_stats['missed']
                percentage = (made / attempts * 100) if attempts > 0 else 0
                writer.writerow([team_id, attempts, made, missed, f"{percentage:.1f}%"])
        
        print(f"Team stats exported to {output_path}")

    def export_shots_to_tsv(self, output_path: str):
        """Export all shots to TSV file."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([
                'shot_id', 'frame_start', 'frame_end', 
                'team_id', 'player_id', 'made', 
                'position_x', 'position_y', 'hoop_id'
            ])
            
            for i, shot in enumerate(self.shots):
                writer.writerow([
                    i + 1,
                    shot.frame_start,
                    shot.frame_end,
                    shot.team_id,
                    shot.player_id,
                    shot.made,
                    f"{shot.position[0]:.1f}",
                    f"{shot.position[1]:.1f}",
                    shot.hoop_id
                ])
        
        print(f"All shots exported to {output_path}")