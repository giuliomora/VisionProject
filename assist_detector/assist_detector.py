from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Assist:
    """Assist data."""
    frame: int
    passer_id: int
    scorer_id: int
    team_id: int
    shot_frame: int
    pass_to_shot_frames: int


class AssistDetector:
    """Detects assists by analyzing passes and made shots."""
    
    def __init__(self, 
                 max_frames_pass_to_shot: int = 200,
                 passer_search_frames: int = 100,
                 receiver_search_frames: int = 100,
                 debug: bool = False):
        """
        Args:
            max_frames_pass_to_shot: Max frames between pass and shot for assist
            passer_search_frames: Frames to search backwards for passer
            receiver_search_frames: Frames to search forwards for receiver
            debug: Enable debug output
        """
        self.max_frames_pass_to_shot = max_frames_pass_to_shot
        self.passer_search_frames = passer_search_frames
        self.receiver_search_frames = receiver_search_frames
        self.debug = debug
        self.assists: List[Assist] = []
    
    def detect_assists(self,
                       passes: List[int],
                       shots: List,
                       ball_acquisition: List[int],
                       player_assignment: List[Dict]) -> List[Assist]:
        """Detect assists by combining pass and shot data."""
        self.assists = []
        
        # Filter made shots only
        made_shots = [shot for shot in shots if shot.made]
        
        if self.debug:
            print("\n" + "="*70)
            print("ASSIST DETECTION")
            print("="*70)
            print(f"Made shots to analyze: {len(made_shots)}")
            print(f"Max frames pass to shot: {self.max_frames_pass_to_shot}")
        
        for shot in made_shots:
            # First attempt: search from frame_start
            assist = self._find_assist_for_shot(
                shot=shot,
                passes=passes,
                ball_acquisition=ball_acquisition,
                player_assignment=player_assignment,
                use_frame_end=False
            )
            
            # Second attempt: search from frame_end
            if assist is None:
                if self.debug:
                    print(f"    Retry with frame_end...")
                assist = self._find_assist_for_shot(
                    shot=shot,
                    passes=passes,
                    ball_acquisition=ball_acquisition,
                    player_assignment=player_assignment,
                    use_frame_end=True
                )
            
            if assist is not None:
                self.assists.append(assist)
                if self.debug:
                    print(f"\n  ✓ ASSIST FOUND!")
                    print(f"    Passer: Player {assist.passer_id}")
                    print(f"    Scorer: Player {assist.scorer_id}")
                    print(f"    Team: {assist.team_id}")
                    print(f"    Pass frame: {assist.frame}")
                    print(f"    Shot frame: {assist.shot_frame}")
                    print(f"    Time: {assist.pass_to_shot_frames} frames "
                          f"(~{assist.pass_to_shot_frames/30:.1f}s)")
        
        if self.debug:
            print(f"\n{'='*70}")
            print(f"TOTAL ASSISTS: {len(self.assists)}")
            print(f"{'='*70}\n")
        
        return self.assists
    
    def _find_assist_for_shot(self,
                               shot,
                               passes: List[int],
                               ball_acquisition: List[int],
                               player_assignment: List[Dict],
                               use_frame_end: bool = False) -> Optional[Assist]:
        """Search for qualifying pass before shot."""
        shot_frame = shot.frame_end if use_frame_end else shot.frame_start
        scorer_id = shot.player_id
        team_id = shot.team_id
        
        if self.debug:
            frame_type = "frame_end" if use_frame_end else "frame_start"
            print(f"\n  Shot analysis at {frame_type}: {shot_frame}")
            print(f"    Scorer: Player {scorer_id}, Team {team_id}")
        
        # Search for last pass before shot
        search_start = max(0, shot_frame - self.max_frames_pass_to_shot)
        
        last_pass_frame = None
        passer_id = None
        
        # Search backwards for same team pass
        for frame in range(shot_frame, search_start - 1, -1):
            if frame >= len(passes):
                continue
                
            if passes[frame] == team_id:
                last_pass_frame = frame
                passer_id = self._find_passer(frame, ball_acquisition, scorer_id)
                if passer_id is not None:
                    break
        
        if last_pass_frame is None or passer_id is None:
            if self.debug:
                print(f"    ✗ No pass found before shot")
            return None
        
        if self.debug:
            print(f"    Pass found at frame {last_pass_frame}")
            print(f"    Passer: Player {passer_id}")
            start = max(0, last_pass_frame - 10)
            end = min(len(ball_acquisition), last_pass_frame + 20)
            acquisitions = [(f, ball_acquisition[f]) for f in range(start, end) if ball_acquisition[f] != -1]
            print(f"    Ball acquisition from {start} to {end}: {acquisitions}")
        
        # Verify passer is not shooter
        if passer_id == scorer_id:
            if self.debug:
                print(f"    ✗ Passer is same as shooter")
            return None
        
        # Verify receiver is shooter
        receiver_id = self._find_receiver(last_pass_frame, ball_acquisition, scorer_id)
        
        if receiver_id != scorer_id:
            if self.debug:
                print(f"    ✗ Receiver ({receiver_id}) is not shooter ({scorer_id})")
            return None
        
        # Verify passer and scorer are same team
        passer_team = -1
        search_range = 50
        for offset in range(search_range):
            for direction in [0, -1, 1]:
                check_frame = last_pass_frame + (direction * offset)
                if 0 <= check_frame < len(player_assignment):
                    found_team = player_assignment[check_frame].get(passer_id, -1)
                    if found_team != -1:
                        passer_team = found_team
                        break
            if passer_team != -1:
                break
        
        if passer_team != team_id:
            if self.debug:
                print(f"    ✗ Passer (Team {passer_team}) not same team")
            return None
        
        pass_to_shot_frames = shot_frame - last_pass_frame
        
        return Assist(
            frame=last_pass_frame,
            passer_id=passer_id,
            scorer_id=scorer_id,
            team_id=team_id,
            shot_frame=shot_frame,
            pass_to_shot_frames=pass_to_shot_frames
        )
    
    def _find_passer(self, pass_frame: int, ball_acquisition: List[int], scorer_id: int = None) -> Optional[int]:
        """Find passer by searching backwards for ball holder."""
        search_limit = max(0, pass_frame - self.passer_search_frames)
        for frame in range(pass_frame, search_limit - 1, -1):
            if frame < len(ball_acquisition) and ball_acquisition[frame] != -1:
                holder = ball_acquisition[frame]
                if scorer_id is not None and holder == scorer_id:
                    continue
                return holder
        return None
    
    def _find_receiver(self, pass_frame: int, ball_acquisition: List[int], scorer_id: int = None) -> Optional[int]:
        """Find receiver by searching forwards for ball holder."""
        search_limit = min(len(ball_acquisition), pass_frame + self.receiver_search_frames)
        
        if scorer_id is not None:
            for frame in range(pass_frame, search_limit):
                if ball_acquisition[frame] == scorer_id:
                    return scorer_id
        
        for frame in range(pass_frame, search_limit):
            if ball_acquisition[frame] != -1:
                return ball_acquisition[frame]
        return None
    
    def get_team_stats(self) -> Dict[int, Dict]:
        """Get assist statistics by team."""
        stats = {}
        
        for assist in self.assists:
            if assist.team_id not in stats:
                stats[assist.team_id] = {
                    'total_assists': 0,
                    'assists_by_player': {}
                }
            
            stats[assist.team_id]['total_assists'] += 1
            
            passer = assist.passer_id
            if passer not in stats[assist.team_id]['assists_by_player']:
                stats[assist.team_id]['assists_by_player'][passer] = 0
            stats[assist.team_id]['assists_by_player'][passer] += 1
        
        return stats
    
    def export_to_csv(self, output_path: str, shots: List = None, 
                       passes: List[int] = None, interceptions: List[int] = None,
                       player_assignment: List = None, ball_acquisition: List[int] = None):
        """Export statistics to CSV files (one per team). Stats are cumulative."""
        import csv
        import os
        import pickle
        from collections import defaultdict
        
        base_dir = os.path.dirname(output_path)
        accumulated_stats_path = os.path.join(base_dir, 'accumulated_stats.pkl')
        
        # Load previous stats
        accumulated_stats = self._load_accumulated_stats(accumulated_stats_path)
        
        # Group assists by player
        assists_by_player = defaultdict(list)
        for assist in self.assists:
            assists_by_player[assist.passer_id].append(assist)
        
        # Group shots by player
        shots_by_player = defaultdict(list)
        if shots:
            for shot in shots:
                shots_by_player[shot.player_id].append(shot)
        
        # Count passes by player
        passes_by_player = defaultdict(int)
        if passes and ball_acquisition:
            prev_holder = -1
            for frame in range(1, len(passes)):
                if ball_acquisition[frame - 1] != -1:
                    prev_holder = ball_acquisition[frame - 1]
                if passes[frame] != -1 and prev_holder != -1:
                    passes_by_player[prev_holder] += 1
        
        # Count interceptions by player
        interceptions_by_player = defaultdict(int)
        if interceptions and ball_acquisition:
            for frame in range(len(interceptions)):
                if interceptions[frame] != -1 and ball_acquisition[frame] != -1:
                    interceptions_by_player[ball_acquisition[frame]] += 1
        
        # Determine team for each player
        player_teams = {}
        if player_assignment:
            for frame_data in player_assignment:
                for player_id, team_id in frame_data.items():
                    if player_id not in player_teams:
                        player_teams[player_id] = team_id
        
        # Accumulate stats
        accumulated_stats = self._accumulate_stats(
            accumulated_stats, 
            assists_by_player, shots_by_player, 
            passes_by_player, interceptions_by_player,
            player_teams
        )
        
        # Save accumulated stats
        self._save_accumulated_stats(accumulated_stats_path, accumulated_stats)
        
        # Export per team
        for team_id in [1, 2]:
            team_file = os.path.join(base_dir, f'stats_{team_id}.csv')
            self._export_team_csv_accumulated(team_file, team_id, accumulated_stats)
    
    def _load_accumulated_stats(self, path: str) -> Dict:
        """Load accumulated stats from pickle file."""
        import pickle
        import os
        
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    stats = pickle.load(f)
                    print(f"Loaded accumulated stats from {path}")
                    for team_id in [1, 2]:
                        if 'actions' not in stats[team_id]:
                            stats[team_id]['actions'] = {}
                        if 'action_count' not in stats[team_id]:
                            stats[team_id]['action_count'] = 0
                    return stats
            except Exception as e:
                print(f"Error loading stats: {e}")
        
        return {
            1: {'players': {}, 'actions': {}, 'action_count': 0},
            2: {'players': {}, 'actions': {}, 'action_count': 0}
        }
    
    def _save_accumulated_stats(self, path: str, stats: Dict):
        """Save accumulated stats to pickle file."""
        import pickle
        
        with open(path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Accumulated stats saved to {path}")
    
    def _accumulate_stats(self, accumulated: Dict, 
                          assists_by_player, shots_by_player,
                          passes_by_player, interceptions_by_player,
                          player_teams) -> Dict:
        """Merge new stats with existing accumulated stats."""
        
        # Increment action count
        for team_id in [1, 2]:
            accumulated[team_id]['action_count'] += 1
            action_num = accumulated[team_id]['action_count']
            
            team_passes_this_action = sum(
                passes_by_player[pid] for pid in passes_by_player 
                if player_teams.get(pid) == team_id
            )
            
            accumulated[team_id]['actions'][action_num] = {
                'passes': team_passes_this_action
            }
        
        all_players = set(assists_by_player.keys()) | set(shots_by_player.keys()) | \
                      set(passes_by_player.keys()) | set(interceptions_by_player.keys())
        
        for player_id in all_players:
            team_id = player_teams.get(player_id, 0)
            if team_id not in [1, 2]:
                continue
            
            # Initialize player if not exists
            if player_id not in accumulated[team_id]['players']:
                accumulated[team_id]['players'][player_id] = {
                    'shots_total': 0,
                    'shots_made': 0,
                    'shots_missed': 0,
                    'assists': 0,
                    'passes': 0,
                    'interceptions': 0
                }
            
            player_stats = accumulated[team_id]['players'][player_id]
            
            # Accumulate shots
            player_shots = shots_by_player[player_id]
            player_stats['shots_total'] += len(player_shots)
            player_stats['shots_made'] += sum(1 for s in player_shots if s.made)
            player_stats['shots_missed'] += sum(1 for s in player_shots if not s.made)
            
            # Accumulate assists, passes, interceptions
            player_stats['assists'] += len(assists_by_player[player_id])
            player_stats['passes'] += passes_by_player[player_id]
            player_stats['interceptions'] += interceptions_by_player[player_id]
        
        return accumulated
    
    def _export_team_csv_accumulated(self, output_path: str, team_id: int, accumulated_stats: Dict):
        """Export accumulated team stats to CSV file."""
        import csv
        
        team_data = accumulated_stats.get(team_id, {'players': {}, 'actions': {}, 'action_count': 0})
        players = team_data['players']
        actions = team_data.get('actions', {})
        action_count = team_data.get('action_count', 0)
        
        # Sort players by total actions
        sorted_players = sorted(players.keys(), 
                                key=lambda p: -(players[p]['shots_total'] + players[p]['assists'] + 
                                               players[p]['passes'] + players[p]['interceptions']))
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Team statistics
            team_shots = sum(players[p]['shots_total'] for p in players)
            team_made = sum(players[p]['shots_made'] for p in players)
            team_missed = sum(players[p]['shots_missed'] for p in players)
            team_assists = sum(players[p]['assists'] for p in players)
            team_passes = sum(players[p]['passes'] for p in players)
            team_interceptions = sum(players[p]['interceptions'] for p in players)
            
            writer.writerow(['=' * 50])
            writer.writerow([f'TEAM {team_id} STATISTICS (CUMULATIVE)'])
            writer.writerow(['=' * 50])
            writer.writerow(['Statistic', 'Value'])
            writer.writerow(['Actions analyzed', action_count])
            writer.writerow(['Total shots', team_shots])
            writer.writerow(['Shots made', team_made])
            writer.writerow(['Shots missed', team_missed])
            writer.writerow(['FG%', f'{(team_made/team_shots*100):.1f}%' if team_shots > 0 else 'N/A'])
            writer.writerow(['Total assists', team_assists])
            writer.writerow(['Total passes', team_passes])
            writer.writerow(['Total interceptions', team_interceptions])
            writer.writerow([])
            
            # Passes per action
            writer.writerow(['=' * 50])
            writer.writerow(['PASSES PER ACTION'])
            writer.writerow(['=' * 50])
            writer.writerow(['Action', 'Passes'])
            
            for action_num in sorted(actions.keys()):
                action_data = actions[action_num]
                writer.writerow([f'Action {action_num}', action_data.get('passes', 0)])
            
            writer.writerow([])
            
            # Player statistics
            writer.writerow(['=' * 50])
            writer.writerow(['PLAYER STATISTICS'])
            writer.writerow(['=' * 50])
            writer.writerow([])
            
            for player_id in sorted_players:
                p_stats = players[player_id]
                
                writer.writerow([f'PLAYER {player_id}'])
                writer.writerow(['', 'Statistic', 'Value'])
                writer.writerow(['', 'Total shots', p_stats['shots_total']])
                writer.writerow(['', 'Shots made', p_stats['shots_made']])
                writer.writerow(['', 'Shots missed', p_stats['shots_missed']])
                fg_pct = f'{(p_stats["shots_made"]/p_stats["shots_total"]*100):.1f}%' if p_stats['shots_total'] > 0 else 'N/A'
                writer.writerow(['', 'FG%', fg_pct])
                writer.writerow(['', 'Assists', p_stats['assists']])
                writer.writerow(['', 'Passes', p_stats['passes']])
                writer.writerow(['', 'Interceptions', p_stats['interceptions']])
                writer.writerow([])
                writer.writerow([])
        
        print(f"Team {team_id} cumulative stats exported to: {output_path}")
