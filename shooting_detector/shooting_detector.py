import numpy as np
import csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import sys
sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, measure_distance


@dataclass
class Shot:
    """Rappresenta un tentativo di tiro."""
    frame_start: int
    frame_end: int
    team_id: int
    player_id: int
    made: bool
    position: Tuple[float, float]
    hoop_id: int


class ShootingDetector:
    """Rileva i tiri e se sono andati a segno."""
    
    def __init__(self, 
                 hoop_proximity_threshold: float = 250,    # Aumentato
                 ball_rising_frames: int = 3,              # Ridotto
                 made_shot_threshold: float = 100,         # Aumentato
                 min_frames_between_shots: int = 90,       # Ridotto (per tiri sbagliati)
                 cooldown_after_made_shot: int = 150,      # ~5 secondi a 30fps dopo un canestro segnato
                 debug: bool = True):                      # Debug mode
        self.hoop_proximity_threshold = hoop_proximity_threshold
        self.ball_rising_frames = ball_rising_frames
        self.made_shot_threshold = made_shot_threshold
        self.min_frames_between_shots = min_frames_between_shots
        self.cooldown_after_made_shot = cooldown_after_made_shot  # Cooldown dopo tiro segnato
        self.debug = debug
        self.shots: List[Shot] = []

    def _get_ball_center(self, ball_track: Dict) -> Optional[Tuple[float, float]]:
        """Estrae il centro della bbox della palla."""
        if not ball_track or 1 not in ball_track:
            return None
        
        bbox = ball_track[1].get('bbox')
        if bbox is None:
            return None
        
        return get_center_of_bbox(bbox)

    def _get_hoop_centers(self, hoop_track: Dict) -> Dict[int, Tuple[float, float]]:
        """Estrae i centri dei canestri dal tracking."""
        hoops = {}
        if not hoop_track:
            return hoops
        for hoop_id, hoop_data in hoop_track.items():
            if isinstance(hoop_data, dict):
                bbox = hoop_data.get('bbox')
                if bbox:
                    hoops[hoop_id] = get_center_of_bbox(bbox)
        return hoops

    def _get_hoop_bottom(self, hoop_track: Dict) -> Dict[int, Tuple[float, float]]:
        """Estrae il punto inferiore dei canestri (dove passa la palla)."""
        hoops = {}
        if not hoop_track:
            return hoops
        for hoop_id, hoop_data in hoop_track.items():
            if isinstance(hoop_data, dict):
                bbox = hoop_data.get('bbox')
                if bbox:
                    # Punto inferiore centrale del canestro
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_bottom = bbox[3]  # Parte inferiore
                    hoops[hoop_id] = (x_center, y_bottom)
        return hoops

    def _get_player_with_ball(self, frame_idx: int, ball_acquisition: List[int]) -> Optional[int]:
        """Trova il giocatore che ha la palla in un dato frame."""
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
        """
        Controlla se la palla si sta avvicinando a un canestro.
        Ritorna (True/False, hoop_id).
        """
        if frame_idx < look_back:
            return False, None
        
        current_ball = self._get_ball_center(ball_tracks[frame_idx])
        if current_ball is None:
            return False, None
        
        current_hoops = self._get_hoop_centers(hoop_tracks[frame_idx]) if frame_idx < len(hoop_tracks) else {}
        if not current_hoops:
            return False, None
        
        # Calcola distanze attuali
        current_distances = {hoop_id: measure_distance(current_ball, pos) 
                            for hoop_id, pos in current_hoops.items()}
        
        # Calcola distanze passate
        past_frame = frame_idx - look_back
        past_ball = self._get_ball_center(ball_tracks[past_frame])
        if past_ball is None:
            return False, None
        
        past_hoops = self._get_hoop_centers(hoop_tracks[past_frame]) if past_frame < len(hoop_tracks) else {}
        
        for hoop_id, current_dist in current_distances.items():
            if hoop_id in past_hoops:
                past_dist = measure_distance(past_ball, past_hoops[hoop_id])
                # Se la palla si è avvicinata significativamente
                if past_dist - current_dist > 50:
                    return True, hoop_id
        
        return False, None

    def _is_ball_near_hoop(self, ball_center: Tuple[float, float], 
                           hoop_track: Dict,
                           threshold: float) -> Tuple[bool, Optional[int], float]:
        """
        Controlla se la palla è vicina a un canestro.
        Ritorna (True/False, hoop_id, distanza).
        """
        hoops = self._get_hoop_centers(hoop_track)
        
        for hoop_id, hoop_center in hoops.items():
            dist = measure_distance(ball_center, hoop_center)
            if dist < threshold:
                return True, hoop_id, dist
        
        return False, None, float('inf')

    def _detect_made_shot(self, ball_tracks: List[Dict], 
                          hoop_tracks: List[Dict],
                          frame_start: int, 
                          hoop_id: int,
                          look_ahead_frames: int = 30) -> bool:
        """
        Rileva se il tiro è andato a segno.
        Cerca se la palla passa vicino/attraverso il canestro e poi scende.
        """
        end_frame = min(frame_start + look_ahead_frames, len(ball_tracks))
        
        min_distance = float('inf')
        ball_y_at_min_dist = 0
        frame_at_min_dist = 0
        
        # Trova il frame in cui la palla è più vicina al canestro
        for i in range(frame_start, end_frame):
            ball_center = self._get_ball_center(ball_tracks[i])
            if ball_center is None:
                continue
            
            hoops = self._get_hoop_bottom(hoop_tracks[i]) if i < len(hoop_tracks) else {}
            if hoop_id not in hoops:
                hoops = self._get_hoop_centers(hoop_tracks[i]) if i < len(hoop_tracks) else {}
            
            if hoop_id not in hoops:
                continue
            
            hoop_pos = hoops[hoop_id]
            dist = measure_distance(ball_center, hoop_pos)
            
            if dist < min_distance:
                min_distance = dist
                ball_y_at_min_dist = ball_center[1]
                frame_at_min_dist = i
        
        if self.debug:
            print(f"  Min distance to hoop: {min_distance:.1f} at frame {frame_at_min_dist}")
        
        # Se la palla non è mai stata abbastanza vicina, non è un canestro
        if min_distance > self.made_shot_threshold:
            return False
        
        # Controlla se la palla scende dopo il punto più vicino
        ball_descended = False
        for i in range(frame_at_min_dist + 1, min(frame_at_min_dist + 15, len(ball_tracks))):
            ball_center = self._get_ball_center(ball_tracks[i])
            if ball_center and ball_center[1] > ball_y_at_min_dist + 30:
                ball_descended = True
                break
        
        if self.debug:
            print(f"  Ball descended after: {ball_descended}")
        
        return ball_descended

    def _find_shooter(self, 
                      frame_idx: int,
                      ball_acquisition: List[int],
                      player_assignment: List[Dict],
                      player_tracks: List[Dict],
                      look_back_frames: int = 60) -> Tuple[Optional[int], Optional[int], Optional[Tuple[float, float]]]:
        """Trova il giocatore che ha tirato cercando indietro nel tempo."""
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

    def debug_shooter_detection(self,
                                 frame_idx: int,
                                 ball_acquisition: List[int],
                                 player_assignment: List[Dict],
                                 player_tracks: List[Dict],
                                 tactical_player_positions: List[Dict] = None,
                                 look_back_frames: int = 60):
        """
        METODO DI DEBUG: Mostra passo per passo come viene identificato il tiratore.
        
        Questo metodo spiega:
        1. Come viene trovato chi ha la palla
        2. Come viene determinata la posizione video
        3. Come viene trasformata in posizione tattica
        """
        print("\n" + "="*70)
        print(f"DEBUG SHOOTER DETECTION - Frame {frame_idx}")
        print("="*70)
        
        print(f"\n[STEP 1] Ricerca del possessore della palla")
        print(f"         Cerco indietro fino a {look_back_frames} frame...")
        
        found_frame = None
        found_player = None
        
        for look_back in range(min(look_back_frames, frame_idx)):
            check_frame = frame_idx - look_back
            
            # Mostra cosa c'è in ball_acquisition per questo frame
            if check_frame < len(ball_acquisition):
                ball_holder = ball_acquisition[check_frame]
                
                if ball_holder != -1:
                    print(f"\n         ✓ Frame {check_frame} (look_back={look_back}): "
                          f"Giocatore {ball_holder} ha la palla!")
                    found_frame = check_frame
                    found_player = ball_holder
                    break
                else:
                    if look_back < 5:  # Mostra solo i primi 5 per non intasare
                        print(f"         Frame {check_frame}: nessuno ha la palla (ball_acquisition={ball_holder})")
        
        if found_player is None:
            print(f"\n         ✗ Nessun giocatore trovato con la palla negli ultimi {look_back_frames} frame")
            return
        
        print(f"\n[STEP 2] Identificazione del team")
        team_id = 1
        if found_frame < len(player_assignment):
            team_id = player_assignment[found_frame].get(found_player, 1)
            print(f"         player_assignment[{found_frame}][{found_player}] = Team {team_id}")
        else:
            print(f"         ✗ player_assignment non disponibile per frame {found_frame}, uso Team 1 di default")
        
        print(f"\n[STEP 3] Posizione VIDEO del giocatore")
        position = None
        if found_frame < len(player_tracks) and found_player in player_tracks[found_frame]:
            player_data = player_tracks[found_frame][found_player]
            bbox = player_data.get('bbox')
            if bbox:
                position = get_center_of_bbox(bbox)
                print(f"         Bounding box: {bbox}")
                print(f"         Centro (posizione video): ({position[0]:.1f}, {position[1]:.1f}) pixel")
            else:
                print(f"         ✗ Nessun bbox disponibile per player {found_player}")
        else:
            print(f"         ✗ Player {found_player} non trovato in player_tracks[{found_frame}]")
        
        print(f"\n[STEP 4] Posizione TATTICA del giocatore")
        if tactical_player_positions is not None:
            tactical_pos = None
            for offset in range(0, 30):
                check_frame = frame_idx - offset
                if check_frame >= 0 and check_frame < len(tactical_player_positions):
                    frame_positions = tactical_player_positions[check_frame]
                    if found_player in frame_positions:
                        tactical_pos = frame_positions[found_player]
                        print(f"         Trovata posizione tattica al frame {check_frame} (offset={offset})")
                        print(f"         Posizione tattica: ({tactical_pos[0]:.1f}, {tactical_pos[1]:.1f})")
                        print(f"         (Coordinate nel sistema 300x161 del campo)")
                        break
            
            if tactical_pos is None:
                print(f"         ✗ Nessuna posizione tattica trovata per player {found_player}")
                print(f"           Possibili cause:")
                print(f"           - Il giocatore era fuori dal campo")
                print(f"           - I keypoint del campo non erano visibili")
                print(f"           - L'omografia non era calcolabile")
        else:
            print(f"         (tactical_player_positions non fornito)")
        
        print(f"\n[RISULTATO]")
        print(f"         Tiratore: Player {found_player}")
        print(f"         Team: {team_id}")
        print(f"         Posizione video: {position}")
        print("="*70 + "\n")

    def detect_shots(self,
                     ball_tracks: List[Dict],
                     hoop_tracks: List[Dict],
                     player_tracks: List[Dict],
                     player_assignment: List[Dict],
                     ball_acquisition: List[int]) -> List[Shot]:
        """Rileva tutti i tiri nel video."""
        self.shots = []
        last_shot_frame = -self.min_frames_between_shots
        last_shot_was_made = False  # Traccia se l'ultimo tiro era segnato
        
        if self.debug:
            print("\n=== SHOOTING DETECTION ===")
            print(f"Threshold proximity: {self.hoop_proximity_threshold}")
            print(f"Threshold made: {self.made_shot_threshold}")
            print(f"Cooldown after made shot: {self.cooldown_after_made_shot} frames")
            print(f"Min frames between missed shots: {self.min_frames_between_shots} frames")
        
        for frame_idx in range(len(ball_tracks)):
            # Determina il cooldown in base al risultato dell'ultimo tiro
            # Se l'ultimo tiro era segnato, usa un cooldown più lungo
            # Se era sbagliato, usa un cooldown più breve (può esserci un rimbalzo offensivo)
            if last_shot_was_made:
                current_cooldown = self.cooldown_after_made_shot
            else:
                current_cooldown = self.min_frames_between_shots
            
            # Salta se troppo vicino all'ultimo tiro
            if frame_idx - last_shot_frame < current_cooldown:
                continue
            
            ball_center = self._get_ball_center(ball_tracks[frame_idx])
            if ball_center is None:
                continue
            
            if frame_idx >= len(hoop_tracks):
                continue
                
            # Controlla se la palla è vicina a un canestro
            is_near, hoop_id, dist = self._is_ball_near_hoop(
                ball_center, hoop_tracks[frame_idx], self.hoop_proximity_threshold
            )
            
            if is_near:
                # Controlla se la palla si sta avvicinando (non allontanando)
                approaching, _ = self._is_ball_approaching_hoop(
                    ball_tracks, hoop_tracks, frame_idx
                )
                
                if approaching or dist < self.made_shot_threshold * 1.5:
                    if self.debug:
                        print(f"\nPotential shot at frame {frame_idx}, dist: {dist:.1f}, hoop: {hoop_id}")
                    
                    # Determina se è andato a segno
                    made = self._detect_made_shot(
                        ball_tracks, hoop_tracks, 
                        frame_idx, hoop_id
                    )
                    
                    # Trova chi ha tirato
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
                    last_shot_was_made = made  # Memorizza se questo tiro era segnato
                    
                    cooldown_info = f"(cooldown: {self.cooldown_after_made_shot}f)" if made else f"(cooldown: {self.min_frames_between_shots}f)"
                    print(f"Shot detected: Frame {frame_idx}, "
                          f"Team {team_id}, Player {player_id}, "
                          f"Made: {made}, Dist: {dist:.1f} {cooldown_info}")
        
        return self.shots

    def get_team_stats(self) -> Dict[int, Dict[str, int]]:
        """Calcola le statistiche per squadra."""
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
        """Esporta le statistiche per squadra in un file TSV."""
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
        """Esporta tutti i tiri singoli in un file TSV."""
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