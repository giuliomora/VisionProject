from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Assist:
    """Rappresenta un assist."""
    frame: int                      # Frame in cui è avvenuto il passaggio
    passer_id: int                  # ID del giocatore che ha passato
    scorer_id: int                  # ID del giocatore che ha segnato
    team_id: int                    # ID della squadra
    shot_frame: int                 # Frame del tiro
    pass_to_shot_frames: int        # Numero di frame tra passaggio e tiro


class AssistDetector:
    """
    Rileva gli assist analizzando i passaggi e i tiri andati a segno.
    
    Un assist viene rilevato quando:
    1. Un giocatore passa la palla a un compagno di squadra
    2. Il compagno tira entro un certo numero di frame
    3. Il tiro va a segno
    """
    
    def __init__(self, 
                 max_frames_pass_to_shot: int = 150,  # ~5 secondi a 30fps
                 debug: bool = True):
        """
        Args:
            max_frames_pass_to_shot: Numero massimo di frame tra il passaggio e il tiro
                                     per considerarlo un assist (default ~5 secondi)
            debug: Se True, stampa informazioni di debug
        """
        self.max_frames_pass_to_shot = max_frames_pass_to_shot
        self.debug = debug
        self.assists: List[Assist] = []
    
    def detect_assists(self,
                       passes: List[int],
                       shots: List,  # List[Shot]
                       ball_acquisition: List[int],
                       player_assignment: List[Dict]) -> List[Assist]:
        """
        Rileva gli assist combinando informazioni su passaggi e tiri.
        
        Args:
            passes: Lista con team_id del passaggio per frame (-1 se nessun passaggio)
            shots: Lista di oggetti Shot (tiri rilevati)
            ball_acquisition: Lista con player_id che ha la palla per frame (-1 se nessuno)
            player_assignment: Lista di dict {player_id: team_id} per frame
        
        Returns:
            Lista di oggetti Assist
        """
        self.assists = []
        
        # Filtra solo i tiri andati a segno
        made_shots = [shot for shot in shots if shot.made]
        
        if self.debug:
            print("\n" + "="*70)
            print("ASSIST DETECTION")
            print("="*70)
            print(f"Tiri segnati da analizzare: {len(made_shots)}")
            print(f"Frame massimi tra passaggio e tiro: {self.max_frames_pass_to_shot}")
        
        for shot in made_shots:
            assist = self._find_assist_for_shot(
                shot=shot,
                passes=passes,
                ball_acquisition=ball_acquisition,
                player_assignment=player_assignment
            )
            
            if assist is not None:
                self.assists.append(assist)
                if self.debug:
                    print(f"\n  ✓ ASSIST TROVATO!")
                    print(f"    Passatore: Player {assist.passer_id}")
                    print(f"    Scorer: Player {assist.scorer_id}")
                    print(f"    Team: {assist.team_id}")
                    print(f"    Frame passaggio: {assist.frame}")
                    print(f"    Frame tiro: {assist.shot_frame}")
                    print(f"    Tempo: {assist.pass_to_shot_frames} frame "
                          f"(~{assist.pass_to_shot_frames/30:.1f}s)")
        
        if self.debug:
            print(f"\n{'='*70}")
            print(f"TOTALE ASSIST RILEVATI: {len(self.assists)}")
            print(f"{'='*70}\n")
        
        return self.assists
    
    def _find_assist_for_shot(self,
                               shot,  # Shot object
                               passes: List[int],
                               ball_acquisition: List[int],
                               player_assignment: List[Dict]) -> Optional[Assist]:
        """
        Cerca se c'è stato un passaggio prima del tiro che qualifica come assist.
        
        Logica:
        1. Cerca indietro dal frame del tiro per trovare l'ultimo passaggio
        2. Verifica che il passaggio sia della stessa squadra del tiratore
        3. Verifica che chi ha ricevuto il passaggio sia il tiratore
        """
        shot_frame = shot.frame_start
        scorer_id = shot.player_id
        team_id = shot.team_id
        
        if self.debug:
            print(f"\n  Analisi tiro al frame {shot_frame}")
            print(f"    Scorer: Player {scorer_id}, Team {team_id}")
        
        # Cerca l'ultimo passaggio prima del tiro
        search_start = max(0, shot_frame - self.max_frames_pass_to_shot)
        
        last_pass_frame = None
        passer_id = None
        
        # Cerca indietro per trovare l'ultimo passaggio della stessa squadra
        for frame in range(shot_frame - 1, search_start - 1, -1):
            if frame >= len(passes):
                continue
                
            if passes[frame] == team_id:
                # Trovato un passaggio della stessa squadra
                last_pass_frame = frame
                
                # Trova chi ha passato (il giocatore che aveva la palla prima del passaggio)
                passer_id = self._find_passer(frame, ball_acquisition)
                
                if passer_id is not None:
                    break
        
        if last_pass_frame is None or passer_id is None:
            if self.debug:
                print(f"    ✗ Nessun passaggio trovato prima del tiro")
            return None
        
        # Verifica che il passatore non sia il tiratore
        if passer_id == scorer_id:
            if self.debug:
                print(f"    ✗ Il passatore è lo stesso del tiratore")
            return None
        
        # Verifica che chi ha ricevuto il passaggio sia effettivamente il tiratore
        receiver_id = self._find_receiver(last_pass_frame, ball_acquisition)
        
        if receiver_id != scorer_id:
            if self.debug:
                print(f"    ✗ Il ricevente ({receiver_id}) non è il tiratore ({scorer_id})")
            return None
        
        # Verifica che passatore e scorer siano della stessa squadra
        if last_pass_frame < len(player_assignment):
            passer_team = player_assignment[last_pass_frame].get(passer_id, -1)
            if passer_team != team_id:
                if self.debug:
                    print(f"    ✗ Il passatore (Team {passer_team}) non è della stessa squadra")
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
    
    def _find_passer(self, pass_frame: int, ball_acquisition: List[int]) -> Optional[int]:
        """Trova chi ha passato la palla cercando chi aveva il possesso prima del passaggio."""
        # Cerca indietro chi aveva la palla prima del passaggio
        for frame in range(pass_frame - 1, max(0, pass_frame - 30), -1):
            if frame < len(ball_acquisition) and ball_acquisition[frame] != -1:
                return ball_acquisition[frame]
        return None
    
    def _find_receiver(self, pass_frame: int, ball_acquisition: List[int]) -> Optional[int]:
        """Trova chi ha ricevuto la palla cercando chi ha il possesso dopo il passaggio."""
        # Cerca avanti chi ha la palla dopo il passaggio
        for frame in range(pass_frame, min(len(ball_acquisition), pass_frame + 30)):
            if ball_acquisition[frame] != -1:
                return ball_acquisition[frame]
        return None
    
    def get_team_stats(self) -> Dict[int, Dict]:
        """Restituisce statistiche sugli assist per squadra."""
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
        """
        Esporta statistiche in due file CSV separati per Team 1 e Team 2.
        Include: tiri, assist, passaggi, intercetti, FG%.
        
        Args:
            output_path: Percorso base del file CSV (verrà creato stats_1.csv e stats_2.csv)
            shots: Lista di oggetti Shot
            passes: Lista con team_id del passaggio per frame (-1 se nessuno)
            interceptions: Lista con team_id dell'intercetto per frame (-1 se nessuno)
            player_assignment: Lista di dict {player_id: team_id} per frame
            ball_acquisition: Lista con player_id che ha la palla per frame (-1 se nessuno)
        """
        import csv
        import os
        from collections import defaultdict
        
        # Raggruppa assist per giocatore (passatore)
        assists_by_player = defaultdict(list)
        for assist in self.assists:
            assists_by_player[assist.passer_id].append(assist)
        
        # Raggruppa tiri per giocatore
        shots_by_player = defaultdict(list)
        if shots:
            for shot in shots:
                shots_by_player[shot.player_id].append(shot)
        
        # Conta passaggi per giocatore
        passes_by_player = defaultdict(int)
        if passes and ball_acquisition:
            prev_holder = -1
            for frame in range(1, len(passes)):
                if ball_acquisition[frame - 1] != -1:
                    prev_holder = ball_acquisition[frame - 1]
                if passes[frame] != -1 and prev_holder != -1:
                    passes_by_player[prev_holder] += 1
        
        # Conta intercetti per giocatore
        interceptions_by_player = defaultdict(int)
        if interceptions and ball_acquisition:
            for frame in range(len(interceptions)):
                if interceptions[frame] != -1 and ball_acquisition[frame] != -1:
                    interceptions_by_player[ball_acquisition[frame]] += 1
        
        # Determina team per ogni giocatore
        player_teams = {}
        if player_assignment:
            for frame_data in player_assignment:
                for player_id, team_id in frame_data.items():
                    if player_id not in player_teams:
                        player_teams[player_id] = team_id
        
        # Tutti i giocatori
        all_players = set(assists_by_player.keys()) | set(shots_by_player.keys()) | \
                      set(passes_by_player.keys()) | set(interceptions_by_player.keys())
        
        # Separa giocatori per team
        team_players = {1: [], 2: []}
        for player_id in all_players:
            team = player_teams.get(player_id, 0)
            if team in team_players:
                team_players[team].append(player_id)
        
        # Ordina giocatori per numero di azioni
        for team in team_players:
            team_players[team] = sorted(team_players[team], 
                                        key=lambda p: -(len(assists_by_player[p]) + len(shots_by_player[p]) + 
                                                       passes_by_player[p] + interceptions_by_player[p]))
        
        # Genera path per i due file
        base_dir = os.path.dirname(output_path)
        
        # Esporta un file per ogni team
        for team_id in [1, 2]:
            team_file = os.path.join(base_dir, f'stats_{team_id}.csv')
            self._export_team_csv(
                team_file, team_id, team_players[team_id],
                assists_by_player, shots_by_player, passes_by_player,
                interceptions_by_player, shots
            )
    
    def _export_team_csv(self, output_path: str, team_id: int, players: List[int],
                          assists_by_player, shots_by_player, passes_by_player,
                          interceptions_by_player, all_shots):
        """Esporta le statistiche di un singolo team in un file CSV."""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # === STATISTICHE TEAM ===
            team_shots = sum(len(shots_by_player[p]) for p in players)
            team_made = sum(sum(1 for s in shots_by_player[p] if s.made) for p in players)
            team_missed = team_shots - team_made
            team_assists = sum(len(assists_by_player[p]) for p in players)
            team_passes = sum(passes_by_player[p] for p in players)
            team_interceptions = sum(interceptions_by_player[p] for p in players)
            
            writer.writerow(['=' * 50])
            writer.writerow([f'STATISTICHE TEAM {team_id}'])
            writer.writerow(['=' * 50])
            writer.writerow(['Statistica', 'Valore'])
            writer.writerow(['Tiri totali', team_shots])
            writer.writerow(['Tiri segnati', team_made])
            writer.writerow(['Tiri sbagliati', team_missed])
            writer.writerow(['FG%', f'{(team_made/team_shots*100):.1f}%' if team_shots > 0 else 'N/A'])
            writer.writerow(['Assist totali', team_assists])
            writer.writerow(['Passaggi totali', team_passes])
            writer.writerow(['Intercetti totali', team_interceptions])
            writer.writerow([])
            
            # === STATISTICHE PER GIOCATORE ===
            writer.writerow(['=' * 50])
            writer.writerow(['STATISTICHE PER GIOCATORE'])
            writer.writerow(['=' * 50])
            writer.writerow([])
            
            for player_id in players:
                player_assists = assists_by_player[player_id]
                player_shots = shots_by_player[player_id]
                made_shots = [s for s in player_shots if s.made]
                missed_shots = [s for s in player_shots if not s.made]
                player_passes = passes_by_player[player_id]
                player_interceptions = interceptions_by_player[player_id]
                
                # Header giocatore
                writer.writerow([f'PLAYER {player_id}'])
                writer.writerow(['', 'Statistica', 'Valore'])
                writer.writerow(['', 'Tiri totali', len(player_shots)])
                writer.writerow(['', 'Tiri segnati', len(made_shots)])
                writer.writerow(['', 'Tiri sbagliati', len(missed_shots)])
                fg_pct = f'{(len(made_shots)/len(player_shots)*100):.1f}%' if player_shots else 'N/A'
                writer.writerow(['', 'FG%', fg_pct])
                writer.writerow(['', 'Assist', len(player_assists)])
                writer.writerow(['', 'Passaggi', player_passes])
                writer.writerow(['', 'Intercetti', player_interceptions])
                writer.writerow([])
                
                # Dettagli tiri
                if player_shots:
                    writer.writerow(['', 'DETTAGLI TIRI'])
                    writer.writerow(['', 'N.', 'Frame', 'Risultato', 'Posizione'])
                    for i, shot in enumerate(player_shots, 1):
                        result = 'SEGNATO' if shot.made else 'SBAGLIATO'
                        pos = f'({shot.position[0]:.0f}, {shot.position[1]:.0f})' if shot.position else 'N/A'
                        writer.writerow(['', f'Tiro #{i}', shot.frame_start, result, pos])
                    writer.writerow([])
                
                # Dettagli assist
                if player_assists:
                    writer.writerow(['', 'DETTAGLI ASSIST'])
                    writer.writerow(['', 'N.', 'Frame Pass', 'Scorer', 'Frame Tiro', 'Tempo'])
                    for i, assist in enumerate(player_assists, 1):
                        seconds = assist.pass_to_shot_frames / 30.0
                        writer.writerow(['', f'Assist #{i}', assist.frame, 
                                        f'Player {assist.scorer_id}', assist.shot_frame, f'{seconds:.2f}s'])
                    writer.writerow([])
                
                writer.writerow([])
        
        print(f"Statistiche Team {team_id} esportate in: {output_path}")
