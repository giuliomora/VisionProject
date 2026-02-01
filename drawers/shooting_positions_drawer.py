import cv2
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont


class ShootingPositionsDrawer:
    """Genera un'immagine del campo con le posizioni dei tiri."""
    
    def __init__(self, court_image_path: str, tactical_width: int = 300, tactical_height: int = 161):
        self.court_image_path = court_image_path
        self.tactical_width = tactical_width
        self.tactical_height = tactical_height
        
    def draw_shooting_positions(self, 
                                  shots: List,
                                  tactical_player_positions: List[Dict],
                                  output_path: str = "images/shooting_positions.pdf"):
        """
        Disegna le posizioni dei tiri sul campo.
        
        Args:
            shots: Lista di oggetti Shot con le informazioni sui tiri
            tactical_player_positions: Posizioni tattiche dei giocatori per ogni frame
            output_path: Percorso del file di output (PDF)
        """
        # Carica l'immagine del campo
        court_img = cv2.imread(self.court_image_path)
        if court_img is None:
            raise FileNotFoundError(f"Cannot load court image from {self.court_image_path}")
        
        # Converti in RGB per PIL
        court_img = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)
        
        # Crea oggetto PIL per disegno
        pil_img = Image.fromarray(court_img)
        draw = ImageDraw.Draw(pil_img)
        
        # Calcola fattori di scala tra coordinate tattiche e immagine
        img_width, img_height = pil_img.size
        scale_x = img_width / self.tactical_width
        scale_y = img_height / self.tactical_height
        
        print(f"Court image size: {img_width}x{img_height}")
        print(f"Tactical system: {self.tactical_width}x{self.tactical_height}")
        print(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
        
        # Prova a caricare un font, altrimenti usa quello di default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Colori
        made_color = (0, 200, 0)     # Verde per canestri segnati
        missed_color = (200, 0, 0)   # Rosso per tiri sbagliati
        
        # Statistiche per team
        team_stats = {1: {'made': 0, 'missed': 0}, 2: {'made': 0, 'missed': 0}}
        
        for shot in shots:
            # Filtra solo i tiri del Team 1
            if shot.team_id != 1:
                continue
                
            # Trova la posizione tattica del giocatore al frame del tiro
            frame_idx = shot.frame_start
            player_id = shot.player_id
            
            tactical_pos = None
            
            # Cerca la posizione tattica nel frame del tiro o nei frame vicini
            for offset in range(0, 30):  # Cerca fino a 30 frame prima
                check_frame = frame_idx - offset
                if check_frame >= 0 and check_frame < len(tactical_player_positions):
                    frame_positions = tactical_player_positions[check_frame]
                    if player_id in frame_positions:
                        tactical_pos = frame_positions[player_id]
                        break
            
            if tactical_pos is None:
                print(f"Warning: No tactical position found for shot at frame {frame_idx}, player {player_id}")
                continue
            
            # Scala le coordinate tattiche alle dimensioni dell'immagine
            x = int(tactical_pos[0] * scale_x)
            y = int(tactical_pos[1] * scale_y)
            
            print(f"Shot at tactical ({tactical_pos[0]:.1f}, {tactical_pos[1]:.1f}) -> image ({x}, {y})")
            
            # Aggiorna statistiche
            team_id = shot.team_id if shot.team_id in [1, 2] else 1
            if shot.made:
                team_stats[team_id]['made'] += 1
            else:
                team_stats[team_id]['missed'] += 1
            
            # Disegna il simbolo (scala anche la dimensione del simbolo)
            symbol_size = int(8 * min(scale_x, scale_y))
            line_width = max(2, int(3 * min(scale_x, scale_y)))
            
            if shot.made:
                # Cerchio verde "O" per canestri segnati
                draw.ellipse(
                    [x - symbol_size, y - symbol_size, x + symbol_size, y + symbol_size],
                    outline=made_color,
                    width=line_width
                )
            else:
                # X rossa per tiri sbagliati
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
        
        # Salva come PDF
        if output_path.endswith('.pdf'):
            pil_img.save(output_path, "PDF", resolution=100.0)
        else:
            # Se non è PDF, salva come immagine
            pil_img.save(output_path)
        
        print(f"Shooting positions exported to {output_path}")
        print(f"  Team 1: {team_stats[1]['made']} made, {team_stats[1]['missed']} missed")
        print(f"  Team 2: {team_stats[2]['made']} made, {team_stats[2]['missed']} missed")
        
        return pil_img
