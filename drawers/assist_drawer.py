import cv2
import numpy as np
from typing import List

class AssistDrawer:
    """Disegna gli assist sulla sequenza di frame."""
    
    def __init__(self, display_duration: int = 60):
        """
        Args:
            display_duration: Numero di frame per cui mostrare la notifica dell'assist
        """
        self.display_duration = display_duration
    
    def draw(self, video_frames: List, assists: List) -> List:
        """
        Disegna gli assist sui frame del video.
        
        Args:
            video_frames: Lista dei frame video
            assists: Lista di oggetti Assist
        
        Returns:
            Lista dei frame con gli assist disegnati
        """
        output_video_frames = []
        
        # Crea un dizionario per sapere quando mostrare gli assist
        # Mostra l'assist dal frame del passaggio per display_duration frames
        assist_display = {}
        for assist in assists:
            for frame in range(assist.frame, min(assist.frame + self.display_duration, len(video_frames))):
                if frame not in assist_display:
                    assist_display[frame] = []
                assist_display[frame].append(assist)
        
        for frame_num, frame in enumerate(video_frames):
            frame_drawn = frame.copy()
            
            # Disegna statistiche assist cumulative
            frame_drawn = self.draw_assist_stats(frame_drawn, frame_num, assists)
            
            # Se c'è un assist attivo in questo frame, mostra notifica
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
        """Disegna una notifica per l'assist sul frame."""
        overlay = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Calcola opacità per fade out
        frames_since_pass = current_frame - assist.frame
        alpha = max(0.3, 1.0 - (frames_since_pass / self.display_duration) * 0.7)
        
        # Box notifica in alto al centro
        box_width = 350
        box_height = 80
        box_x1 = (frame_width - box_width) // 2
        box_y1 = 20
        box_x2 = box_x1 + box_width
        box_y2 = box_y1 + box_height
        
        # Colore del team (team 1 = blu, team 2 = rosso)
        if assist.team_id == 1:
            color = (255, 150, 0)  # Blu
            text_color = (255, 255, 255)
        else:
            color = (0, 100, 255)  # Rosso
            text_color = (255, 255, 255)
        
        # Disegna box con bordo
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), color, -1)
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)
        
        # Testo
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
        
        # Applica overlay con trasparenza
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def draw_assist_stats(self, frame, frame_num: int, assists: List):
        """Disegna le statistiche cumulative degli assist."""
        # Conta assist per team fino a questo frame
        team1_assists = sum(1 for a in assists if a.frame <= frame_num and a.team_id == 1)
        team2_assists = sum(1 for a in assists if a.frame <= frame_num and a.team_id == 2)
        
        overlay = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Posiziona box statistiche (sotto il box di pass/interceptions)
        rect_x1 = int(frame_width * 0.56)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.82)
        rect_y2 = int(frame_height * 0.90)
        
        text_x = int(frame_width * 0.58)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)
        
        # Box semi-trasparente
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Testo statistiche
        cv2.putText(
            frame,
            f"Team 1 - Assists: {team1_assists}",
            (text_x, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        cv2.putText(
            frame,
            f"Team 2 - Assists: {team2_assists}",
            (text_x, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        return frame
