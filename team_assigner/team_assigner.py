import cv2
import numpy as np
from collections import Counter

import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class TeamAssigner:
    """Assegna i giocatori alle squadre in base al colore della maglia."""
    def __init__(self,
                 team_1_class_name= "white shirt",
                 team_2_class_name= "dark blue shirt",
                 ):
        """
        Inizializza l'assegnatore di squadre con le descrizioni delle maglie.
        """
        self.team_colors = {}
        self.player_team_dict = {}        
    
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        
        # Colori squadre (verranno inizializzati automaticamente)
        self.team_1_color = None
        self.team_2_color = None
        self.team_colors_initialized = False

    def load_model(self):
        """Placeholder per compatibilità - non carica nessun modello."""
        pass

    def _extract_shirt_region(self, frame, bbox):
        """Estrae la regione della maglia (parte superiore del bounding box)."""
        x1, y1, x2, y2 = map(int, bbox)
        
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return None
        
        # Estrai parte centrale (maglia)
        margin_x = int(width * 0.3)
        top_y = int(height * 0.2)
        bottom_y = int(height * 0.5)
        
        shirt_x1 = x1 + margin_x
        shirt_x2 = x2 - margin_x
        shirt_y1 = y1 + top_y
        shirt_y2 = y1 + bottom_y
        
        if shirt_x2 <= shirt_x1 or shirt_y2 <= shirt_y1:
            return None
            
        shirt_region = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
        
        if shirt_region.size == 0:
            return None
            
        return shirt_region

    def _remove_court_colors(self, image_hsv):
        """Rimuove i pixel del campo (verde/marrone/arancione)."""
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
        
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([25, 255, 200])
        brown_mask = cv2.inRange(image_hsv, lower_brown, upper_brown)
        
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 50])
        dark_mask = cv2.inRange(image_hsv, lower_dark, upper_dark)
        
        invalid_mask = green_mask | brown_mask | dark_mask
        valid_mask = cv2.bitwise_not(invalid_mask)
        
        return valid_mask

    def _get_dominant_color(self, shirt_region):
        """Calcola il colore dominante della maglia usando K-means in HSV."""
        if shirt_region is None or shirt_region.size == 0:
            return None
        
        hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
        valid_mask = self._remove_court_colors(hsv)
        valid_pixels = hsv[valid_mask > 0]
        
        if len(valid_pixels) < 10:
            return None
        
        pixels = valid_pixels.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        try:
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        except cv2.error:
            return None
        
        label_counts = Counter(labels.flatten())
        dominant_label = label_counts.most_common(1)[0][0]
        dominant_color = centers[dominant_label]
        
        return dominant_color

    def _color_distance(self, color1, color2):
        """Calcola la distanza tra due colori HSV."""
        if color1 is None or color2 is None:
            return float('inf')
        
        h1, s1, v1 = color1
        h2, s2, v2 = color2
        
        h_diff = min(abs(h1 - h2), 180 - abs(h1 - h2))
        distance = np.sqrt((h_diff * 2) ** 2 + (s1 - s2) ** 2 + (v1 - v2) ** 2)
        
        return distance

    def _initialize_team_colors(self, video_frames, player_tracks, sample_frames=5):
        """Inizializza i colori delle squadre analizzando i primi frame."""
        all_colors = []
        
        frame_indices = np.linspace(0, min(len(video_frames) - 1, 100), sample_frames, dtype=int)
        
        for frame_idx in frame_indices:
            frame = video_frames[frame_idx]
            player_track = player_tracks[frame_idx]
            
            for player_id, track in player_track.items():
                bbox = track['bbox']
                shirt_region = self._extract_shirt_region(frame, bbox)
                color = self._get_dominant_color(shirt_region)
                
                if color is not None:
                    all_colors.append(color)
        
        if len(all_colors) < 4:
            self.team_1_color = np.array([0, 0, 255])
            self.team_2_color = np.array([110, 200, 100])
            self.team_colors_initialized = True
            return
        
        colors_array = np.array(all_colors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        try:
            _, labels, centers = cv2.kmeans(colors_array, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        except cv2.error:
            self.team_1_color = np.array([0, 0, 255])
            self.team_2_color = np.array([110, 200, 100])
            self.team_colors_initialized = True
            return
        
        self.team_1_color = centers[0]
        self.team_2_color = centers[1]
        self.team_colors_initialized = True
        
        print(f"Team 1 color (HSV): {self.team_1_color}")
        print(f"Team 2 color (HSV): {self.team_2_color}")

    def get_player_color(self, frame, bbox):
        """Analizza il colore della maglia del giocatore."""
        shirt_region = self._extract_shirt_region(frame, bbox)
        color = self._get_dominant_color(shirt_region)
        
        if color is None:
            return self.team_1_class_name
        
        dist_team_1 = self._color_distance(color, self.team_1_color)
        dist_team_2 = self._color_distance(color, self.team_2_color)
        
        if dist_team_1 < dist_team_2:
            return self.team_1_class_name
        else:
            return self.team_2_class_name

    def get_player_team(self, frame, player_bbox, player_id):
        """Ottiene l'assegnazione di squadra del giocatore (con cache)."""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        """Assegna le squadre ai giocatori in tutti i frame."""
        
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()
        
        # Inizializza i colori delle squadre
        if not self.team_colors_initialized:
            print("Inizializzazione colori squadre...")
            self._initialize_team_colors(video_frames, player_tracks)

        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):        
            player_assignment.append({})
            
            if frame_num % 50 == 0:
                self.player_team_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                player_assignment[frame_num][player_id] = team
            
            if frame_num % 100 == 0:
                print(f"Processed frame {frame_num}/{len(player_tracks)}")
        
        save_stub(stub_path, player_assignment)

        return player_assignment