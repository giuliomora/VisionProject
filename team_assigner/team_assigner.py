from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict, Counter

import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class TeamAssigner:
    """
    A class that assigns players to teams based on their jersey colors using visual analysis.

    The class uses a pre-trained vision model to classify players into teams based on their
    appearance. It maintains a consistent team assignment for each player across frames
    using a voting system for stability.

    Attributes:
        team_colors (dict): Dictionary storing team color information.
        player_team_dict (dict): Dictionary mapping player IDs to their team assignments.
        team_1_class_name (str): Description of Team 1's jersey appearance.
        team_2_class_name (str): Description of Team 2's jersey appearance.
    """
    def __init__(self,
                 team_1_class_name= "white shirt",
                 team_2_class_name= "black shirt",
                 ):
        """
        Initialize the TeamAssigner with specified team jersey descriptions.

        Args:
            team_1_class_name (str): Description of Team 1's jersey appearance.
            team_2_class_name (str): Description of Team 2's jersey appearance.
        """
        self.team_colors = {}
        self.player_team_dict = {}        
    
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

    def load_model(self):
        """
        Loads the pre-trained vision model for jersey color classification.
        """
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_player_color(self,frame,bbox):
        """
        Analyzes the jersey color of a player within the given bounding box.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            bbox (tuple): Bounding box coordinates of the player.

        Returns:
            str: The classified jersey color/description.
        """
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        # Convert to PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image = pil_image

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1) 


        class_name=  classes[probs.argmax(dim=1)[0]]

        return class_name

    def get_player_team(self,frame,player_bbox,player_id):
        """
        Gets the team assignment for a player, using cached results if available.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            player_bbox (tuple): Bounding box coordinates of the player.
            player_id (int): Unique identifier for the player.

        Returns:
            int: Team ID (1 or 2) assigned to the player.
        """
        if player_id in self.player_team_dict:
          return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id=2
        if player_color==self.team_1_class_name:
            team_id=1

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(self,video_frames,player_tracks,read_from_stub=False, stub_path=None):
        """
        Processes all video frames to assign teams to players, with optional caching.
        Uses a voting system to stabilize assignments across frames.

        Args:
            video_frames (list): List of video frames to process.
            player_tracks (list): List of player tracking information for each frame.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries mapping player IDs to team assignments for each frame.
        """
        
        player_assignment = read_stub(read_from_stub,stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()

        # Sistema di voting per stabilizzare le assegnazioni
        player_votes = defaultdict(lambda: {1: 0, 2: 0})
        
        # Prima passata: raccogli voti per ogni giocatore
        print("Raccolta voti per assegnazione team...")
        for frame_num, player_track in enumerate(player_tracks):
            # Reset cache ogni 30 frame per permettere ri-valutazione
            if frame_num % 30 == 0:
                self.player_team_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                player_votes[player_id][team] += 1
            
            if frame_num % 100 == 0:
                print(f"Processed frame {frame_num}/{len(player_tracks)}")
        
        # Determina team finale per ogni giocatore basato sui voti
        final_team_assignment = {}
        print("\nVoti per giocatore:")
        for player_id, votes in player_votes.items():
            total = votes[1] + votes[2]
            if total > 0:
                print(f"  Player {player_id}: Team1={votes[1]} ({votes[1]/total*100:.0f}%), Team2={votes[2]} ({votes[2]/total*100:.0f}%)")
            
            if votes[1] > votes[2]:
                final_team_assignment[player_id] = 1
            elif votes[2] > votes[1]:
                final_team_assignment[player_id] = 2
            else:
                final_team_assignment[player_id] = 1  # Default in caso di parità
        
        # Seconda passata: applica assegnazioni finali stabili
        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})
            for player_id in player_track.keys():
                if player_id in final_team_assignment:
                    player_assignment[frame_num][player_id] = final_team_assignment[player_id]
                else:
                    player_assignment[frame_num][player_id] = 1  # Default
        
        # Stampa statistiche finali
        self._print_assignment_stats(player_assignment)
        
        save_stub(stub_path,player_assignment)

        return player_assignment
    
    def _print_assignment_stats(self, player_assignment):
        """Stampa statistiche sulle assegnazioni per debug."""
        all_assignments = []
        for frame in player_assignment:
            all_assignments.extend(frame.values())
        
        if all_assignments:
            team_counts = Counter(all_assignments)
            total = len(all_assignments)
            print(f"\nStatistiche assegnazione:")
            print(f"  Team 1: {team_counts.get(1, 0)} ({team_counts.get(1, 0)/total*100:.1f}%)")
            print(f"  Team 2: {team_counts.get(2, 0)} ({team_counts.get(2, 0)/total*100:.1f}%)")