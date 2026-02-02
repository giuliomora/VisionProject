from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class TeamAssigner:
    """Assigns players to teams based on jersey colors using CLIP vision model."""
    
    def __init__(self,
                 team_1_class_name= "dark blue shirt",
                 team_2_class_name= "white shirt"):
        self.team_colors = {}
        self.player_team_dict = {}        
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

    def load_model(self):
        """Load CLIP model for jersey classification."""
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_player_color(self, frame, bbox):
        """Classify player jersey color from bbox."""
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image = pil_image

        classes = [self.team_1_class_name, self.team_2_class_name]
        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        class_name = classes[probs.argmax(dim=1)[0]]

        return class_name

    def get_player_team(self, frame, player_bbox, player_id):
        """Get team assignment for player (cached)."""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        """Assign teams to all players using voting system for stability."""
        
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()

        # Voting for stable assignments
        player_votes = defaultdict(lambda: {1: 0, 2: 0})
        
        # First pass: collect votes
        print("Collecting team assignment votes...")
        for frame_num, player_track in enumerate(player_tracks):
            # Reset cache every 30 frames
            if frame_num % 30 == 0:
                self.player_team_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                player_votes[player_id][team] += 1
            
            if frame_num % 100 == 0:
                print(f"Processed frame {frame_num}/{len(player_tracks)}")
        
        # Determine final team by votes
        final_team_assignment = {}
        print("\nVotes per player:")
        for player_id, votes in player_votes.items():
            total = votes[1] + votes[2]
            if total > 0:
                print(f"  Player {player_id}: Team1={votes[1]} ({votes[1]/total*100:.0f}%), Team2={votes[2]} ({votes[2]/total*100:.0f}%)")
            
            if votes[1] > votes[2]:
                final_team_assignment[player_id] = 1
            elif votes[2] > votes[1]:
                final_team_assignment[player_id] = 2
            else:
                final_team_assignment[player_id] = 1
        
        # Second pass: apply stable assignments
        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})
            for player_id in player_track.keys():
                if player_id in final_team_assignment:
                    player_assignment[frame_num][player_id] = final_team_assignment[player_id]
                else:
                    player_assignment[frame_num][player_id] = 1
        
        save_stub(stub_path, player_assignment)

        return player_assignment