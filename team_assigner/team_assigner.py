from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
import sys
sys.path.append("../")
from utils import read_stubs, save_stubs



class TeamAssigner:
    def __init__(self, 
                 team_1_class_name = "white shirt",
                 team_2_class_name = "dark blue shirt",):
        
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

        self.player_team_dict = {}

    def load_model(self):
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        classes = [self.team_1_class_name, self.team_2_class_name] # white and blue
        
        inputs = self.processor(text=classes, images=pil_image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        class_name = classes[probs.argmax(dim=1)[0]]
        return class_name
    
    def get_player_team(self, frame, player_bbox, player_id):
        
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
    
    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        
        player_assignments = read_stubs(read_from_stub, stub_path)
        if player_assignments is not None:
             if len(player_assignments) == len(video_frames):
                 return player_assignments
        
        self.load_model()

        player_assignments = []

        for frame_numb, player_tracks in enumerate(player_tracks):
            player_assignments.append({})

            if frame_numb %50 == 0:
                self.player_team_dict = {}

            for player_id, track in player_tracks.items():
                team = self.get_player_team(video_frames[frame_numb], track['bbox'], player_id)
                player_assignments[frame_numb][player_id] = team

        save_stubs(stub_path, player_assignments)

        return player_assignments
        