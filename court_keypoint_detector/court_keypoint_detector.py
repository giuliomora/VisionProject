from ultralytics import YOLO
import supervision as sv
import numpy as np
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class CourtKeypointDetector:
    """Rileva i keypoint del campo usando un modello YOLO."""
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def get_court_keypoints(self, frames,read_from_stub=False, stub_path=None):
        """Rileva keypoint del campo. Può leggere da stub o eseguire il modello YOLO."""
        court_keypoints = read_stub(read_from_stub,stub_path)
        if court_keypoints is not None:
            if len(court_keypoints) == len(frames):
                return court_keypoints
        
        batch_size=20
        court_keypoints = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            for detection in detections_batch:
                court_keypoints.append(detection.keypoints)

        save_stub(stub_path,court_keypoints)
        
        return court_keypoints