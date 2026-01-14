import os
from ultralytics import YOLO

model = YOLO("models/ball_detector_model.pt") #il modello della palla
video_path = "input_videos/video_1.mp4"
#.predict for detection, .track for tracking
results = model.track(video_path, save=True)
print(results)
print("########################")
for box in results[0].boxes:
    print(box.xyxy)
