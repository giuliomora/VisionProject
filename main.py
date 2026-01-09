import os
from ultralytics import YOLO

model = YOLO("yolov8s.pt") #il modello che penso sia meglio usare con il nostro hardware, magari si può upgradare

video_path = "input_videos/video_1.mp4"

results = model.predict(video_path, save=True)
print(results)
print("########################")
for box in results[0].boxes:
    print(box.xyxy)

#gergh