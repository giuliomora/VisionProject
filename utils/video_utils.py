import cv2
import os

def read_video(video_path):

    """Reads a video from the given path and returns the video capture object."""
    
    cap = cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames

def save_video(frames, output_path, ):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out= cv2.VideoWriter(output_path, fourcc, 24.0, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        out.write(frame)
    out.release()

