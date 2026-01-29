from .utils import draw_traingle

class BallTracksDrawer:
    """Disegna i tracciati della palla sui frame video."""

    def __init__(self):
        """Inizializza il disegnatore dei tracciati della palla."""
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frames, tracks):
        """Disegna i tracciati della palla sui frame."""
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracks[frame_num]

            # Draw ball 
            for _, ball in ball_dict.items():
                if ball["bbox"] is None:
                    continue
                frame = draw_traingle(frame, ball["bbox"],self.ball_pointer_color)

            output_video_frames.append(frame)
            
        return output_video_frames