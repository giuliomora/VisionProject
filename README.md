# VisionProject

**Basketball Video Analysis System using Computer Vision and Deep Learning**

A comprehensive computer vision application for analyzing basketball game videos, providing automated player tracking, shot detection, assist recognition, and tactical visualization.

---

## Overview

VisionProject processes basketball game footage to automatically extract game statistics and generate annotated video output. The system uses state-of-the-art deep learning models for object detection combined with custom algorithms for game event recognition.

### Key Features

- **Player Detection & Tracking** - Multi-object tracking using YOLO + ByteTrack
- **Ball Tracking** - Continuous ball position tracking with interpolation for occluded frames
- **Hoop Detection** - Automatic basketball hoop localization
- **Team Assignment** - Jersey color classification using Fashion-CLIP model
- **Shot Detection** - Automatic identification of shot attempts with made/missed determination
- **Assist Detection** - Recognition of passes leading to successful baskets
- **Pass & Interception Detection** - Team-based ball control analysis
- **Tactical View** - Bird's-eye court visualization using homography transformation
- **Statistics Export** - Game stats in CSV/TSV format

---

## Architecture

```
VisionProject/
├── main.py                         # Application entry point
├── configs/                        # Global configuration
│   └── configs.py                  # Paths and constants
├── models/                         # Pre-trained YOLO models
│   ├── player_detector.pt          # Player detection model
│   ├── ball_detector_model.pt      # Ball and hoop detection model
│   └── court_keypoint_detector.pt  # Court keypoint detection model
├── trackers/                       # Object tracking modules
│   ├── player_tracker.py           # PlayerTracker class
│   └── ball_tracker.py             # BallTracker class
├── drawers/                        # Visualization modules
│   ├── player_tracks_drawer.py     # Player annotations
│   ├── ball_tracks_drawer.py       # Ball annotations
│   ├── hoop_drawer.py              # Hoop annotations
│   ├── shooting_drawer.py          # Shot event overlays
│   ├── shooting_positions_drawer.py# Shot chart generation
│   ├── pass_and_interceptions_drawer.py # Stats overlay
│   ├── court_key_points_drawer.py  # Court keypoint visualization
│   ├── tactical_view_drawer.py     # Tactical minimap
│   ├── frame_number_drawer.py      # Frame counter
│   ├── assist_drawer.py            # Assist notifications
│   └── utils.py                    # Drawing utilities
├── team_assigner/                  # Team classification
│   └── team_assigner.py            # TeamAssigner (Fashion-CLIP)
├── ball_aquisition/                # Ball possession detection
│   └── ball_aquisition_detector.py # BallAquisitionDetector
├── pass_and_interception_detector/ # Pass/interception detection
│   └── pass_and_interception_detector.py
├── shooting_detector/              # Shot detection
│   └── shooting_detector.py        # ShootingDetector
├── assist_detector/                # Assist detection
│   └── assist_detector.py          # AssistDetector
├── court_keypoint_detector/        # Court keypoint detection
│   └── court_keypoint_detector.py  # CourtKeypointDetector
├── tactical_view_converter/        # Coordinate transformation
│   ├── tactical_view_converter.py  # TacticalViewConverter
│   └── homography.py               # Homography utilities
├── utils/                          # General utilities
│   ├── video_utils.py              # Video I/O
│   ├── stubs_utils.py              # Caching (pickle)
│   └── bbox_utils.py               # Bounding box utilities
├── input_videos/                   # Input video files
├── output_videos/                  # Processed video output
├── images/                         # Court images and shot charts
└── stubs/                          # Cached data and statistics
```

---

## Processing Pipeline

The system processes videos through the following stages:

### 1. Object Detection & Tracking

**PlayerTracker**
- Uses YOLO model for player detection
- ByteTrack algorithm for persistent ID assignment across frames
- Batch processing (20 frames) for efficiency
- Results cached to avoid recomputation

**BallTracker**
- YOLO-based ball and hoop detection
- Selects highest confidence detection per frame
- Filters outliers based on maximum allowed distance between frames
- Interpolates missing positions using pandas for smooth trajectories

### 2. Team Assignment

**TeamAssigner**
- Uses Fashion-CLIP model for jersey color classification
- Voting system across multiple frames for stable assignments
- Configurable team color descriptions (e.g., "white shirt", "black shirt")
- Cache resets every 30 frames for re-evaluation

### 3. Ball Possession Detection

**BallAquisitionDetector**
- Calculates minimum distance between ball and player bounding boxes
- Uses containment ratio (ball inside player bbox) for improved accuracy
- Requires minimum consecutive frames for confirmed possession
- Configurable thresholds for distance and containment

### 4. Court Keypoint Detection & Tactical View

**CourtKeypointDetector**
- YOLO-based detection of 18 court reference points
- Batch processing with caching support

**TacticalViewConverter**
- Validates keypoints using proportional distance checks
- Handles court half switching with temporal stability (30-frame threshold)
- Computes homography matrix for coordinate transformation
- Corrects positions based on nearest keypoint proximity

### 5. Event Detection

**PassAndInterceptionDetector**
- Pass: Ball possession changes between players of the same team
- Interception: Ball possession changes between different teams

**ShootingDetector**
- Monitors ball-hoop proximity and approach trajectory
- Determines shot outcome by tracking ball path through hoop
- Identifies shooter by backtracking to last player with possession
- Configurable parameters:
  - `hoop_proximity_threshold`: Distance to trigger shot detection (default: 120px)
  - `made_shot_threshold`: Distance for successful basket (default: 80px)
  - `min_frames_between_shots`: Cooldown period (default: 60 frames)

**AssistDetector**
- Links passes to subsequent made shots within time window
- Validates same-team relationship between passer and scorer
- Excludes self-passes
- Configurable `max_frames_pass_to_shot` (default: 150 frames, ~5 seconds)

### 6. Visualization

The system generates annotated video with:

| Component | Description |
|-----------|-------------|
| Player ellipses | Team-colored ellipses under each player |
| Player IDs | Numeric identifiers for tracking |
| Ball pointer | Green triangle above ball position |
| Possession indicator | Red triangle above player with ball |
| Hoop boxes | Orange rectangles around detected hoops |
| Shot banners | Green (made) / Red (missed) overlay with player info |
| Assist notifications | Centered popup showing passer → scorer |
| Stats panel | Bottom-left overlay with cumulative stats |
| Tactical minimap | Scaled court view with player positions |
| Frame counter | Current frame number |

### 7. Statistics Export

**Team Shooting Stats** (`team_shooting_stats.tsv`)
```
team_id    attempts    made    missed    percentage
1          15          8       7         53.3
2          12          5       7         41.7
```

**Individual Shots** (`all_shots.tsv`)
```
frame_start    team_id    player_id    made    hoop_id
245            1          5            True    1
512            2          12           False   2
```

**Game Stats** (`game_stats.csv`)
- Cumulative statistics per team across multiple video runs
- Includes: shots made/missed, assists, passes, interceptions

**Shot Chart** (`shooting_positions.pdf`)
- Visual representation of shot locations on court
- Green circles for made shots, red X for misses
- Accumulated across multiple video analyses

---

## Data Structures

### Shot
```python
@dataclass
class Shot:
    frame_start: int      # Shot start frame
    frame_end: int        # Shot end frame
    team_id: int          # Team (1 or 2)
    player_id: int        # Shooter ID
    made: bool            # True if basket made
    position: Tuple       # Shooter (x, y) position
    hoop_id: int          # Target hoop ID
```

### Assist
```python
@dataclass
class Assist:
    frame: int            # Pass frame
    passer_id: int        # Passer ID
    scorer_id: int        # Scorer ID
    team_id: int          # Team
    shot_frame: int       # Shot frame
    pass_to_shot_frames: int  # Frames between pass and shot
```

---

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd VisionProject

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **ultralytics** - YOLO object detection
- **supervision** - Detection utilities and ByteTrack
- **transformers** - Fashion-CLIP model
- **opencv-python** - Video processing
- **numpy** - Numerical operations
- **pandas** - Data manipulation and interpolation
- **Pillow** - Image processing

---

## Usage

### Basic Usage

```bash
python main.py --input_video input_videos/game.mp4
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_video` | `input_videos/video_5.mp4` | Input video path |
| `--output_video` | `output_videos/output.avi` | Output video path |
| `--stub_path` | `stubs/` | Cache directory |

### Example

```bash
python main.py \
    --input_video input_videos/quarter1.mp4 \
    --output_video output_videos/quarter1_analyzed.avi \
    --stub_path stubs/quarter1/
```

---

## Configuration

Edit `configs/configs.py` to modify:

```python
STUBS_DEFAULT_PATH = "stubs/"
PLAYER_DETECTOR_PATH = "models/player_detector.pt"
BALL_DETECTOR_PATH = "models/ball_detector_model.pt"
COURT_KEYPOINT_DETECTOR_PATH = "models/court_keypoint_detector.pt"
OUTPUT_VIDEO_PATH = "output_videos/output.avi"
```

### Team Colors

Modify in `main.py` when initializing `TeamAssigner`:

```python
team_assigner = TeamAssigner(
    team_1_class_name="white shirt",
    team_2_class_name="black shirt"
)
```

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| Annotated video | `output_videos/` | Video with all visual annotations |
| Team stats | `stubs/team_shooting_stats.tsv` | Team shooting percentages |
| All shots | `stubs/all_shots.tsv` | Individual shot records |
| Game stats | `stubs/game_stats.csv` | Cumulative game statistics |
| Shot chart | `images/shooting_positions.pdf` | Visual shot distribution |

---

## Caching System

The system uses pickle-based caching to avoid recomputing expensive operations:

| Cache File | Contents |
|------------|----------|
| `player_track_stubs.pkl` | Player tracking results |
| `ball_track_stubs.pkl` | Ball tracking results |
| `hoop_detections.pkl` | Hoop detection results |
| `court_key_points_stub.pkl` | Court keypoint detections |
| `player_assignment_stub.pkl` | Team assignments |
| `accumulated_shots.pkl` | Shot history across runs |
| `accumulated_stats.pkl` | Cumulative game statistics |

To force recomputation, delete the relevant cache files or change `read_from_stub=False`.

---

## Models

The system requires three pre-trained YOLO models:

1. **player_detector.pt** - Trained for basketball player detection
2. **ball_detector_model.pt** - Trained for ball and hoop detection (classes: Ball, Hoop)
3. **court_keypoint_detector.pt** - Trained for court keypoint detection (18 keypoints)

Place models in the `models/` directory.

---

## Technical Details

### Court Keypoints

18 reference points are detected on the basketball court:

```
Keypoints 0-5: Left sideline (top to bottom)
Keypoints 6-7: Center court (bottom, top)
Keypoints 8-9: Left paint area
Keypoints 10-15: Right sideline (bottom to top)
Keypoints 16-17: Right paint area
```

### Homography Transformation

Player positions are transformed from video coordinates to tactical view using:

1. Detection of visible court keypoints
2. Matching to known court coordinates (28m × 15m standard court)
3. Homography matrix computation with minimum 4 point correspondences
4. Position correction based on nearest keypoint proximity

### Shot Detection Algorithm

1. Monitor ball-hoop distance continuously
2. Detect approach: ball moving toward hoop AND within proximity threshold
3. Track trajectory through hoop region
4. Classify outcome: ball crosses below hoop center → MADE, otherwise → MISSED
5. Identify shooter: backtrack possession to find last player with ball

---

## Performance Considerations

- **GPU Acceleration**: YOLO inference significantly faster with CUDA
- **Batch Processing**: Detection runs in batches of 20 frames
- **Caching**: First run computes all detections; subsequent runs load from cache
- **Memory**: Large videos may require significant RAM for frame storage

---

## License

This project was developed for educational purposes as part of a Computer Vision and Cognitive Systems course.

---

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection framework
- [Supervision](https://github.com/roboflow/supervision) - Detection utilities
- [Fashion-CLIP](https://huggingface.co/patrickjohncyh/fashion-clip) - Visual classification model
