# VisionProject

**Progetto di Computer Vision and Cognitive Systems**

Sistema di tracking dei giocatori in video di partite sportive utilizzando tecniche di deep learning e computer vision.

---

## 📋 Descrizione del Progetto

VisionProject è un'applicazione di computer vision che analizza video di partite sportive per:
- **Rilevare** i giocatori presenti in ogni frame
- **Tracciare** i movimenti dei giocatori nel tempo (tracking multi-oggetto)
- **Visualizzare** le tracce con annotazioni grafiche (ellissi e ID)

Il sistema utilizza modelli YOLO pre-addestrati per la detection e l'algoritmo ByteTrack per il tracking persistente degli oggetti tra i frame.

---

## 🏗️ Architettura del Progetto

```
VisionProject/
├── main.py                    # Entry point dell'applicazione
├── models/                    # Modelli YOLO pre-addestrati
│   ├── player_detector.pt     # Modello per detection giocatori
│   ├── ball_detector_model.pt # Modello per detection pallone (futuro)
│   └── court_keypoint_detector.pt # Modello per keypoint campo (futuro)
├── trackers/                  # Moduli di tracking
│   ├── __init__.py
│   └── playerTracker.py       # Classe PlayerTracker
├── drawers/                   # Moduli di visualizzazione
│   ├── __init__.py
│   ├── player_tracks_drawer.py # Classe PlayerTracksDrawer
│   └── utils.py               # Funzioni di disegno (ellissi, rettangoli)
├── utils/                     # Utility generiche
│   ├── __init__.py
│   ├── video_utils.py         # Lettura/scrittura video
│   ├── stubs_utils.py         # Gestione cache (stubs)
│   └── bbox_utils.py          # Utility per bounding box
├── input_videos/              # Video di input
├── output_videos/             # Video processati
├── stubs/                     # Cache delle tracce (pickle)
└── .venv/                     # Virtual environment (non versionato)
```

---

## 🔧 Componenti Principali

### 1. **main.py** - Entry Point
Il file principale che orchestra l'intera pipeline:
1. Carica il video di input
2. Inizializza il tracker dei giocatori
3. Esegue il tracking (o carica dalla cache)
4. Disegna le annotazioni sui frame
5. Salva il video di output

### 2. **PlayerTracker** (`trackers/playerTracker.py`)
Classe responsabile del rilevamento e tracking dei giocatori:

- **`__init__(model_path)`**: Inizializza il modello YOLO e il tracker ByteTrack
- **`detect_frames(frames)`**: Esegue la detection su tutti i frame in batch da 20
- **`get_object_tracks(frames, read_from_stub, stub_path)`**: 
  - Se esiste una cache (stub), la carica per evitare ricalcoli
  - Altrimenti esegue detection + tracking
  - Salva i risultati in cache per usi futuri

**Output**: Lista di dizionari, uno per frame, con struttura:
```python
{
    track_id: {"box": [x1, y1, x2, y2]},
    ...
}
```

### 3. **PlayerTracksDrawer** (`drawers/player_tracks_drawer.py`)
Classe per la visualizzazione delle tracce:

- **`draw(video_frames, tracks)`**: Per ogni frame, disegna un'ellisse colorata sotto ogni giocatore con il suo ID di tracking

### 4. **Funzioni di Disegno** (`drawers/utils.py`)
- **`draw_ellypse(frame, bbox, color, track_id)`**: 
  - Disegna un'ellisse ai piedi del giocatore (posizione y2 del bounding box)
  - Aggiunge un rettangolo con l'ID del track
  - L'ellisse ha forma proporzionale alla larghezza del bounding box

### 5. **Utility Video** (`utils/video_utils.py`)
- **`read_video(video_path)`**: Legge un video e restituisce una lista di frame (array numpy)
- **`save_video(frames, output_path)`**: Salva i frame in un file AVI (codec XVID, 24 fps)

### 6. **Sistema di Cache - Stubs** (`utils/stubs_utils.py`)
Sistema di caching per evitare ricalcoli costosi:
- **`save_stubs(stub_path, object)`**: Salva un oggetto Python in formato pickle
- **`read_stubs(read_from_stub, stub_path)`**: Carica un oggetto dalla cache se esiste

### 7. **Utility Bounding Box** (`utils/bbox_utils.py`)
- **`get_center_of_bbox(bbox)`**: Calcola il centro di un bounding box
- **`get_bbox_width(bbox)`**: Calcola la larghezza di un bounding box

---

## 🛠️ Tecnologie Utilizzate

| Tecnologia | Versione | Utilizzo |
|------------|----------|----------|
| **Python** | 3.12 | Linguaggio principale |
| **Ultralytics YOLO** | 8.4.6 | Object detection |
| **Supervision** | 0.27.0 | ByteTrack e utility per detection |
| **OpenCV** | 4.13.0 | Elaborazione video e disegno |
| **PyTorch** | 2.9.1 | Backend per YOLO |

---

## 🚀 Installazione

### 1. Clonare il repository
```bash
git clone <url-repository>
cd VisionProject
```

### 2. Creare e attivare il virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installare le dipendenze
```bash
pip install ultralytics supervision opencv-python
```

### 4. Posizionare i modelli
Inserire i file `.pt` dei modelli nella cartella `models/`

---

## ▶️ Esecuzione

```bash
python main.py
```

Il programma:
1. Legge `input_videos/video_1.mp4`
2. Esegue il tracking (o usa la cache da `stubs/player_tracks.stub.pkl`)
3. Genera `output_videos/output_video.avi` con le annotazioni

---

## 📊 Pipeline di Elaborazione

```
┌─────────────────┐
│  Video Input    │
│  (MP4)          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  read_video()   │  Estrae tutti i frame
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PlayerTracker  │
│  ┌────────────┐ │
│  │ YOLO       │ │  Detection giocatori
│  │ Detection  │ │  (batch da 20 frame)
│  └─────┬──────┘ │
│        ▼        │
│  ┌────────────┐ │
│  │ ByteTrack  │ │  Assegna ID persistenti
│  │ Tracking   │ │  ai giocatori
│  └────────────┘ │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PlayerTracks   │  Cache opzionale
│  Drawer         │  (stubs/*.pkl)
│  ┌────────────┐ │
│  │ draw_      │ │  Ellissi + ID
│  │ ellypse()  │ │
│  └────────────┘ │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  save_video()   │  Output AVI
│  (XVID, 24fps)  │
└─────────────────┘
```

---

## 🔮 Sviluppi Futuri

Il progetto prevede l'implementazione di:
- **Ball Tracking**: Rilevamento e tracking del pallone (`ball_detector_model.pt`)
- **Court Detection**: Rilevamento dei keypoint del campo (`court_keypoint_detector.pt`)
- **Analisi tattica**: Posizionamento dei giocatori rispetto al campo
- **Statistiche**: Velocità, distanze percorse, heat map

---

## 📁 File Ignorati (.gitignore)

```
models/*.pt          # Modelli troppo grandi
input_videos/        # Video di input
runs/                # Output YOLO
__pycache__/         # Cache Python
.venv/               # Virtual environment
```

---

## 👤 Autore

Progetto sviluppato per il corso di **Computer Vision and Cognitive Systems**
