# VisionProject

**Progetto di Computer Vision and Cognitive Systems**

Sistema di tracking dei giocatori e del pallone in video di partite sportive utilizzando tecniche di deep learning e computer vision.

---

## 📋 Descrizione del Progetto

VisionProject è un'applicazione di computer vision che analizza video di partite sportive per:
- **Rilevare** i giocatori e il pallone presenti in ogni frame
- **Tracciare** i movimenti dei giocatori nel tempo (tracking multi-oggetto con ByteTrack)
- **Tracciare** la posizione del pallone (detection con selezione per confidence)
- **Assegnare i giocatori alle squadre** in base al colore della maglia (utilizzando Fashion-CLIP)
- **Visualizzare** le tracce con annotazioni grafiche (ellissi colorate per squadra, triangoli per il pallone)

Il sistema utilizza modelli YOLO pre-addestrati per la detection, l'algoritmo ByteTrack per il tracking persistente degli oggetti tra i frame, e il modello Fashion-CLIP per il riconoscimento dei colori delle maglie.

---

## 🏗️ Architettura del Progetto

```
VisionProject/
├── main.py                    # Entry point dell'applicazione
├── models/                    # Modelli YOLO pre-addestrati
│   ├── player_detector.pt     # Modello per detection giocatori
│   ├── ball_detector_model.pt # Modello per detection pallone
│   └── court_keypoint_detector.pt # Modello per keypoint campo (futuro)
├── trackers/                  # Moduli di tracking
│   ├── __init__.py
│   ├── playerTracker.py       # Classe PlayerTracker
│   └── ballTracker.py         # Classe BallTracker
├── drawers/                   # Moduli di visualizzazione
│   ├── __init__.py
│   ├── player_tracks_drawer.py # Classe PlayerTracksDrawer
│   ├── ball_tracks_drawer.py  # Classe BallTracksDrawer
│   ├── team_ball_control_drawer.py # Classe TeamBallControlDrawer
│   └── utils.py               # Funzioni di disegno (ellissi, triangoli)
├── team_assigner/             # Modulo assegnazione squadre
│   ├── __init__.py
│   └── team_assigner.py       # Classe TeamAssigner (Fashion-CLIP)
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
2. Inizializza i tracker (giocatori e pallone)
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

### 3. **BallTracker** (`trackers/ballTracker.py`)
Classe responsabile del rilevamento del pallone:

- **`__init__(model_path)`**: Inizializza il modello YOLO per la detection del pallone
- **`detect_frames(frames)`**: Esegue la detection su tutti i frame in batch da 20
- **`get_object_tracks(frames, read_from_stub, stub_path)`**: 
  - Se esiste una cache (stub), la carica per evitare ricalcoli
  - Seleziona la detection con confidence massima per ogni frame
  - Salva i risultati in cache per usi futuri
- **`remuve_wrong_detections(ball_positions)`**: Filtra outlier spaziali rimuovendo rimbalzi irreali tra frame (distanza massima scalata per il gap tra frame)
- **`interpolate_ball_positions(ball_positions)`**: Usa `pandas` per interpolare e backfillare le posizioni mancanti del pallone

**Output**: Lista di dizionari, uno per frame, con struttura:
```python
{
    1: {"bbox": [x1, y1, x2, y2]},  # Una sola detection per frame
}
```

### 4. **PlayerTracksDrawer** (`drawers/player_tracks_drawer.py`)
Classe per la visualizzazione delle tracce dei giocatori:

- **`draw(video_frames, tracks, player_assignments)`**: Per ogni frame, disegna un'ellisse colorata (in base alla squadra) sotto ogni giocatore con il suo ID di tracking

### 5. **BallTracksDrawer** (`drawers/ball_tracks_drawer.py`)
Classe per la visualizzazione della posizione del pallone:

- **`draw(video_frames, tracks)`**: Per ogni frame, disegna un triangolo verde sopra il pallone

### 6. **TeamBallControlDrawer** (`drawers/team_ball_control_drawer.py`)
Classe per il calcolo e visualizzazione delle statistiche di possesso palla per squadra:

- **`get_team_ball_control(player_assignment, ball_aquisition)`**: Calcola quale squadra ha il controllo del pallone per ogni frame, restituendo array (1=Team1, 2=Team2, -1=nessuno)
- **`draw(video_frames, player_assignment, ball_aquisition)`**: Disegna overlay semi-trasparente con percentuali di possesso palla per entrambe le squadre
- **`draw_frame(frame, frame_num, team_ball_control)`**: Disegna statistiche su singolo frame con rettangolo semi-trasparente e testo percentuale

**Output**: Overlay bottom-right con statistiche real-time tipo:
```
Team 1 Ball Control: 45.23%
Team 2 Ball Control: 54.77%
```

### 7. **Funzioni di Disegno** (`drawers/utils.py`)
- **`draw_ellypse(frame, bbox, color, track_id)`**: 
  - Disegna un'ellisse ai piedi del giocatore (posizione y2 del bounding box)
  - Aggiunge un rettangolo con l'ID del track
  - L'ellisse ha forma proporzionale alla larghezza del bounding box
- **`draw_triangle(frame, bbox, color)`**:
  - Disegna un triangolo sopra il pallone (posizione y1 del bounding box)
  - Il triangolo punta verso il basso per indicare la posizione

### 7. **Utility Video** (`utils/video_utils.py`)
- **`read_video(video_path)`**: Legge un video e restituisce una lista di frame (array numpy)
- **`save_video(frames, output_path)`**: Salva i frame in un file AVI (codec XVID, 24 fps)

### 8. **Sistema di Cache - Stubs** (`utils/stubs_utils.py`)
Sistema di caching per evitare ricalcoli costosi:
- **`save_stubs(stub_path, object)`**: Salva un oggetto Python in formato pickle
- **`read_stubs(read_from_stub, stub_path)`**: Carica un oggetto dalla cache se esiste

### 9. **Utility Bounding Box** (`utils/bbox_utils.py`)
Utility per operazioni su bounding box:

- **`get_center_of_bbox(bbox)`**: Calcola il centro geometrico di un bbox (x1,y1,x2,y2)
- **`get_bbox_width(bbox)`**: Calcola la larghezza di un bbox
- **`measure_distance(point1, point2)`**: Calcola distanza euclidea tra due punti (usato per ball acquisition)

### 10. **TeamAssigner** (`team_assigner/team_assigner.py`)
Classe per l'assegnazione dei giocatori alle squadre in base al colore della maglia:

- **`__init__(team_1_class_name, team_2_class_name)`**: Inizializza i nomi dei colori delle squadre (default: "white shirt", "dark blue shirt")
- **`load_model()`**: Carica il modello Fashion-CLIP per il riconoscimento dei colori
- **`get_player_color(frame, bbox)`**: Classifica il colore della maglia di un giocatore usando CLIP
- **`get_player_team(frame, player_bbox, player_id)`**: Assegna un giocatore a una squadra (1 o 2) in base al colore
- **`get_player_teams_across_frames(video_frames, player_tracks, read_from_stub, stub_path)`**: Assegna le squadre a tutti i giocatori in tutti i frame

> ⚠️ **Limitazione attuale**: Il sistema riconosce attualmente solo **due colori hardcoded**: `"white shirt"` (squadra 1) e `"dark blue shirt"` (squadra 2). Per supportare altri colori, è necessario modificare i parametri `team_1_class_name` e `team_2_class_name` nel costruttore di `TeamAssigner` in `main.py`.

### 11. **BallAquisitionDetector** (`ball_acquisition/ball_aquisition_detector.py`)
Classe per il rilevamento del possesso palla da parte dei giocatori:

- **`__init__()`**: Inizializza soglie:
  - `possession_threshold` (50px): Distanza massima per possession senza alta containment
  - `min_frames` (11): Frame consecutivi richiesti per confermare il possesso
  - `containment_threshold` (0.8): Ratio di contenimento del pallone nel bbox del giocatore
  
- **`get_key_basketball_player_assignment_points(player_bbox, ball_center)`**: Genera punti chiave attorno al bbox del giocatore (corners, edges, center) per misure di distanza accurate
  
- **`calculate_ball_containment_ratio(player_bbox, ball_bbox)`**: Calcola la percentuale di pallone contenuta nel bbox del giocatore (area intersezione / area pallone)
  
- **`find_minimum_distance_to_ball(ball_center, player_bbox)`**: Trova la minima distanza dal centro pallone ai punti chiave del giocatore
  
- **`find_best_candidate_for_possession(ball_center, player_tracks_frame, ball_bbox)`**: 
  - Priorità 1: Giocatori con high containment (>80%)
  - Priorità 2: Giocatori entro soglia di distanza (<50px)
  - Restituisce player_id o -1
  
- **`detect_ball_possession(player_tracks, ball_tracks)`**: Rileva possesso per ogni frame, richiedendo 11 frame consecutivi per confermare

**Logica di rilevamento**:
1. Per ogni frame, calcola distanza e containment verso ogni giocatore
2. Prioritizza giocatori con alta containment (ball quasi dentro il bbox)
3. Se nessuno ha alta containment, seleziona il più vicino entro soglia
4. Conferma il possesso solo se persistente per min_frames consecutivi

### 12. **PassAndInterceptionDetector** (`pass_and_interception_detector/pass_and_interception_detector.py`)
Classe per il rilevamento di passaggi e intercetti:

- **`detect_passes(ball_acquisition, player_assignment)`**: Rileva passaggi riusciti tra giocatori della stessa squadra
  - Identifica cambi di possesso tra frame
  - Verifica che il possesso passi tra giocatori della stessa squadra
  - Restituisce lista con 1=passaggio Team1, 2=passaggio Team2, -1=nessuno

- **`detect_interceptions(ball_acquisition, player_assignment)`**: Rileva intercetti da parte delle squadre avversarie
  - Identifica cambi di possesso tra frame
  - Verifica che il possesso cambi tra squadre diverse
  - Restituisce lista con 1=intercetto Team1, 2=intercetto Team2, -1=nessuno

**Output**: Liste parallele a `ball_acquisition` per ogni frame

### 13. **PassInterceptionDrawer** (`drawers/pass_and_iterceptions_drawer.py`)
Classe per la visualizzazione di passaggi e intercetti:

- **`get_stats(passes, interceptions)`**: Conta totali di passaggi e intercetti per squadra fino a frame corrente
  - Restituisce tupla (team1_passes, team2_passes, team1_interceptions, team2_interceptions)

- **`draw(video_frames, passes, interceptions)`**: Disegna statistiche cumulative su tutti i frame

- **`draw_frame(frame, frame_num, passes, interceptions)`**: Disegna overlay semi-trasparente con statistiche su singolo frame
  - Rettangolo bianco semi-trasparente in basso a sinistra
  - Visualizza passaggi e intercetti cumulativi per entrambe le squadre

**Output**: Overlay tipo:
```
Team 1 - Passes: 12 Interceptions: 3
Team 2 - Passes: 15 Interceptions: 5
```

---

## 🛠️ Tecnologie Utilizzate

| Tecnologia | Versione | Utilizzo |
|------------|----------|----------|
| **Python** | 3.12 | Linguaggio principale |
| **Ultralytics YOLO** | 8.4.6 | Object detection |
| **Supervision** | 0.27.0 | ByteTrack e utility per detection |
| **OpenCV** | 4.13.0 | Elaborazione video e disegno |
| **PyTorch** | 2.9.1 | Backend per YOLO |
| **Transformers** | latest | Libreria Hugging Face per CLIP |
| **Fashion-CLIP** | latest | Modello per riconoscimento colori maglie |

---


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
pip install ultralytics supervision opencv-python transformers pillow
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
2. Esegue il tracking dei giocatori (o usa la cache da `stubs/player_tracks.stub.pkl`)
3. Esegue il tracking del pallone (o usa la cache da `stubs/ball_tracks.stub.pkl`)
4. Assegna ogni giocatore a una squadra in base al colore della maglia (o usa la cache da `stubs/player_assignement_stub.pkl`)
5. Genera `output_videos/output_video.avi` con le annotazioni (ellissi colorate per squadra, triangolo per il pallone)

### Troubleshooting

**Out of Memory**: Attivare la riduzione di risoluzione in `main.py`:
```python
video_frames = [cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4)) for frame in video_frames]
```

**Stubs Corrupted**: Eliminare manualmente:
```bash
rm stubs/*.pkl
```

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
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ Player │ │  Ball  │
│ Tracker│ │ Tracker│
│ (YOLO  │ │ (YOLO  │
│  +     │ │  +     │
│ByteTrack│ │MaxConf)│
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌────────┐ ┌────────┐
│ Player │ │  Ball  │
│ Tracks │ │ Tracks │
│ Drawer │ │ Drawer │
│(ellissi│ │(triangoli)│
│  + ID) │ │        │
└───┬────┘ └───┬────┘
    │          │
    └────┬─────┘
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
- **Riconoscimento automatico colori squadre**: Attualmente il sistema riconosce solo "white shirt" e "dark blue shirt". Si prevede di implementare un rilevamento automatico dei colori dominanti delle squadre
- **Visualizzazione ball acquisition**: Integrare il `BallAquisitionDetector` nel pipeline di disegno per visualizzare il possesso palla
- **Court Detection**: Rilevamento dei keypoint del campo (`court_keypoint_detector.pt`)
- **Analisi tattica**: Posizionamento dei giocatori rispetto al campo
- **Statistiche**: Velocità, distanze percorse, heat map
- **Interpolazione pallone avanzata**: Affinare i filtri e l'interpolazione per casi limite

---

## 📁 File Ignorati (.gitignore)

```
models/*.pt          # Modelli troppo grandi
input_videos/        # Video di input
runs/                # Output YOLO
stubs/               # Cache delle tracce
__pycache__/         # Cache Python
.venv/               # Virtual environment
```

---

## 👤 Autore

Progetto sviluppato per il corso di **Computer Vision and Cognitive Systems**
