# VisionProject

**Progetto di Computer Vision and Cognitive Systems**

Sistema di analisi avanzata di video di partite di basket utilizzando tecniche di deep learning e computer vision per il tracking, l'analisi tattica e le statistiche di gioco.

---

## 📋 Descrizione del Progetto

VisionProject è un'applicazione di computer vision che analizza video di partite di basket per:
- **Rilevare** i giocatori e il pallone presenti in ogni frame
- **Tracciare** i movimenti dei giocatori nel tempo (tracking multi-oggetto con ByteTrack)
- **Tracciare** la posizione del pallone (detection con selezione per confidence)
- **Rilevare i canestri** (hoop detection) per l'analisi dei tiri
- **Assegnare i giocatori alle squadre** in base al colore della maglia (utilizzando Fashion-CLIP)
- **Rilevare i tiri** e determinare se sono andati a segno (shooting detection)
- **Rilevare i passaggi e gli intercetti** tra i giocatori
- **Rilevare gli assist** (passaggi seguiti da tiri segnati)
- **Generare statistiche** dettagliate (tiri, assist, possesso palla)
- **Visualizzare** le tracce con annotazioni grafiche (ellissi colorate per squadra, triangoli per il pallone, banner per i tiri e assist)
- **Generare vista tattica** con posizioni dei giocatori sul campo

Il sistema utilizza modelli YOLO pre-addestrati per la detection, l'algoritmo ByteTrack per il tracking persistente degli oggetti tra i frame, e il modello Fashion-CLIP per il riconoscimento dei colori delle maglie.

---

## 🏗️ Architettura del Progetto

```
VisionProject/
├── main.py                    # Entry point dell'applicazione
├── configs/                   # Configurazioni globali
│   ├── __init__.py
│   └── configs.py
├── models/                    # Modelli YOLO pre-addestrati
│   ├── player_detector.pt     # Modello per detection giocatori
│   ├── ball_detector_model.pt # Modello per detection pallone e canestri
│   └── court_keypoint_detector.pt # Modello per keypoint campo
├── trackers/                  # Moduli di tracking
│   ├── __init__.py
│   ├── player_tracker.py      # Classe PlayerTracker
│   └── ball_tracker.py        # Classe BallTracker (include HoopTracker)
├── drawers/                   # Moduli di visualizzazione
│   ├── __init__.py
│   ├── player_tracks_drawer.py # Classe PlayerTracksDrawer
│   ├── ball_tracks_drawer.py  # Classe BallTracksDrawer
│   ├── hoop_drawer.py         # Classe HoopDrawer
│   ├── shooting_drawer.py     # Classe ShootingDrawer
│   ├── shooting_positions_drawer.py # Classe ShootingPositionsDrawer
│   ├── team_ball_control_drawer.py # Classe TeamBallControlDrawer
│   ├── pass_and_interceptions_drawer.py # Classe PassInterceptionDrawer
│   ├── court_key_points_drawer.py # Classe CourtKeypointDrawer
│   ├── tactical_view_drawer.py # Classe TacticalViewDrawer
│   ├── frame_number_drawer.py # Classe FrameNumberDrawer
│   ├── assist_drawer.py       # Classe AssistDrawer
│   └── utils.py               # Funzioni di disegno (ellissi, triangoli)
├── team_assigner/             # Modulo assegnazione squadre
│   ├── __init__.py
│   └── team_assigner.py       # Classe TeamAssigner (Fashion-CLIP)
├── ball_aquisition/           # Modulo rilevamento possesso palla
│   ├── __init__.py
│   └── ball_aquisition_detector.py # Classe BallAquisitionDetector
├── pass_and_interception_detector/ # Modulo rilevamento passaggi
│   ├── __init__.py
│   └── pass_and_interception_detector.py # Classe PassAndInterceptionDetector
├── shooting_detector/         # Modulo rilevamento tiri
│   ├── __init__.py
│   └── shooting_detector.py   # Classe ShootingDetector
├── assist_detector/           # Modulo rilevamento assist
│   ├── __init__.py
│   └── assist_detector.py     # Classe AssistDetector
├── court_keypoint_detector/   # Modulo rilevamento keypoint campo
│   ├── __init__.py
│   └── court_keypoint_detector.py # Classe CourtKeypointDetector
├── tactical_view_converter/   # Modulo conversione vista tattica
│   ├── __init__.py
│   ├── tactical_view_converter.py # Classe TacticalViewConverter
│   └── homography.py          # Utility omografia
├── utils/                     # Utility generiche
│   ├── __init__.py
│   ├── video_utils.py         # Lettura/scrittura video
│   ├── stubs_utils.py         # Gestione cache (stubs)
│   └── bbox_utils.py          # Utility per bounding box
├── input_videos/              # Video di input
├── output_videos/             # Video processati
├── images/                    # Immagini del campo (PNG, PDF)
├── stubs/                     # Cache delle tracce (pickle) e statistiche (TSV, CSV)
├── requirements.txt           # Dipendenze Python
└── .venv/                     # Virtual environment (non versionato)
```

---

## 🔧 Componenti Principali

### 1. **main.py** - Entry Point
Il file principale che orchestra l'intera pipeline:
1. Carica il video di input
2. Inizializza i tracker (giocatori, pallone e canestri)
3. Esegue il tracking (o carica dalla cache)
4. Rileva i tiri e determina se sono andati a segno
5. Disegna le annotazioni sui frame
6. Esporta le statistiche in formato TSV
7. Salva il video di output

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
Classe responsabile del rilevamento del pallone e dei canestri (hoop):

- **`__init__(model_path)`**: Inizializza il modello YOLO per la detection del pallone e dei canestri
- **`detect_frames(frames)`**: Esegue la detection su tutti i frame in batch da 20
- **`get_object_tracks(frames, read_from_stub, stub_path)`**: 
  - Se esiste una cache (stub), la carica per evitare ricalcoli
  - Seleziona la detection con confidence massima per ogni frame
  - Salva i risultati in cache per usi futuri
- **`get_hoop_tracks(frames, read_from_stub, stub_path)`**: 
  - Rileva e traccia i canestri (hoop) in tutti i frame
  - Supporta cache per evitare ricalcoli
  - Restituisce le posizioni dei canestri per ogni frame
- **`remuve_wrong_detections(ball_positions)`**: Filtra outlier spaziali rimuovendo rimbalzi irreali tra frame (distanza massima scalata per il gap tra frame)
- **`interpolate_ball_positions(ball_positions)`**: Usa `pandas` per interpolare e backfillare le posizioni mancanti del pallone

**Output Ball Tracks**: Lista di dizionari, uno per frame:
```python
{
    1: {"bbox": [x1, y1, x2, y2]},  # Una sola detection per frame
}
```

**Output Hoop Tracks**: Lista di dizionari, uno per frame:
```python
{
    1: {"bbox": [x1, y1, x2, y2], "confidence": 0.95},  # Canestro 1
    2: {"bbox": [x1, y1, x2, y2], "confidence": 0.93},  # Canestro 2 (se presente)
}
```

### 4. **ShootingDetector** (`shooting_detector/shooting_detector.py`)
Classe per il rilevamento dei tiri e la determinazione se sono andati a segno:

- **`__init__(hoop_proximity_threshold, made_shot_threshold, min_frames_between_shots, debug)`**: Inizializza il detector con soglie configurabili:
  - `hoop_proximity_threshold` (300px): Distanza massima palla-canestro per considerare un tiro
  - `made_shot_threshold` (100px): Distanza per considerare un canestro fatto
  - `min_frames_between_shots` (20): Frame minimi tra due tiri consecutivi
  - `debug` (True): Abilita logging dettagliato

- **`detect_shots(ball_tracks, hoop_tracks, player_tracks, player_assignment, ball_acquisition)`**: Rileva tutti i tiri nel video
  - Analizza la prossimità della palla al canestro
  - Verifica se la palla si sta avvicinando (traiettoria)
  - Determina se il tiro è andato a segno (palla passa attraverso e scende)
  - Identifica il tiratore cercando l'ultimo giocatore con possesso palla

- **`get_team_stats()`**: Calcola statistiche aggregate per squadra
  - Restituisce: `{team_id: {'attempts': int, 'made': int, 'missed': int}}`

- **`export_to_tsv(output_path)`**: Esporta statistiche squadra in formato TSV
- **`export_shots_to_tsv(output_path)`**: Esporta tutti i singoli tiri in formato TSV
- **`debug_shooter_detection(...)`**: Mostra dettagli debug su come viene identificato il tiratore

**Output `shots`**: Lista di oggetti `Shot` (dataclass):
```python
@dataclass
class Shot:
    frame_start: int          # Frame di inizio tiro
    frame_end: int            # Frame di fine tiro
    team_id: int              # Squadra (1 o 2)
    player_id: int            # ID giocatore tiratore
    made: bool                # True se canestro segnato
    position: Tuple[float, float]  # Posizione (x, y) del tiratore
    hoop_id: int              # ID del canestro target
```

**Logica di rilevamento tiri**:
1. Monitora la distanza palla-canestro per ogni frame
2. Rileva potenziale tiro quando palla è vicina al canestro E si sta avvicinando
3. Determina esito verificando se la palla passa vicino al centro del canestro e poi scende
4. Trova il tiratore cercando indietro nel tempo l'ultimo giocatore con possesso palla

### 5. **AssistDetector** (`assist_detector/assist_detector.py`)
Classe per il rilevamento degli assist (passaggi che portano a canestri):

- **`__init__(max_frames_pass_to_shot, debug)`**: Inizializza il detector con:
  - `max_frames_pass_to_shot` (150): Frame massimi tra passaggio e tiro (~5 secondi a 30fps)
  - `debug` (True): Abilita logging dettagliato

- **`detect_assists(passes, shots, ball_acquisition, player_assignment)`**: Rileva tutti gli assist nel video
  - Filtra i tiri andati a segno
  - Per ogni tiro segnato, cerca il passaggio precedente entro la finestra temporale
  - Verifica che passatore e tiratore siano della stessa squadra
  - Esclude auto-passaggi (stesso giocatore)

- **`get_team_stats()`**: Calcola statistiche aggregate per squadra
  - Restituisce: `{team_id: {'total_assists': int, 'assists_by_player': {player_id: count}}}`

- **`export_to_csv(output_path, shots)`**: Esporta statistiche assist in formato CSV

**Output `assists`**: Lista di oggetti `Assist` (dataclass):
```python
@dataclass
class Assist:
    frame: int                # Frame del passaggio
    passer_id: int            # ID del passatore
    scorer_id: int            # ID del tiratore che ha segnato
    team_id: int              # Squadra
    shot_frame: int           # Frame del tiro
    pass_to_shot_frames: int  # Frame tra passaggio e tiro
```

### 6. **HoopDrawer** (`drawers/hoop_drawer.py`)
Classe per la visualizzazione dei canestri rilevati:

- **`__init__(color, thickness)`**: Inizializza con colore (default arancione) e spessore linea
- **`draw(frames, hoop_tracks)`**: Disegna i canestri su tutti i frame
  - Rettangolo colorato attorno al canestro
  - Etichetta "Hoop 1", "Hoop 2", etc.
  - Punto verde al centro del canestro

**Output**: Frame con annotazioni canestri

### 7. **ShootingDrawer** (`drawers/shooting_drawer.py`)
Classe per la visualizzazione degli eventi di tiro:

- **`__init__(made_color, missed_color, display_frames)`**: Inizializza con:
  - `made_color` (verde): Colore per canestri segnati
  - `missed_color` (rosso): Colore per canestri sbagliati
  - `display_frames` (60): Durata visualizzazione evento

- **`draw(frames, shots)`**: Disegna gli eventi di tiro su tutti i frame
  - Banner colorato in alto (verde=segnato, rosso=sbagliato)
  - Testo con squadra e giocatore
  - Cerchio pulsante nella posizione del tiratore
  - Freccia indicante direzione del tiro
  - Fade out graduale dopo N frame

**Output**: Frame con overlay eventi tiro

### 8. **ShootingPositionsDrawer** (`drawers/shooting_positions_drawer.py`)
Classe per la generazione di PDF con le posizioni di tiro sul campo:

- **`__init__(court_image_path)`**: Inizializza con il percorso all'immagine del campo
- **`draw_shooting_positions(shots, tactical_player_positions, output_path)`**: Genera PDF con:
  - Visualizzazione del campo tattico
  - Cerchi colorati per ogni tiro (verde=segnato, rosso=sbagliato)
  - Legenda con statistiche per squadra

**Output**: File PDF con shot chart

### 9. **AssistDrawer** (`drawers/assist_drawer.py`)
Classe per la visualizzazione degli assist:

- **`draw(frames, assists)`**: Disegna gli eventi di assist su tutti i frame
  - Banner informativo con passatore e tiratore
  - Visualizzazione animata dell'assist

**Output**: Frame con overlay assist

### 10. **PlayerTracksDrawer** (`drawers/player_tracks_drawer.py`)
Classe per la visualizzazione delle tracce dei giocatori:

- **`draw(video_frames, tracks, player_assignments)`**: Per ogni frame, disegna un'ellisse colorata (in base alla squadra) sotto ogni giocatore con il suo ID di tracking

### 11. **BallTracksDrawer** (`drawers/ball_tracks_drawer.py`)
Classe per la visualizzazione della posizione del pallone:

- **`draw(video_frames, tracks)`**: Per ogni frame, disegna un triangolo verde sopra il pallone

### 12. **TeamBallControlDrawer** (`drawers/team_ball_control_drawer.py`)
Classe per il calcolo e visualizzazione delle statistiche di possesso palla per squadra:

- **`get_team_ball_control(player_assignment, ball_aquisition)`**: Calcola quale squadra ha il controllo del pallone per ogni frame, restituendo array (1=Team1, 2=Team2, -1=nessuno)
- **`draw(video_frames, player_assignment, ball_aquisition)`**: Disegna overlay semi-trasparente con percentuali di possesso palla per entrambe le squadre
- **`draw_frame(frame, frame_num, team_ball_control)`**: Disegna statistiche su singolo frame con rettangolo semi-trasparente e testo percentuale

**Output**: Overlay bottom-right con statistiche real-time tipo:
```
Team 1 Ball Control: 45.23%
Team 2 Ball Control: 54.77%
```

### 13. **Funzioni di Disegno** (`drawers/utils.py`)
- **`draw_ellypse(frame, bbox, color, track_id)`**: 
  - Disegna un'ellisse ai piedi del giocatore (posizione y2 del bounding box)
  - Aggiunge un rettangolo con l'ID del track
  - L'ellisse ha forma proporzionale alla larghezza del bounding box
- **`draw_triangle(frame, bbox, color)`**:
  - Disegna un triangolo sopra il pallone (posizione y1 del bounding box)
  - Il triangolo punta verso il basso per indicare la posizione

### 14. **Utility Video** (`utils/video_utils.py`)
- **`read_video(video_path)`**: Legge un video e restituisce una lista di frame (array numpy)
- **`save_video(frames, output_path)`**: Salva i frame in un file AVI (codec XVID, 24 fps)

### 15. **Sistema di Cache - Stubs** (`utils/stubs_utils.py`)
Sistema di caching per evitare ricalcoli costosi:
- **`save_stubs(stub_path, object)`**: Salva un oggetto Python in formato pickle
- **`read_stubs(read_from_stub, stub_path)`**: Carica un oggetto dalla cache se esiste

### 16. **Utility Bounding Box** (`utils/bbox_utils.py`)
Utility per operazioni su bounding box:

- **`get_center_of_bbox(bbox)`**: Calcola il centro geometrico di un bbox (x1,y1,x2,y2)
- **`get_bbox_width(bbox)`**: Calcola la larghezza di un bbox
- **`measure_distance(point1, point2)`**: Calcola distanza euclidea tra due punti (usato per ball acquisition e shooting detection)

### 17. **TeamAssigner** (`team_assigner/team_assigner.py`)
Classe per l'assegnazione dei giocatori alle squadre in base al colore della maglia:

- **`__init__(team_1_class_name, team_2_class_name)`**: Inizializza i nomi dei colori delle squadre (default: "white shirt", "black shirt")
- **`load_model()`**: Carica il modello Fashion-CLIP per il riconoscimento dei colori
- **`get_player_color(frame, bbox)`**: Classifica il colore della maglia di un giocatore usando CLIP
- **`get_player_team(frame, player_bbox, player_id)`**: Assegna un giocatore a una squadra (1 o 2) in base al colore
- **`get_player_teams_across_frames(video_frames, player_tracks, read_from_stub, stub_path)`**: Assegna le squadre a tutti i giocatori in tutti i frame utilizzando un **sistema di voting multi-frame** per stabilizzare le assegnazioni
- **`_print_assignment_stats(player_assignment)`**: Stampa statistiche di debug sulla distribuzione Team 1 vs Team 2

**Sistema di Voting Multi-Frame**:
Il TeamAssigner utilizza un sistema di voting per stabilizzare le assegnazioni:
1. **Prima passata**: Per ogni giocatore, raccoglie voti su tutti i frame in cui appare
2. **Reset cache periodico**: La cache viene resettata ogni 30 frame per permettere ri-valutazione
3. **Assegnazione finale**: Il team viene assegnato in base alla maggioranza dei voti
4. **Consistenza**: Una volta determinato, il team rimane costante per tutto il video

**Debug Output**:
```
Voti per giocatore:
  Player 1: Team1=118 (64%), Team2=66 (36%)
  Player 2: Team1=95 (61%), Team2=60 (39%)
  ...

Statistiche assegnazione:
  Team 1: 2956 (50.2%)
  Team 2: 2930 (49.8%)
```

> ⚠️ **Configurazione colori**: Per modificare i colori delle squadre, cambiare i parametri `team_1_class_name` e `team_2_class_name` nel costruttore di `TeamAssigner`. Default: `"white shirt"` (squadra 1) e `"black shirt"` (squadra 2).

### 18. **BallAquisitionDetector** (`ball_aquisition/ball_aquisition_detector.py`)
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

### 19. **PassAndInterceptionDetector** (`pass_and_interception_detector/pass_and_interception_detector.py`)
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

### 20. **PassInterceptionDrawer** (`drawers/pass_and_interceptions_drawer.py`)
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

### 21. **FrameNumberDrawer** (`drawers/frame_number_drawer.py`)
Classe per la visualizzazione del numero del frame corrente:

- **`draw(video_frames)`**: Disegna il numero del frame in alto a sinistra
  - Posizione fissa (20px da sinistra, 50px da sopra)
  - Testo bianco su sfondo nero semi-trasparente
  - Utile per debugging e sincronizzazione

**Output**: Numero del frame progressivo in ogni frame

### 22. **CourtKeypointDetector** (`court_keypoint_detector/court_keypoint_detector.py`)
Classe per il rilevamento dei keypoint del campo:

- **`__init__(model_path)`**: Inizializza il modello YOLO per la detection dei keypoint del campo
- **`get_court_keypoints(frames, read_from_stub, stub_path)`**: 
  - Rileva i keypoint del campo (linee, angoli, etc.) su tutti i frame in batch da 20
  - Se esiste una cache (stub), la carica per evitare ricalcoli
  - Utilizza confidence threshold di 0.5 per filtrare detection deboli
  - Salva i risultati in cache per usi futuri

**Output**: Lista di keypoint per frame, formato PyTorch tensor con coordinate (x, y)

### 23. **CourtKeypointDrawer** (`drawers/court_key_points_drawer.py`)
Classe per la visualizzazione dei keypoint del campo:

- **`__init__()`**: Inizializza con colore rosso (`#ff2c2c`) per i keypoint
- **`draw(frames, court_keypoints)`**: Disegna keypoint e relative label su tutti i frame
  - Utilizza `VertexAnnotator` per disegnare i punti (raggio 8px)
  - Utilizza `VertexLabelAnnotator` per le etichette numeriche
  - Converte tensor PyTorch in numpy array per compatibilità

**Output**: Frame con keypoint rossi numerati per identificare i punti del campo

### 24. **TacticalViewConverter** (`tactical_view_converter/tactical_view_converter.py`)
Classe per la conversione delle posizioni dal sistema di coordinate video a quello tattico del campo:

- **`__init__(court_image_path)`**: Inizializza il converter con:
  - Dimensioni campo tattico: 300x161 pixel (proporzioni 28m x 15m reali)
  - Keypoint di riferimento del campo per l'omografia
  - Percorso all'immagine del campo tattico

- **`validate_keypoints(keypoints_list)`**: Valida i keypoint rilevati dal modello di court detection
  - Confronta le distanze proporzionali tra keypoint con le distanze attese
  - Scarta keypoint non validi che violano la geometria del campo
  - Mantiene la robustezza anche con detection incomplete

- **`transform_players_to_tactical_view(court_keypoints_per_frame, player_tracks)`**: Trasforma le posizioni dei giocatori nel sistema di coordinate tattico
  - Calcola la matrice di omografia per ogni frame usando i keypoint del campo validati
  - Trasforma le posizioni foot (piedi) dei giocatori dal video al campo tattico
  - Restituisce lista di dizionari con posizioni tattiche per ogni frame

**Output**: Lista di frame con posizioni tattiche (x, y) per ogni giocatore

### 25. **TacticalViewDrawer** (`drawers/tactical_view_drawer.py`)
Classe per la visualizzazione della vista tattica del campo con i giocatori:

- **`__init__(team_1_color, team_2_color)`**: Inizializza i colori per le squadre
  - Team 1: [255, 245, 238] (bianco/azzurrino)
  - Team 2: [128, 0, 0] (rosso scuro)

- **`draw(video_frames, court_image_path, width, height, tactical_court_keypoints, tactical_player_positions, player_assignment, ball_acquisition)`**: Disegna la vista tattica completa
  - Carica e ridimensiona l'immagine del campo (300x161 pixel)
  - Applica trasparenza (alpha=0.6) all'overlay del campo
  - Posiziona il campo in alto a sinistra (x=20, y=40)
  - Disegna i giocatori come cerchi colorati in base alla squadra
  - Disegna il possesso palla con marcatore speciale
  - Restituisce frame annotati

- **`draw_frame(frame, court_image, width, height, frame_idx, tactical_court_keypoints, tactical_player_positions, player_assignment, ball_acquisition)`**: Disegna la vista tattica su singolo frame
  - Applica la sovrimpressione del campo
  - Disegna i giocatori (cerchi pieni per Team 1, cerchi per Team 2)
  - Evidenzia il giocatore con il possesso palla

**Output**: Frame con overlay tattico che mostra la disposizione dei giocatori sul campo in tempo reale

---

## 📊 Output Files

### TSV Files (Statistiche Tiri)
Il sistema genera file TSV con le statistiche di tiro:

**stubs/team_shooting_stats.tsv**
```
team_id	attempts	made	missed	percentage
1	5	2	3	40.0%
2	4	3	1	75.0%
```

**stubs/all_shots.tsv**
```
shot_id	frame_start	frame_end	team_id	player_id	made	position_x	position_y	hoop_id
1	120	140	1	5	True	450.5	320.0	1
2	280	300	2	8	False	520.3	310.5	2
```

### CSV Files (Statistiche per Team)
Il sistema genera due file CSV separati, uno per ogni squadra:

**stubs/stats_1.csv** (Team 1)
```
==================================================
STATISTICHE TEAM 1
==================================================
Statistica,Valore
Tiri totali,3
Tiri segnati,2
Tiri sbagliati,1
FG%,66.7%
Assist totali,1
Passaggi totali,10
Intercetti totali,1

==================================================
STATISTICHE PER GIOCATORE
==================================================

PLAYER 5
,Statistica,Valore
,Tiri totali,2
,Tiri segnati,1
,Tiri sbagliati,1
,FG%,50.0%
,Assist,1
,Passaggi,5
,Intercetti,0
```

**stubs/stats_2.csv** (Team 2) - Stesso formato con le statistiche del Team 2

### PDF Files (Shot Chart)
**images/shooting_positions.pdf** - Mappa delle posizioni di tiro sul campo tattico

---

## 🛠️ Tecnologie Utilizzate

| Tecnologia | Versione | Utilizzo |
|------------|----------|----------|
| **Python** | 3.12 | Linguaggio principale |
| **Ultralytics YOLO** | 8.3.67 | Object detection e court keypoint detection |
| **Supervision** | 0.25.1 | ByteTrack e utility per annotazioni |
| **OpenCV** | 4.9.0 | Elaborazione video e disegno |
| **PyTorch** | 2.2.0 | Backend per YOLO |
| **Pandas** | 2.0.3 | Interpolazione e manipolazione dati |
| **NumPy** | 1.26.4 | Operazioni numeriche |
| **Transformers** | 4.46.3 | Libreria Hugging Face per CLIP |
| **Fashion-CLIP** | latest | Modello per riconoscimento colori maglie |

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
pip install -r requirements.txt
```

Oppure manualmente:
```bash
pip install ultralytics supervision opencv-python transformers pillow pandas numpy torch roboflow
```

### 4. Posizionare i modelli
Inserire i file `.pt` dei modelli nella cartella `models/`:
- `player_detector.pt` - Modello per detection giocatori
- `ball_detector_model.pt` - Modello per detection pallone e canestri
- `court_keypoint_detector.pt` - Modello per keypoint campo

---

## ▶️ Esecuzione

### Esecuzione base
```bash
python main.py
```

Processa il video di default `input_videos/video_2.mp4` e genera `output_videos/output_video.avi`

### Esecuzione con parametri
```bash
python main.py --input_video input_videos/video_3.mp4 --output_video output_videos/video3_analyzed.avi --stub_path stubs
```

**Parametri disponibili**:
- `--input_video`: Percorso al video di input (default: `input_videos/video_2.mp4`)
- `--output_video`: Percorso al video di output (default: `output_videos/output_video.avi`)
- `--stub_path`: Percorso alla cartella con i cache (stubs) (default: `stubs`)

### Pipeline di elaborazione
Il programma elabora il video nel seguente ordine:
1. **Lettura video**: Carica tutti i frame dall'input video
2. **Player Tracking**: Rileva e traccia i giocatori (usa cache se disponibile)
3. **Ball Tracking**: Rileva e traccia il pallone (filtra per confidence + interpolazione)
4. **Hoop Tracking**: Rileva e traccia i canestri (usa lo stesso modello del ball tracker)
5. **Court Keypoint Detection**: Rileva i keypoint del campo tattico
6. **Team Assignment**: Assegna i giocatori alle squadre (Fashion-CLIP based)
7. **Ball Acquisition**: Rileva il possesso palla
8. **Pass/Interception Detection**: Identifica passaggi e intercetti
9. **Shooting Detection**: Rileva i tiri e determina se sono andati a segno
10. **Assist Detection**: Identifica gli assist (passaggi che portano a canestri)
11. **Tactical View Conversion**: Trasforma posizioni nel sistema tattico
12. **Drawing**: Disegna tutti gli overlay (tracce, canestri, tiri, assist, tattica)
13. **Export Statistics**: Esporta statistiche in formato TSV/CSV
14. **Shot Chart PDF**: Genera PDF con mappa delle posizioni di tiro
15. **Video Output**: Salva il video processato in AVI

### Troubleshooting

**Out of Memory**: Attivare la riduzione di risoluzione in `main.py`:
```python
video_frames = [cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4)) for frame in video_frames]
```

**Stubs Corrupted**: Eliminare manualmente:
```bash
rm stubs/*.pkl
```

**Video Processing Too Slow**: Abilitare CUDA se disponibile (PyTorch/YOLO useranno GPU automaticamente)

**Memory Error During Video Save**: Verificare spazio disponibile e ridurre risoluzione del video di output

**Shooting Detection non accurato**: Modificare le soglie in `main.py`:
```python
shooting_detector = ShootingDetector(
    hoop_proximity_threshold=300,  # Aumentare se tiri non rilevati
    made_shot_threshold=100,       # Diminuire per essere più precisi
    min_frames_between_shots=20    # Aumentare per evitare duplicati
)
```

---

## 📊 Pipeline di Elaborazione

```
┌──────────────────┐
│   Video Input    │
│     (MP4)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  read_video()    │  Estrae tutti i frame
└────────┬─────────┘
         │
    ┌────┴──────────────────────────────┐
    │                                   │
    ▼                                   ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Player  │  │   Ball   │  │  Hoop    │  │  Court   │
│ Tracker │  │ Tracker  │  │ Tracker  │  │Keypoint  │
│ (YOLO   │  │  (YOLO   │  │  (YOLO   │  │Detector  │
│ +ByteT) │  │+MaxConf) │  │  same    │  │  (YOLO)  │
│         │  │          │  │  model)  │  │          │
└────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │            │             │             │
     ▼            ▼             ▼             ▼
 ┌────────┐  ┌────────┐  ┌──────────┐  ┌──────────────┐
 │ Player │  │  Ball  │  │  Hoop    │  │ Keypoint Val │
 │ Tracks │  │ Tracks │  │  Tracks  │  │  idation &   │
 │        │  │(+filter│  │          │  │ Homography   │
 │        │  │ +interp│  │          │  │              │
 └────┬───┘  └───┬────┘  └────┬─────┘  └────┬─────────┘
      │          │            │             │
      └──────────┴────────────┴──────┬──────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────┐
        │ Team Assignment (CLIP)                 │
        │ Ball Acquisition Detector              │
        │ Pass/Interception Detect               │
        │ Shooting Detector (uses hoop tracks)   │
        └──────────┬─────────────────────────────┘
                   │
        ┌──────────┴──────────┬──────────┬───────────────┐
        │                     │          │               │
        ▼                     ▼          ▼               ▼
   ┌──────────┐       ┌──────────────┐ ┌────────────┐ ┌──────────┐
   │ Player   │       │    Tactical  │ │  Ball Ctrl │ │ Shooting │
   │ Tracks   │       │   View Conv. │ │  & Passes  │ │ Drawer   │
   │ Drawer   │       │              │ │   Drawer   │ │ (banner) │
   │(ellissi) │       │              │ │   (overlay)│ │          │
   └────┬─────┘       └────┬─────────┘ └────┬───────┘ └────┬─────┘
        │                  │                │              │
        │       ┌──────────┴──────────┐     │              │
        │       │                     │     │              │
        ▼       ▼                     ▼     ▼              ▼
    ┌──────────────┐       ┌──────────────────┐    ┌──────────────┐
    │  Ball Tracks │       │   Tactical View  │    │  Hoop Drawer │
    │  Drawer      │       │    Drawer        │    │  (bbox+label)│
    │  (triangoli) │       │  (players+campo) │    │              │
    └──────┬───────┘       └────────┬─────────┘    └──────┬───────┘
           │                        │                     │
           └────────────┬───────────┴─────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────┐
        │    Composite Frames         │
        │  (tutti gli overlay)        │
        └────────┬────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│  save_video()   │  │  export_to_tsv  │
│ (Output AVI)    │  │ (Shooting Stats)│
└─────────────────┘  └─────────────────┘
```

---

## 🔮 Sviluppi Futuri

Il progetto include le seguenti funzionalità:
- ✅ **Court Detection**: Rilevamento dei keypoint del campo (court_keypoint_detector.pt)
- ✅ **Vista Tattica**: Conversione delle posizioni dal video al campo tattico usando omografia
- ✅ **Visualizzazione Tattica**: Overlay con disposizione dei giocatori sul campo
- ✅ **Possesso Palla**: Rilevamento e visualizzazione del possesso palla
- ✅ **Passaggi e Intercetti**: Rilevamento di passaggi e intercetti tra giocatori
- ✅ **Statistiche Ball Control**: Percentuale di controllo palla per squadra
- ✅ **Hoop Tracking**: Rilevamento e tracking dei canestri
- ✅ **Shooting Detection**: Rilevamento dei tiri e determinazione esito (fatto/sbagliato)
- ✅ **Shooting Statistics**: Export statistiche tiri in formato TSV
- ✅ **Assist Detection**: Rilevamento degli assist (passaggi che portano a canestri)
- ✅ **Shot Chart PDF**: Generazione mappa delle posizioni di tiro sul campo
- ✅ **Export CSV/TSV**: Statistiche complete per giocatore e squadra

Sviluppi futuri previsti:
- **Riconoscimento automatico colori squadre**: Implementare un rilevamento automatico dei colori dominanti delle squadre
- **Statistiche avanzate**: Velocità, distanze percorse, heat map, time-to-goal
- **Analisi tattica avanzata**: Formazioni, pressing, spazi lasciati
- **Interpolazione pallone avanzata**: Affinare i filtri e l'interpolazione per casi limite (rimbalzi, occlusion)
- **Supporto multi-view**: Analisi da telecamere multiple
- **Tiri da 3 punti**: Classificazione tiri da 2 e da 3 punti in base alla posizione

---

## 📁 File Ignorati (.gitignore)

```
models/*.pt          # Modelli troppo grandi
input_videos/        # Video di input
runs/                # Output YOLO
stubs/               # Cache delle tracce
__pycache__/         # Cache Python
.venv/               # Virtual environment
venv/                # Virtual environment alternativo
venv311/             # Virtual environment Python 3.11
```

---

## 👤 Autore

Progetto sviluppato per il corso di **Computer Vision and Cognitive Systems**