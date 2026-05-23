EYESONIC AI — Smart Vision Assistive System

> **An AI-powered assistive device that detects objects, recognizes faces, identifies fire, and speaks alerts in real time — designed to help visually impaired individuals navigate safely.**

🏆 **2nd Place — CodeSphere'36 Hackathon** | AIoT Track

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [How Each Feature Works](#how-each-feature-works)
- [API Reference](#api-reference)
- [Detection Modes](#detection-modes)
- [AI Models](#ai-models)
- [Adding a New Language](#adding-a-new-language)
- [Training a New Face](#training-a-new-face)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)

---

## Overview

EYESONIC AI is a browser-based computer vision system that streams your device's camera through an AI pipeline running on a local Flask server. Every frame is analysed for:

- **80+ object classes** using YOLOv8
- **Known faces** using dlib face encodings
- **Real fire and flame** using a 6-gate precision HSV + flicker engine
- **Distance to every detected object** using focal-length estimation

All results are spoken aloud using the browser's Web Speech API and displayed as colour-coded overlay boxes directly on the live camera feed — with no cloud dependency.

---

## Key Features

| Feature | Details |
|---------|---------|
| 🔍 Real-time object detection | YOLOv8 Nano / Small / Large — swappable from the UI |
| 👤 Face recognition | Registers and identifies known people by name |
| 🔥 Precision fire detection | 6-gate HSV + flicker algorithm; rejects skin/clothing false positives |
| 📏 Distance estimation | Focal-length formula; warns when objects enter 1m danger zone |
| 🔊 Voice alerts | Browser Web Speech API — priority: danger face → danger object → fire → nearest object |
| 🌐 Multi-language | English and Hindi (हिन्दी) — labels, directions, distances, and voice |
| 🎭 Context modes | Home, Outdoor, Crowded, Travel — filters which objects are highlighted |
| ➕ In-browser face training | Capture photo from webcam with 3-second countdown → train → instant recognition |
| 📺 HUD overlay | Marching-ant borders, confidence arcs, direction strips, animated fire boxes |
| 🚫 No internet required | Fully local — Flask + browser, no external API calls |

---

## System Architecture

```
Browser (index.html)
│
│  ┌─ Camera feed → <canvas> (video layer)
│  ├─ Overlay <canvas> (detection boxes, drawn every frame at 60fps)
│  └─ Every 3rd frame → POST /detect (640×480 JPEG, base64)
│
Flask Server (app.py)
│
│  ┌─ Face recognition  (face_recognition + dlib HOG)
│  ├─ Object detection  (YOLOv8 via Ultralytics)
│  ├─ Fire detection    (OpenCV HSV pipeline)
│  └─ JSON response → browser
│
Browser receives JSON
│  ├─ Draws boxes on overlay canvas (mapBox: server 640×480 → display px)
│  ├─ Updates sidebar panels (Objects / Faces tabs)
│  ├─ Triggers Web Speech API voice alert
│  └─ Shows fire alarm badge + strobe overlay if fire detected
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | Python 3.10+ / Flask | REST API server |
| Object AI | Ultralytics YOLOv8 | 80-class COCO object detection |
| Face AI | `face_recognition` + `dlib` | HOG face detection + 128-d encoding |
| Fire CV | OpenCV (HSV, morphology, contours) | Precision flame/fire detection |
| Image processing | OpenCV, NumPy, Pillow | Frame decoding, colour space conversion |
| Frontend | Vanilla HTML/CSS/JS | No framework — single file |
| Fonts | Syne (display) + JetBrains Mono (data) | Google Fonts |
| Voice | Web Speech API (`SpeechSynthesisUtterance`) | Text-to-speech in EN / HI |

---

## Project Structure

```
EYESONIC_AI/
│
├── app.py                  # Flask server — all AI pipelines live here
│
├── templates/
│   └── index.html          # Full frontend — camera, UI, canvas drawing
│
├── known_faces/            # Drop face images here (auto-loaded on start)
│
├── yolov8n.pt              # YOLOv8 Nano  (fastest)
├── yolov8s.pt              # YOLOv8 Small (balanced) ← default
├── yolov8l.pt              # YOLOv8 Large (most accurate)
│
├── fire.pt                 # Optional: fire-specific YOLO model
├── sounds/                 # Optional: custom audio files per object
│
├── debug_dlib.py           # Diagnose dlib / face_recognition install issues
├── fix_faces.py            # Auto-fix images in known_faces/
├── test_script.py          # Quick face detection test on a single image
├── SOUNDS_CONFIG.md        # Guide for adding custom audio per object
└── README.md
```

---

## Installation

### Prerequisites

- Python **3.10.x** (strict — dlib on Windows requires exactly 3.10)
- pip
- A webcam

### Step 1 — Clone the project

```bash
git clone https://github.com/Kushneet/EYESONIC-AI.git
cd EYESONIC-AI
```

### Step 2 — Install dependencies

```bash
pip install flask
pip install opencv-python
pip install "numpy<2.0"
pip install ultralytics
pip install Pillow
pip install face-recognition
pip install cmake dlib
```

> **Windows note:** If `dlib` fails, use a pre-built wheel:
> ```bash
> pip install dlib==19.24.1
> ```
> Run `python debug_dlib.py` to verify your installation.

### Step 3 — (Optional) Add known faces

Drop clear, front-facing JPEG/PNG photos into `known_faces/`. Name each file with the person's name:

```
known_faces/
  name1.jpg
  name2.jpeg
```

Run `python fix_faces.py` if images are not loading correctly.

---

## Running the App

```bash
python app.py
```

Open your browser at `http://localhost:5000` and click **START SYSTEM**.

---

## How Each Feature Works

### 1. Object Detection

YOLOv8 runs on every 3rd camera frame. The frame is downsampled to 640×480 before sending to the model.

| Model | Confidence threshold | Use case |
|-------|---------------------|---------|
| YOLOv8 Nano | 0.40 | Fast devices, less accurate |
| YOLOv8 Small | 0.35 | Balanced (default) |
| YOLOv8 Large | 0.25 | Most accurate, slower |

### 2. Face Recognition

Uses `dlib`'s HOG face detector + a 128-dimensional face encoding per person. Matching tolerance is `0.55` (tighter than default — fewer false positives). A **danger zone** triggers when a known person is estimated within 200cm.

**In-browser training:** The ➕ Train tab opens the front camera, runs a 3-second countdown, captures a still, and sends it to `/train_face`. The server encodes with `num_jitters=20` for maximum quality.

### 3. Fire / Flame Detection

A 6-gate precision pipeline — a region must pass **all 6 gates**:

- **Gate 1** — Skin subtraction
- **Gate 2** — Orange surround HSV match
- **Gate 3** — Bright core ≥ 3% of bounding box
- **Gate 4** — Compactness filter (`perimeter² / area < 120`)
- **Gate 5** — Box fraction < 35% of frame
- **Gate 6** — Skin overlap < 60%

Plus local flicker scoring across a 6-frame buffer contributing 20% of final confidence.

### 4. Distance Estimation

```
distance_cm = (real_object_height_cm × focal_length_px) / box_height_px
```

Objects within **100cm** trigger the danger flag. Face distance uses average face width (20cm) for reliability.

### 5. Multi-Language Support

| Code | Language | Voice tag |
|------|----------|-----------|
| `en` | English | `en-US` |
| `hi` | हिन्दी (Hindi) | `hi-IN` |

Language is stateless — switching in the UI takes effect on the very next frame with no server restart.

### 6. Voice Alert Priority

1. Known person in danger zone (within 200cm)
2. Any known person detected
3. Object in danger zone (within 100cm)
4. Fire detected — 5-second cooldown
5. Nearest other object

Per-label cooldown of 4 seconds prevents repeat alerts.

---

## API Reference

### `POST /detect`

```json
// Request
{
  "image":       "data:image/jpeg;base64,/9j/...",
  "mode":        "home",
  "model":       "yolov8s",
  "lang":        "en",
  "sensitivity": "medium"
}

// Response
{
  "visual_detections": [...],
  "face_detections":   [...],
  "fire_detections":   [...],
  "has_fire":          false,
  "lang":              "en"
}
```

### `POST /train_face`

```json
// Request
{ "name": "Name", "image": "data:image/jpeg;base64,..." }

// Response
{ "success": true, "message": "✓ Face trained for: Name" }
```

### `GET /list_known_faces` — Returns all registered face names
### `GET /languages` — Returns available language options  
### `GET /status` — Server health, loaded models, supported languages

---

## Detection Modes

| Mode | Objects highlighted |
|------|-------------------|
| 🏠 Home | Chair, bed, TV, bottle, cup, couch, remote, person, dog, cat |
| 🌳 Outdoor | Car, bus, person, bicycle, traffic light, stop sign, motorcycle |
| 👥 Crowded | Person, backpack, handbag, car, bicycle, cell phone |
| ✈️ Travel | Suitcase, backpack, handbag, bus, train, car, airplane, person |

All detected objects still trigger voice alerts regardless of mode.

---

## AI Models

| File | Size | Speed | Accuracy |
|------|------|-------|---------|
| `yolov8n.pt` | 6 MB | ⚡⚡⚡ Fastest | ★★☆☆ |
| `yolov8s.pt` | 22 MB | ⚡⚡ Fast | ★★★☆ (default) |
| `yolov8l.pt` | 87 MB | ⚡ Slower | ★★★★ |
| `fire.pt` | varies | ⚡⚡ Fast | ★★★★ (optional) |

---

## Adding a New Language

**Step 1 — Add to `TRANSLATIONS` in `app.py`:**
```python
TRANSLATIONS["ta"] = {
    "on your left": "உங்கள் இடதுபுறம்",
    "fire": "தீ",
    # ... all labels
}
```

**Step 2 — Add to language modal in `index.html`:**
```html
<div class="lopt" id="lopt-ta" onclick="setLang('ta')">
  <div class="lflag">🇮🇳</div>
  <div><div class="lname">தமிழ்</div></div>
</div>
```

**Step 3 — Add BCP-47 tag in `setLang()`:**
```javascript
u.lang = currentLang === 'ta' ? 'ta-IN' : 'en-US';
```

---

## Training a New Face

### Method 1 — In-browser
1. Click **START SYSTEM** → go to **➕ Train** tab
2. Enter name → **Open Camera** → **Snap (3s)** → **Train This Face**

### Method 2 — Drop a photo
1. Place JPEG in `known_faces/` named `firstname.jpg`
2. Restart the server

**Tips:** Clear, well-lit, front-facing photo. One face per image. No sunglasses.

---

## Troubleshooting

**Face not detected:**
```bash
python debug_dlib.py
pip uninstall dlib face-recognition -y
pip install dlib==19.24.1 face-recognition
```

**Images in `known_faces/` not loading:**
```bash
python fix_faces.py
```

**`dlib` import error on NumPy 2.x:**
```bash
pip install "numpy<2.0"
```

---

## Known Limitations

- Distance estimation is approximate — degrades beyond 5m and at extreme angles
- Face recognition degrades with masks, heavy makeup, or side profiles
- Fire detection in very dark environments — use `fire.pt` for better night performance
- Hindi voice (`hi-IN`) may not be available on all systems
- YOLOv8 Large may drop below 5 FPS on CPU-only machines

---

## Credits

Built by **Ayush Garg, Kushneet Kaur, Sachin Gautam, Saurav Anand, Shruti**

🏆 **2nd Place — CodeSphere'36 Hackathon** | AIoT Track

- Object detection: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Face recognition: [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
- Computer vision: [OpenCV](https://opencv.org)
- Web server: [Flask](https://flask.palletsprojects.com)
