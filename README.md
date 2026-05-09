# Face-Attendance-System-from-Live-CCTV


**Ongoing:**
------------
a. Better Face Detection Model

b. Face Landmark improvement(probably)

c. Robust Database

d. Attendance Log Logic

e. Dashboard Plot 

**Neccesary Features to Add:**
------------------------------
1. Liveliness Detection / Anti-spoofing

2. Intrusion Detection(Person detected but no face detected)
  
3. SMS Alert for Unknown + (2.)


***Important:*** Image Registration using InsightFace is optional & is used for testing.

Srry for not creating venv.....

# Face Recognition Attendance System

A real-time multi-person face recognition and attendance system built using:

- YOLO
- InsightFace
- MediaPipe
- FAISS Vector Database
- Streamlit Dashboard

Optimized for **CPU inference**.

---

# Features

- Real-time face detection
- Multi-person face recognition
- Face alignment pipeline
- FAISS-based vector similarity search
- Attendance logging system
- Streamlit dashboard
- CPU-optimized inference pipeline
- Modular recognition architecture

---

# Tech Stack

| Component | Usage |
|---|---|
| YOLO | Face / person detection |
| InsightFace | Face embeddings |
| MediaPipe | Face landmarks |
| FAISS | Vector similarity search |
| Streamlit | Dashboard/UI |
| ONNX Runtime | CPU inference |

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

---

## 2. Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Requirements

```txt
streamlit
opencv-python
numpy
faiss-cpu
pandas
plotly
ultralytics
mediapipe
onnxruntime
insightface
```

---

# Run Application

## Streamlit

```bash
streamlit run app.py
```

or

## Python

```bash
python app.py
```

(depending on your implementation)

---

# Project Structure

```text
project/
│
├── Modele/
├── app.py
├── face_system.py
├── attendance.db
├── face_db.pkl
├── face_index.faiss
├── requirements.txt
└── README.md
```

---

# Current Development Progress

## Ongoing Improvements

- Better face detection model
- Improved face landmark estimation
- More robust database system
- Improved attendance logging logic
- Enhanced dashboard analytics and plots

---

# Planned Features

- Liveness Detection / Anti-Spoofing
- Intrusion Detection
  - Person detected but no face detected
- SMS alerts for:
  - Unknown person detection
  - Intrusion detection events

---

# Notes

- This project is optimized for **CPU-only inference**.
- `onnxruntime-gpu` is intentionally not used.
- Image registration using InsightFace is optional and mainly used for testing/debugging.
- Virtual environment (`venv`) is intentionally not included in the repository.

---

# Recommended .gitignore

```gitignore
venv/
__pycache__/
*.pyc
```

---

# Future Goals

- Stable real-time tracking
- Better unknown-face handling
- Faster inference pipeline
- Web deployment
- Distributed vector database support

---

# Disclaimer

This project is currently under active development and experimental improvements are continuously being added.
