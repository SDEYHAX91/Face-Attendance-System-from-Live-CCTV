import cv2
import numpy as np
import sqlite3
import pickle
import os
from datetime import datetime
import faiss

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Disable GPU globally

from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import onnxruntime as ort


# ====================== ArcFace ONNX ======================
class ArcFaceONNX:
    def __init__(self, model_path="Modele/w600k_r50.onnx"):
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, aligned_face):
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]   # NCHW
        emb = self.session.run(None, {self.input_name: img})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-8)


# ====================== Face Pipeline ======================
class FacePipeline:
    def __init__(self):
        # YOLO ONNX - No .to('cpu') needed
        self.yolo = YOLO("Modele/best.onnx")
        
        # MediaPipe
        base_opts = mp_python.BaseOptions(model_asset_path="Modele/face_landmarker.task")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.5
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        
        # ArcFace
        self.arcface = ArcFaceONNX()
        
        self.ref_landmarks = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)

    def detect_and_align(self, frame):
        results = []
        
        # YOLO Detection - Force CPU via device parameter
        yolo_results = self.yolo(
            frame, 
            verbose=False, 
            conf=0.4, 
            imgsz=320,
            device='cpu'                    # Important
        )
        
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Expand box slightly for better landmarks
                h, w = frame.shape[:2]
                dw = int(0.12 * (x2 - x1))
                dh = int(0.18 * (y2 - y1))
                x1e = max(0, x1 - dw)
                y1e = max(0, y1 - dh)
                x2e = min(w, x2 + dw)
                y2e = min(h, y2 + dh)

                crop = frame[y1e:y2e, x1e:x2e]
                if crop.size == 0:
                    continue

                # MediaPipe Landmarks
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                lm_result = self.landmarker.detect(mp_image)

                if not lm_result.face_landmarks:
                    continue

                landmarks = lm_result.face_landmarks[0]
                ch, cw = crop.shape[:2]

                def centroid(indices):
                    pts = np.array([(landmarks[i].x * cw, landmarks[i].y * ch) for i in indices])
                    return pts.mean(axis=0)

                left_eye   = centroid([33, 133, 160, 159, 158])
                right_eye  = centroid([362, 263, 387, 386, 385])
                nose       = np.array([landmarks[1].x * cw, landmarks[1].y * ch])
                left_mouth = np.array([landmarks[61].x * cw, landmarks[61].y * ch])
                right_mouth = np.array([landmarks[291].x * cw, landmarks[291].y * ch])

                src_pts = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
                src_pts += [x1e, y1e]   # global coordinates

                # Align to ArcFace
                M, _ = cv2.estimateAffinePartial2D(src_pts, self.ref_landmarks)
                if M is None:
                    continue

                aligned = cv2.warpAffine(frame, M, (112, 112))
                embedding = self.arcface.get_embedding(aligned)

                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf[0]),
                    'embedding': embedding
                })
        
        return results


# ====================== MAIN SYSTEM ======================
class FaceAttendanceSystem:
    def __init__(self, threshold=0.35):
        self.pipeline = FacePipeline()
        self.threshold = threshold
        self.db_path = 'attendance.db'
        self.embeddings = []   # (user_id, name, embedding)
        self._init_db()
        self._load_db()
        
        # FAISS
        self.dim = 512
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self._load_faiss()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT, embedding BLOB)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS attendance (timestamp TEXT, date TEXT, user_id TEXT, name TEXT)''')
        conn.commit()
        conn.close()

    def _load_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT id, name, embedding FROM users")
        for row in cursor:
            emb = pickle.loads(row[2])
            self.embeddings.append((row[0], row[1], emb))
        conn.close()

    def _load_faiss(self):
        if self.embeddings:
            embs = np.array([e[2] for e in self.embeddings], dtype=np.float32)
            self.index.add(embs)
            self.metadata = [{"user_id": uid, "name": name} for uid, name, _ in self.embeddings]

    def register_user_from_embeddings(self, embeddings, name):
        if len(embeddings) == 0:
            return 0, None
        # Convert to numpy float32 array
        embeddings = np.asarray(embeddings, dtype=np.float32)
        # -----------------------------
        # Step 1: Normalize each embedding
        # -----------------------------
        embeddings = embeddings / np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )
        # -----------------------------
        # Step 2: Average embeddings
        # -----------------------------
        avg_emb = np.mean(embeddings, axis=0)
        # -----------------------------
        # Step 3: Normalize averaged embedding
        # -----------------------------
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        # Generate unique user ID
        user_id = f"ID-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # Save to SQLite database
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO users VALUES (?, ?, ?)",
            (
                user_id,
                name,
                pickle.dumps(avg_emb)
            )
        )
        conn.commit()
        conn.close()
        # Store in memory
        self.embeddings.append(
            (user_id, name, avg_emb)
        )
        # Add to FAISS index
        self.index.add(
            avg_emb.reshape(1, -1).astype(np.float32)
        )
        # Store metadata
        self.metadata.append({
            "user_id": user_id,
            "name": name
        })
        return len(embeddings), user_id

    def recognize(self, img):
        faces_info = self.pipeline.detect_and_align(img)
        results = []
        
        for face in faces_info:
            emb = face['embedding']
            bbox = face['bbox']
            
            if self.index.ntotal == 0:
                results.append({'bbox': bbox, 'name': 'Unknown', 'confidence': 0.0})
                continue

            q = emb.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(q, 1)
            best_score = float(scores[0][0])
            best_idx = indices[0][0]

            if best_score > self.threshold and best_idx < len(self.metadata):
                match = self.metadata[best_idx]
                results.append({
                    'bbox': bbox,
                    'name': match["name"],
                    'confidence': best_score
                })
            else:
                results.append({
                    'bbox': bbox,
                    'name': 'Unknown',
                    'confidence': best_score
                })
        return results

    def mark_attendance(self, name, user_id):
        if name == 'Unknown':
            return
        today = datetime.now().date().isoformat()
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=? AND user_id=?", 
                            (today, user_id)).fetchone()[0]
        if count == 0:
            conn.execute("INSERT INTO attendance (timestamp, date, user_id, name) VALUES (?, ?, ?, ?)",
                        (timestamp, today, user_id, name))
            conn.commit()
        conn.close()
    
    def delete_identity(self, name_to_delete):
        """Delete a person by name"""
        if not name_to_delete:
            return False

        new_embeddings = []
        new_metadata = []
        deleted = False
        deleted_user_id = None

        for item in self.embeddings:
            # Safe extraction: handle tuples of any length
            if len(item) >= 2:
                uid = item[0]
                name = item[1]
                emb = item[2] if len(item) > 2 else None
            else:
                continue

            if name == name_to_delete:
                deleted = True
                deleted_user_id = uid
                continue  # Skip this one (delete it)

            # Keep this entry
            new_embeddings.append((uid, name, emb))
            new_metadata.append({"user_id": uid, "name": name})

        if not deleted:
            return False

        # Update internal state
        self.embeddings = new_embeddings

        # Rebuild FAISS index
        self.index = faiss.IndexFlatIP(self.dim)
        if self.embeddings:
            embs = np.array([item[2] for item in self.embeddings if item[2] is not None], 
                          dtype=np.float32)
            if len(embs) > 0:
                self.index.add(embs)

        self.metadata = new_metadata

        # Delete from database
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM users WHERE name=?", (name_to_delete,))
        if deleted_user_id:
            conn.execute("DELETE FROM attendance WHERE user_id=?", (deleted_user_id,))
        conn.commit()
        conn.close()

        return True