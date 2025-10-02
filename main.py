import cv2
import os
import json
import numpy as np
from datetime import datetime
from ads import recommend_ad
from pathlib import Path

DB_FILE = "database.json"

# Load or create database
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        database = json.load(f)
else:
    database = {"customers": []}

# Save DB
def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(database, f, indent=4)

# Compute a simple grayscale embedding for a face ROI
def compute_embedding(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    normalized = resized.astype(np.float32) / 255.0
    return normalized.flatten()

# L2 distance between embeddings
def embedding_distance(a, b):
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)) / a.size)

# Find customer by embedding
def find_customer_id(embedding, tolerance=0.12):
    known_embeddings = [np.array(cust["embedding"], dtype=np.float32) for cust in database["customers"] if "embedding" in cust]
    if not known_embeddings:
        return None
    distances = [embedding_distance(embedding, emb) for emb in known_embeddings]
    best_idx = int(np.argmin(distances))
    if distances[best_idx] <= tolerance:
        return database["customers"][best_idx]["id"]
    return None

# Log visit in DB (optionally include age/gender when available)
def log_visit(customer_id, shop="general", age_bucket=None, gender=None):
    for cust in database["customers"]:
        if cust["id"] == customer_id:
            if "visits" not in cust:
                cust["visits"] = []
            visit = {
                "shop": shop,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if age_bucket is not None:
                visit["age_bucket"] = age_bucket
            if gender is not None:
                visit["gender"] = gender
            cust["visits"].append(visit)
            save_db()
            return

# Initialize webcam
video_capture = cv2.VideoCapture(0)
next_id = len(database["customers"]) + 1

print("[INFO] Starting camera... Press 'q' to quit.")

# Prefer user's provided age/gender model folder if available
ALT_MODELS_DIR = Path("Age+Gender identification") / "Gender-and-Age-Detection"
MODELS_DIR = ALT_MODELS_DIR if ALT_MODELS_DIR.exists() else Path("models")

# Age model (Caffe)
AGE_PROTO = MODELS_DIR / "age_deploy.prototxt"
AGE_MODEL = MODELS_DIR / "age_net.caffemodel"
AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
AGE_INPUT_SIZE = (227, 227)
AGE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

# Gender model (Caffe)
GENDER_PROTO = MODELS_DIR / "gender_deploy.prototxt"
GENDER_MODEL = MODELS_DIR / "gender_net.caffemodel"
GENDER_LABELS = ["Male", "Female"]

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect faces using Haar cascade
    if 'face_detector' not in globals():
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        globals()['face_detector'] = cv2.CascadeClassifier(cascade_path)

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Lazy-load age/gender networks once
    if 'age_net' not in globals():
        if AGE_PROTO.exists() and AGE_MODEL.exists():
            try:
                globals()['age_net'] = cv2.dnn.readNetFromCaffe(str(AGE_PROTO), str(AGE_MODEL))
                print("[INFO] Age model loaded from:", MODELS_DIR)
            except Exception as e:
                print("[WARN] Failed to load age model:", e)
                globals()['age_net'] = None
        else:
            globals()['age_net'] = None
            print("[WARN] Age model files not found in:", MODELS_DIR)

    if 'gender_net' not in globals():
        if GENDER_PROTO.exists() and GENDER_MODEL.exists():
            try:
                globals()['gender_net'] = cv2.dnn.readNetFromCaffe(str(GENDER_PROTO), str(GENDER_MODEL))
                print("[INFO] Gender model loaded from:", MODELS_DIR)
            except Exception as e:
                print("[WARN] Failed to load gender model:", e)
                globals()['gender_net'] = None
        else:
            globals()['gender_net'] = None
            print("[WARN] Gender model files not found in:", MODELS_DIR)

    for (x, y, w, h) in faces:
        left, top, right, bottom = x, y, x + w, y + h
        face_roi = frame[top:bottom, left:right]
        if face_roi.size == 0:
            continue
        face_embedding = compute_embedding(face_roi)
        customer_id = find_customer_id(face_embedding)

        if customer_id:
            label = f"{customer_id} (Returning)"
        else:
            customer_id = f"C{next_id}"
            next_id += 1
            database["customers"].append({
                "id": customer_id,
                "embedding": face_embedding.tolist(),
                "visits": []
            })
            save_db()
            label = f"{customer_id} (New)"

        # Predict age
        age_bucket = None
        if 'age_net' in globals() and age_net is not None:
            try:
                age_blob = cv2.dnn.blobFromImage(face_roi, 1.0, AGE_INPUT_SIZE, AGE_MEAN, swapRB=False, crop=False)
                age_net.setInput(age_blob)
                age_preds = age_net.forward()
                age_bucket = AGE_LIST[int(np.argmax(age_preds[0]))]
            except Exception as e:
                age_bucket = None

        # Predict gender
        gender_label = None
        if 'gender_net' in globals() and gender_net is not None:
            try:
                gender_blob = cv2.dnn.blobFromImage(face_roi, 1.0, AGE_INPUT_SIZE, AGE_MEAN, swapRB=False, crop=False)
                gender_net.setInput(gender_blob)
                gender_preds = gender_net.forward()
                gender_label = GENDER_LABELS[int(np.argmax(gender_preds[0]))]
            except Exception as e:
                gender_label = None

        # Log visit (with age/gender if available)
        log_visit(customer_id, age_bucket=age_bucket, gender=gender_label)

        # Get a mock ad recommendation
        visits = [cust["visits"] for cust in database["customers"] if cust["id"] == customer_id][0]
        ad = recommend_ad(customer_id, visits)

        # Draw face box + label + age/gender + ad
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        overlay_label = label
        if gender_label:
            overlay_label += f"  Gender: {gender_label}"
        if age_bucket:
            overlay_label += f"  Age: {age_bucket}"
        cv2.putText(frame, overlay_label, (left, top - 10 if top - 10 > 10 else top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Ad: {ad}", (left, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Mall Profiling Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
