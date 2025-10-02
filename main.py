import cv2
import os
import json
import numpy as np
from datetime import datetime
from ads import recommend_ad

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

# Log visit in DB
def log_visit(customer_id, shop="general"):
    for cust in database["customers"]:
        if cust["id"] == customer_id:
            if "visits" not in cust:
                cust["visits"] = []
            cust["visits"].append({
                "shop": shop,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_db()
            return

# Initialize webcam
video_capture = cv2.VideoCapture(0)
next_id = len(database["customers"]) + 1

print("[INFO] Starting camera... Press 'q' to quit.")

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

        # Log visit
        log_visit(customer_id)

        # Get a mock ad recommendation
        visits = [cust["visits"] for cust in database["customers"] if cust["id"] == customer_id][0]
        ad = recommend_ad(customer_id, visits)

        # Draw face box + label + ad
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Ad: {ad}", (left, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Mall Profiling Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
