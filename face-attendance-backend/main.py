import os
import glob
import pickle
import platform
import threading
import time
import logging
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Face recognition import
try:
    import face_recognition
    HAVE_FACE_RECOG = True
except Exception as e:
    print(f"[WARN] face_recognition missing: {e}")
    HAVE_FACE_RECOG = False

# Config
DEBUG = True
HOST, PORT = "127.0.0.1", 8000

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))

FACE_DATA_DIR = os.path.join(PROJECT_ROOT, "Images")
ENCODINGS_DIR = os.path.join(BACKEND_ROOT, "data", "encodings")
LOGS_DIR = os.path.join(BACKEND_ROOT, "logs")

os.makedirs(ENCODINGS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Logging
log_file = os.path.join(LOGS_DIR, f"server_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger("face-attendance")


# Globals
latest_frame = None
latest_frame_lock = threading.Lock()
latest_frame_cond = threading.Condition(latest_frame_lock)

recognition_running = False
stop_streaming = False
active_camera = None

known_face_encodings, known_face_names = [], []
encodings_lock = threading.Lock()

# Attendance tracking
attendance_records = []
attendance_lock = threading.Lock()

# Database global
db = None

FACE_MATCH_THRESHOLD = 0.5  # Lower threshold for better recognition
PROCESS_EVERY_N_FRAMES = 2
JPEG_QUALITY = 70
FRAME_SCALE = 0.5  # Scale down for faster processing

# App
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Start background face loading
@app.on_event("startup")
def startup():
    threading.Thread(target=background_load_faces, daemon=True).start()
    logger.info("🚀 Server started - loading faces from disk")


# === FACE LOADING === #
def load_faces_from_disk():
    global known_face_encodings, known_face_names
    logger.info("🔄 Loading saved face encodings...")
    logger.info(f"📂 Looking for encodings in: {ENCODINGS_DIR}")

    encs = glob.glob(os.path.join(ENCODINGS_DIR, "*.pkl"))
    logger.info(f"📄 Found {len(encs)} .pkl files")

    with encodings_lock:
        known_face_encodings.clear()
        known_face_names.clear()

        for file in encs:
            name = os.path.splitext(os.path.basename(file))[0]
            try:
                with open(file, 'rb') as f:
                    enc_list = pickle.load(f)
                # enc_list is a list of encodings for this student
                loaded_count = 0
                for enc in enc_list:
                    if isinstance(enc, np.ndarray) and enc.shape == (128,):
                        known_face_encodings.append(enc)
                        known_face_names.append(name)
                        loaded_count += 1
                    else:
                        logger.warning(f"Skipping invalid encoding shape: {getattr(enc, 'shape', 'unknown')} for {name}")
                logger.info(f"✅ Loaded {loaded_count}/{len(enc_list)} encodings for: {name}")
            except Exception as e:
                logger.warning(f"❌ Failed to load {file}: {e}")

    logger.info(f"✅ Loaded {len(known_face_encodings)} known face encodings total")


def background_load_faces():
    try:
        load_faces_from_disk()
    except Exception as e:
        logger.error(f"Error loading faces: {e}")


# === CAMERA === #
def release_camera():
    global active_camera
    if active_camera:
        active_camera.release()
        active_camera = None


def open_camera():
    global active_camera
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            active_camera = cap
            logger.info(f"🎥 Camera opened at index {i}")
            return cap
    logger.error("❌ No camera detected")
    return None


# === RECOGNITION THREAD === #
def recognition_loop():
    global latest_frame, recognition_running, stop_streaming

    cap = open_camera()
    if not cap:
        recognition_running = False
        return

    frame_count = 0

    while recognition_running and not stop_streaming:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces every frame for bounding boxes
        if HAVE_FACE_RECOG:
            faces_small = face_recognition.face_locations(rgb_small, model="hog")

            # Scale back to original size
            faces = [(int(top / FRAME_SCALE), int(right / FRAME_SCALE), int(bottom / FRAME_SCALE), int(left / FRAME_SCALE))
                     for (top, right, bottom, left) in faces_small]

            # Draw bounding boxes for all detected faces
            for (top, right, bottom, left) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Do recognition every N frames
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                encs = face_recognition.face_encodings(rgb_small, faces_small)

                for (top, right, bottom, left), enc in zip(faces, encs):
                    name = "Unknown"
                    if len(known_face_encodings) > 0:
                        dists = face_recognition.face_distance(known_face_encodings, enc)
                        if len(dists) > 0:
                            idx = np.argmin(dists)
                            if dists[idx] <= FACE_MATCH_THRESHOLD:
                                name = known_face_names[idx]

                                # Record attendance
                                with attendance_lock:
                                    if not any(record['name'] == name for record in attendance_records):
                                        record = {
                                            'name': name,
                                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            'course': 'Unknown',
                                            'year': 'Unknown'
                                        }
                                        attendance_records.append(record)

                                        # Save to database asynchronously
                                        if db:
                                            threading.Thread(target=save_attendance_to_db, args=(record,), daemon=True).start()

                    # Draw name immediately
                    cv2.putText(frame, name, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_count += 1

        # Always update the frame for smooth streaming
        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        with latest_frame_lock:
            latest_frame = jpeg.tobytes()
            latest_frame_cond.notify_all()

    release_camera()
    logger.info("🛑 Recognition stopped")


# === STREAM === #
def frame_stream():
    while recognition_running:
        with latest_frame_cond:
            latest_frame_cond.wait(timeout=0.2)
            if latest_frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame + b"\r\n"


# === ROUTES === #
@app.get("/")
def root():
    return {"status": "ok", "faces_loaded": len(known_face_names)}


@app.get("/health")
def health():
    return {"running": recognition_running, "faces": len(known_face_names)}


@app.get("/video")
def video():
    return StreamingResponse(frame_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/start")
def start():
    global recognition_running, stop_streaming
    if recognition_running:
        return {"status": "already_running"}

    recognition_running = True
    stop_streaming = False

    threading.Thread(target=recognition_loop, daemon=True).start()
    return {"status": "started"}


@app.post("/stop")
def stop():
    global recognition_running, stop_streaming
    recognition_running = False
    stop_streaming = True
    release_camera()
    return {"status": "stopped"}


@app.post("/reload_faces")
def reload_faces():
    threading.Thread(target=background_load_faces, daemon=True).start()
    return {"status": "reloading"}




@app.get("/status")
def get_status():
    """Get current system status."""
    return {
        "status": "running" if recognition_running else "stopped",
        "recognition_running": recognition_running,
        "camera_active": active_camera is not None,
        "faces_loaded": len(known_face_names)
    }


@app.get("/camera_status")
def camera_status():
    """Get camera status and frame availability."""
    return {
        "camera_active": active_camera is not None,
        "has_frame": latest_frame is not None,
        "recognition_running": recognition_running
    }


@app.get("/attendance")
def get_attendance():
    """Get current attendance records."""
    with attendance_lock:
        return {"attendance": attendance_records.copy()}

@app.get("/attendance-db")
async def get_attendance_from_db():
    """Get attendance records from database."""
    try:
        attendance_collection = db.attendance
        records = []
        async for record in attendance_collection.find().sort("timestamp", -1).limit(100):
            record['_id'] = str(record['_id'])  # Convert ObjectId to string
            records.append(record)
        return {"attendance": records}
    except Exception as e:
        logger.error(f"❌ Failed to fetch attendance from DB: {e}")
        return JSONResponse({"error": "Failed to fetch attendance"}, status_code=500)


@app.get("/snapshot")
def snapshot():
    """Get a snapshot from the camera."""
    if latest_frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=404)

    from fastapi.responses import Response
    return Response(content=latest_frame, media_type="image/jpeg")


@app.post("/clear_attendance")
def clear_attendance():
    """Clear attendance records."""
    global attendance_records
    with attendance_lock:
        attendance_records.clear()
    return {"status": "cleared"}




# === STARTUP === #
# Note: Database startup is handled above, this is for backward compatibility
# This event is now redundant but kept for compatibility


# === MAIN ENTRY === #
if __name__ == "__main__":
    import uvicorn
    logger.info(f"✅ Server listening at http://{HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)