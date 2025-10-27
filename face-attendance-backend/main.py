# main.py
import os, glob, pickle, platform, threading, time, logging
from datetime import datetime
import cv2, numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Face Recognition import
# -------------------------
try:
    import face_recognition
    HAVE_FACE_RECOG = True
except Exception as e:
    print(f"[WARN] face_recognition import failed: {e}. Running stream-only mode.")
    face_recognition = None
    HAVE_FACE_RECOG = False

# -------------------------
# Config
# -------------------------
# Environment settings
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True")

# Project directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))

# Application folders
FACE_DATA_DIR = os.getenv("FACE_DATA_DIR", os.path.join(PROJECT_ROOT, "Images"))
DATA_DIR = os.path.join(BACKEND_ROOT, "data")
ENCODINGS_DIR = os.getenv("ENCODINGS_DIR", os.path.join(DATA_DIR, "encodings"))
LOGS_DIR = os.path.join(BACKEND_ROOT, "logs")

# Create required directories
for directory in [FACE_DATA_DIR, DATA_DIR, ENCODINGS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
log_file = os.path.join(LOGS_DIR, f"face_recognition_{datetime.now().strftime('%Y%m%d')}.log")
# Use UTF-8 encoding for the file handler to avoid encoding errors on Windows
file_handler = logging.FileHandler(log_file, encoding='utf-8')
console_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger("face-attendance")

# Create required directories
os.makedirs(ENCODINGS_DIR, exist_ok=True)
os.makedirs(FACE_DATA_DIR, exist_ok=True)

# Clean up redundant folders if they exist
redundant_folders = ['dataset', '__pycache__']
for folder in redundant_folders:
    folder_path = os.path.join(PROJECT_ROOT, folder)
    if os.path.exists(folder_path):
        try:
            import shutil
            shutil.rmtree(folder_path)
            logger.info(f"Cleaned up redundant folder: {folder}")
        except Exception as e:
            logger.warning(f"Could not remove {folder}: {e}")

# Recognition parameters (tuned for better accuracy)
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))  # Higher = more permissive (0.6 is a good default)
FACE_DETECTION_CONFIDENCE = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))
USE_GPU = os.getenv("USE_GPU", "1") in ("1", "true", "True")  # Enable GPU if available
MAX_ATTENDANCE = 5000
HOST, PORT = "127.0.0.1", 8000
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True")
ATTENDANCE_COOLDOWN = 30  # Seconds before same person can be logged again

# If the configured FACE_DATA_DIR doesn't contain image files, search a set
# of likely candidate directories and pick the first one that has images.
def discover_face_data_dir(current_dir):
    candidates = [
        current_dir,
        os.path.join(PROJECT_ROOT, 'Images'),
        os.path.join(PROJECT_ROOT, 'images'),
        os.path.join(PROJECT_ROOT, 'face-attendance-frontend', 'Images'),
        os.path.join(PROJECT_ROOT, 'face-attendance-frontend', 'public'),
        os.path.join(BACKEND_ROOT, 'Images'),
    ]
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        try:
            if os.path.exists(c):
                files = [f for f in os.listdir(c) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    logger.info(f"Using Images directory: {c} (found {len(files)} image(s))")
                    return c
        except Exception:
            continue
    # fallback to the original
    logger.info(f"Falling back to configured FACE_DATA_DIR: {current_dir}")
    return current_dir

# Re-evaluate FACE_DATA_DIR in case it's empty or points to an unrelated folder
FACE_DATA_DIR = discover_face_data_dir(FACE_DATA_DIR)

# -------------------------
# FastAPI Setup
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("face-attendance")
if not logger.handlers:
    level = logging.DEBUG if DEBUG else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

# -------------------------
# Globals
# -------------------------
latest_frame = None
latest_frame_lock = threading.Lock()
latest_frame_cond = threading.Condition(latest_frame_lock)
camera_lock = threading.Lock()
state_lock = threading.Lock()
recognition_thread = None
recognition_running = False
stop_streaming = False
active_camera = None
attendance_data, attendance_lock = [], threading.Lock()
known_face_encodings, known_face_names = [], []
last_attendance_time = {}  # Track last attendance time per person
encodings_lock = threading.Lock()

# -------------------------
# Known face setup
# -------------------------
known_face_files = [
    (os.path.join(FACE_DATA_DIR, "Web.jpeg"), "Jhon Webster P. Fortuna"),
    (os.path.join(FACE_DATA_DIR, "Denver.jpeg"), "John Denver A. Ezperanzate"),
    (os.path.join(FACE_DATA_DIR, "Jeric.jpeg"), "JC Jeric M. Rodelas"),
    (os.path.join(FACE_DATA_DIR, "Maricon.jpeg"), "Maricon S. Dela Cruz"),
    (os.path.join(FACE_DATA_DIR, "Mark.jpeg"), "John Mark S. Manangkil"),
    (os.path.join(FACE_DATA_DIR, "Dilean.jpeg"), "Dilean James B. Vito"),
    (os.path.join(FACE_DATA_DIR, "Urie.jpeg"), "John Uriel F. Medina"),
    (os.path.join(FACE_DATA_DIR, "Kua gelo.jpeg"), "Kua Gelo"),
]

# Dynamically discover any additional images
additional_images = [
    f for f in os.listdir(FACE_DATA_DIR) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
    and os.path.join(FACE_DATA_DIR, f) not in [path for path, _ in known_face_files]
]
if additional_images:
    logger.info("Found additional images: " + ", ".join(additional_images))

student_info = {
    "Jhon Webster P. Fortuna": {"course": "BSIS", "year": "4th Year", "section": "C"},
    "JC Jeric M. Rodelas": {"course": "BSIT", "year": "3rd Year", "section": "A"},
    "John Denver A. Ezperanzate": {"course": "BSCS", "year": "2nd Year", "section": "B"},
    "John Uriel F. Medina": {"course": "BSPsych", "year": "1st Year", "section": "D"},
    "Maricon S. Dela Cruz": {"course": "BSBA", "year": "2nd Year", "section": "A"},
    "John Mark S. Manangkil": {"course": "BSAIS", "year": "3rd Year", "section": "D"},
    "Dilean James B. Vito": {"course": "BSEntrep", "year": "4th Year", "section": "D"},
}

# -------------------------
# Load known faces (from both Images and encodings folder)
# -------------------------
def load_all_trained_faces():
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = [], []
    logger.info("=" * 60)
    logger.info("LOADING FACE ENCODINGS")
    logger.info("=" * 60)

    # Helper function to encode faces from image
    def encode_image(img_path, name, save_encoding=True):
        """
        Encode a face image and optionally save the encoding to the encodings folder.
        Returns: (success, encoding)
        """
        try:
            logger.info(f"Processing: {img_path}")
            img = face_recognition.load_image_file(img_path)
            
            # Try different face locations to get the best encoding
            locations = face_recognition.face_locations(img, model="cnn" if USE_GPU else "hog")
            if not locations:
                logger.info(f"  Trying HOG model for {os.path.basename(img_path)}")
                locations = face_recognition.face_locations(img, model="hog")
            
            if locations:
                logger.info(f"  Found {len(locations)} face(s) in {os.path.basename(img_path)}")
                # Get multiple encodings if possible
                encodings = face_recognition.face_encodings(img, locations)
                if encodings:
                    # Use the first encoding (usually the clearest face)
                    encoding = encodings[0]
                    
                    # Save encoding to file if requested
                    if save_encoding:
                        # sanitize filename for the encoding file
                        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
                        encoding_path = os.path.join(ENCODINGS_DIR, f"{safe_name}.pkl")
                        try:
                            with open(encoding_path, 'wb') as f:
                                pickle.dump(encoding, f)
                            logger.info(f"  [OK] Saved encoding to {encoding_path}")
                        except Exception as e:
                            logger.error(f"  [ERROR] Could not save encoding: {e}")
                    
                    # Add to runtime lists
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                    logger.info(f"  [OK] Successfully encoded {name} from {os.path.basename(img_path)}")
                    return True, encoding
                else:
                    logger.warning(f"  [WARN] Could not generate encoding for {img_path}")
            else:
                logger.warning(f"  [WARN] No face detected in {img_path}")
            return False, None
        except Exception as e:
            logger.error(f"  ✗ Error processing {img_path}: {e}")
            return False

    # First load from known_face_files (Images folder)
    logger.info(f"\nChecking Images folder: {FACE_DATA_DIR}")
    logger.info("-" * 60)
    
    # First verify the Images directory exists
    if not os.path.exists(FACE_DATA_DIR):
        logger.error(f"Images directory not found at: {FACE_DATA_DIR}")
        logger.error(f"Current directory is: {os.getcwd()}")
        logger.error("Please check the path configuration")
    else:
        logger.info(f"Images directory found at: {FACE_DATA_DIR}")
        logger.info("Found images: " + ", ".join(
            f for f in os.listdir(FACE_DATA_DIR) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ))
        
    # Build lists of files to load/encode
    tasks = []  # list of (path, name) images to encode

    # Explicit known files
    for path, name in known_face_files:
        if os.path.exists(path):
            tasks.append((path, name))
        else:
            logger.warning(f"File not found: {path} (this is normal if the student's image hasn't been added yet)")

    # Additional images discovered in the images folder
    for fname in additional_images:
        full = os.path.join(FACE_DATA_DIR, fname)
        derived_name = os.path.splitext(fname)[0].replace('_', ' ').strip()
        logger.info(f"Queuing additional image {full} as '{derived_name}'")
        tasks.append((full, derived_name))

    # Load any existing pickled encodings first (fast)
    logger.info(f"\nChecking Encodings folder: {ENCODINGS_DIR}")
    logger.info("-" * 60)
    pkl_files = glob.glob(os.path.join(ENCODINGS_DIR, "*.pkl"))
    logger.info(f"Found {len(pkl_files)} .pkl file(s)")
    for pkl_file in pkl_files:
        try:
            name = os.path.splitext(os.path.basename(pkl_file))[0]
            logger.info(f"Loading: {pkl_file}")
            with open(pkl_file, 'rb') as f:
                enc = pickle.load(f)
                if isinstance(enc, list) and enc:
                    with encodings_lock:
                        known_face_encodings.append(enc[0])
                        known_face_names.append(name)
                    logger.info(f"  [OK] Loaded encoding for {name} (list format)")
                elif enc is not None:
                    with encodings_lock:
                        known_face_encodings.append(enc)
                        known_face_names.append(name)
                    logger.info(f"  [OK] Loaded encoding for {name}")
        except Exception as e:
            logger.error(f"  [ERROR] Could not load {pkl_file}: {e}")

    # Then look for image files in the encodings folder to encode if present
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(ENCODINGS_DIR, ext)))
    logger.info(f"Found {len(image_files)} image file(s) in encodings folder")
    for img_path in image_files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        tasks.append((img_path, name))

    # Encode images in parallel to speed up startup. face_recognition operations are
    # mostly implemented in C and release the GIL, so a ThreadPoolExecutor is a good fit.
    if tasks:
        logger.info(f"Encoding {len(tasks)} image(s) using ThreadPoolExecutor")
        try:
            import concurrent.futures
            max_workers = min(4, (os.cpu_count() or 1))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(encode_image, path, name) for path, name in tasks]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        logger.error(f"Error encoding image in worker: {e}")
        except Exception as e:
            logger.warning(f"Parallel encoding failed, falling back to sequential: {e}")
            for path, name in tasks:
                try:
                    encode_image(path, name)
                except Exception as ee:
                    logger.error(f"Sequential encode failed for {path}: {ee}")

    logger.info("=" * 60)
    logger.info(f"TOTAL KNOWN FACES LOADED: {len(known_face_encodings)}")
    logger.info("=" * 60)
    if known_face_encodings:
        logger.info("Known names:")
        for i, name in enumerate(known_face_names, 1):
            logger.info(f"  {i}. {name}")
    else:
        logger.error("WARNING: NO FACES LOADED! Recognition will not work.")
    logger.info("=" * 60)

# -------------------------
# Camera control
# -------------------------
def release_camera():
    global active_camera
    with camera_lock:
        if active_camera is not None:
            try: active_camera.release()
            except: pass
            active_camera = None
            logger.info("Camera released")

def open_camera(max_idx=5):
    global active_camera
    with camera_lock:
        if active_camera and active_camera.isOpened():
            return active_camera
        for idx in range(max_idx):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            ret, _ = cap.read()
            if ret:
                active_camera = cap
                logger.info(f"Camera opened at index {idx}")
                return cap
        logger.error("No camera found")
        return None

# -------------------------
# Recognition worker
# -------------------------
def background_face_recognition():
    global recognition_running, stop_streaming, latest_frame
    logger.info("Recognition worker started")
    cap = open_camera()
    if cap is None:
        logger.error("Camera unavailable")
        return

    # Use 'cnn' if available, else 'hog' for speed
    model = "hog"  # Default to HOG for better compatibility
    try:
        import importlib
        spec = importlib.util.find_spec("torch")
        if spec is not None:
            torch = importlib.import_module("torch")
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                model = "cnn"
    except Exception:
        pass
    logger.info(f"Using face_recognition model: {model}")
    
    if not HAVE_FACE_RECOG or not known_face_encodings:
        logger.warning("No face recognition available - streaming only mode")

    frame_count = 0
    process_every_n_frames = 1  # Process every frame (improves detection reliability)

    while recognition_running and not stop_streaming:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Only process face recognition every N frames
        if frame_count % process_every_n_frames == 0 and HAVE_FACE_RECOG and known_face_encodings:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Increased from 0.25 to 0.5 for better detection
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            try:
                # Detect faces (upsample once to improve small-face detection)
                locs = face_recognition.face_locations(rgb_small, model=model, number_of_times_to_upsample=1)
                
                if locs:
                    # Get encodings for all detected faces
                    encs = face_recognition.face_encodings(rgb_small, locs)
                    
                    for (t, r, b, l), enc in zip(locs, encs):
                        name = "Unknown"
                        best_dist = 1.0
                        
                        # Compare against all known faces
                        if known_face_encodings:
                            dists = face_recognition.face_distance(known_face_encodings, enc)
                            min_dist = float(np.min(dists))
                            min_idx = int(np.argmin(dists))
                            
                            logger.debug(f"Best match: {known_face_names[min_idx]} with distance {min_dist:.3f}")
                            
                            if min_dist <= FACE_MATCH_THRESHOLD:
                                name = known_face_names[min_idx]
                                best_dist = min_dist
                                logger.info(f"[OK] Recognized: {name} (confidence: {(1-min_dist)*100:.1f}%)")
                        
                        # Scale coordinates back to full size
                        top, right, bottom, left = [int(v / 0.5) for v in (t, r, b, l)]
                        
                        # Draw bounding box and label
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Show name and confidence score
                        conf_text = f"{name} ({(1-best_dist)*100:.1f}%)"
                        cv2.putText(frame, conf_text, (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # Log attendance with cooldown
                        if name != "Unknown":
                            current_time = time.time()
                            last_time = last_attendance_time.get(name, 0)
                            
                            if current_time - last_time >= ATTENDANCE_COOLDOWN:
                                with attendance_lock:
                                    attendance_data.append({
                                        "name": name,
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "course": student_info.get(name, {}).get("course", "N/A"),
                                        "year": student_info.get(name, {}).get("year", "N/A"),
                                        "section": student_info.get(name, {}).get("section", "N/A"),
                                    })
                                    last_attendance_time[name] = current_time
                                    logger.info(f"[ATTENDANCE] {name}")
                            else:
                                remaining = int(ATTENDANCE_COOLDOWN - (current_time - last_time))
                                logger.debug(f"Cooldown active for {name}: {remaining}s remaining")

            except Exception as e:
                logger.exception(f"Recognition error: {e}")

        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        with latest_frame_lock:
            latest_frame = jpeg.tobytes()
            latest_frame_cond.notify_all()
        time.sleep(0.03)

    release_camera()
    logger.info("Recognition stopped")

# -------------------------
# Streaming
# -------------------------
def generate_frames():
    while recognition_running and not stop_streaming:
        with latest_frame_cond:
            latest_frame_cond.wait(timeout=0.1)
            frame = latest_frame
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

# -------------------------
# API Endpoints
# -------------------------
@app.post("/start")
def start():
    global recognition_running, recognition_thread, stop_streaming
    with state_lock:
        if recognition_running:
            return {"status": "already_running"}
        recognition_running = True
        stop_streaming = False

    recognition_thread = threading.Thread(target=background_face_recognition, daemon=True)
    recognition_thread.start()

    start_time = time.time()
    while time.time() - start_time < 5:
        with latest_frame_lock:
            if latest_frame is not None:
                return {"status": "started", "ready": True}
        time.sleep(0.2)
    return {"status": "started", "ready": False}

@app.post("/stop")
def stop():
    global recognition_running, stop_streaming
    with state_lock:
        recognition_running = False
        stop_streaming = True
    release_camera()
    return {"status": "stopped"}

@app.get("/camera_status")
def camera_status():
    return {"running": recognition_running, "has_frame": latest_frame is not None,
            "camera_active": active_camera is not None, "threshold": FACE_MATCH_THRESHOLD,
            "known_faces": len(known_face_encodings)}

@app.get("/video")
def video():
    return StreamingResponse(generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/attendance")
def attendance():
    with attendance_lock:
        return {"attendance": attendance_data, "count": len(attendance_data)}

@app.get("/status")
def status():
    return {"recognition_running": recognition_running, "camera_active": active_camera is not None,
            "status": "running" if recognition_running else "stopped", 
            "known_faces": len(known_face_encodings)}

@app.get("/known_faces")
def get_known_faces():
    """List all loaded face names"""
    return {"faces": known_face_names, "count": len(known_face_names)}

if __name__ == "__main__":
    load_all_trained_faces()
    logger.info(f"Starting server on {HOST}:{PORT}")
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
                