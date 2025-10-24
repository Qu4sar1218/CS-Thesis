import cv2
import os
try:
    import face_recognition
    HAVE_FACE_RECOG = True
except Exception as e:
    print(f"[WARN] face_recognition import failed: {e}. Running in stream-only mode.")
    face_recognition = None
    HAVE_FACE_RECOG = False

import numpy as np
import mediapipe as mp
from datetime import datetime
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import uvicorn
import platform
import logging

# FastAPI Setup
app = FastAPI()
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Globals and synchronization
latest_frame = None
latest_frame_lock = threading.Lock()
latest_frame_cond = threading.Condition(latest_frame_lock)
camera_lock = threading.Lock()
state_lock = threading.Lock()

recognition_thread = None
recognition_running = False
stop_streaming = False
active_camera = None

attendance_data = []
attendance_lock = threading.Lock()

# Logging & Config
logger = logging.getLogger("face-attendance")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.48"))
BASE_DIR = os.getenv("FACE_DATA_DIR", "C:/CS Thesis/Images")
MAX_ATTENDANCE = int(os.getenv("MAX_ATTENDANCE", "5000"))
MAX_STREAMS = int(os.getenv("MAX_STREAMS", "3"))
active_streams = 0
stream_lock = threading.Lock()
rate_lock = threading.Lock()
rate_buckets = {}
RATE_WINDOW = float(os.getenv("RATE_WINDOW", "1.0"))
RATE_MAX = int(os.getenv("RATE_MAX", "10"))

# Load known faces
known_face_files = [
    (os.path.join(BASE_DIR, "Denver.jpeg"), "John Denver A. Ezperanzate"),
    (os.path.join(BASE_DIR, "JC.jpeg"), "JC Jeric M. Rodelas"),
    (os.path.join(BASE_DIR, "Web.jpeg"), "Jhon Webster P. Fortuna"),
    (os.path.join(BASE_DIR, "Urie.jpeg"), "John Uriel F. Medina"),
    (os.path.join(BASE_DIR, "Maricon.jpeg"), "Maricon S. Dela Cruz"),
    (os.path.join(BASE_DIR, "Mark.jpeg"), "John Mark S. Manangkil"),
    (os.path.join(BASE_DIR, "Dilean.jpeg"), "Dilean James B. Vito"),
]

known_face_encodings = []
known_face_names = []

print("Loading face encodings...")
for file, name in known_face_files:
    try:
        if HAVE_FACE_RECOG:
            img = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"[OK] Loaded encoding for {name}")
            else:
                print(f"[WARN] No face found in {file}")
    except Exception as e:
        print(f"[ERROR] Could not load {file}: {e}")

# Student info
student_info = {
    "JC Jeric M. Rodelas": {"course": "BSIT", "year": "3rd Year", "section": "A"},
    "John Denver A. Ezperanzate": {"course": "BSCS", "year": "2nd Year", "section": "B"},
    "Jhon Webster P. Fortuna": {"course": "BSIS", "year": "4th Year", "section": "C"},
    "John Uriel F. Medina": {"course": "BSPsych", "year": "1st Year", "section": "D"},
    "Maricon S. Dela Cruz": {"course": "BSBA", "year": "2nd Year", "section": "A"},
    "John Mark S. Manangkil": {"course": "BSAIS", "year": "3rd Year", "section": "D"},
    "Dilean James B. Vito": {"course": "BSEntrep", "year": "4th Year", "section": "D"},
}

# Mediapipe blink detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = None  # disabled to save resources; enable if blink detection is used

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.2

def eye_aspect_ratio(landmarks, eye_points):
    p2_p6 = np.linalg.norm(landmarks[eye_points[1]] - landmarks[eye_points[5]])
    p3_p5 = np.linalg.norm(landmarks[eye_points[2]] - landmarks[eye_points[4]])
    p1_p4 = np.linalg.norm(landmarks[eye_points[0]] - landmarks[eye_points[3]])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

# Brightness control
DARK_THRESHOLD = 60
MIN_BRIGHTNESS = -64
MAX_BRIGHTNESS = 100
BRIGHTNESS_STEP = 10

class BrightnessController:
    def __init__(self):
        self.current_brightness = 0
        self.last_adjustment_time = 0
        self.cooldown = 3.0
        
    def calculate_frame_brightness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def adjust_brightness(self, frame, target_brightness):
        if time.time() - self.last_adjustment_time < self.cooldown:
            return frame
        diff = target_brightness - np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if abs(diff) < 10:
            return frame
        self.last_adjustment_time = time.time()
        adjustment = np.clip(diff / 2, -BRIGHTNESS_STEP, BRIGHTNESS_STEP)
        self.current_brightness = np.clip(self.current_brightness + adjustment, MIN_BRIGHTNESS, MAX_BRIGHTNESS)
        adjusted = np.clip(frame.astype(np.float32) + self.current_brightness, 0, 255)
        return adjusted.astype(np.uint8)

brightness_controller = BrightnessController()

# Camera utilities
def release_camera():
    global active_camera
    with camera_lock:
        if active_camera is not None:
            try:
                active_camera.release()
                print("[INFO] Camera released")
            except Exception as e:
                print(f"[WARN] Error releasing camera: {e}")
            finally:
                active_camera = None
                time.sleep(0.4)

def open_camera():
    global active_camera
    with camera_lock:
        if active_camera is not None and getattr(active_camera, 'isOpened', lambda: False)():
            return active_camera

        use_directshow = platform.system().lower().startswith("windows")
        for idx in range(0, 3):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if use_directshow else cv2.VideoCapture(idx)
                if not cap or not cap.isOpened():
                    try:
                        cap.release()
                    except:
                        pass
                    continue
                ret, _ = cap.read()
                if ret:
                    active_camera = cap
                    # try to minimize buffering and target a reasonable FPS
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                    except Exception as e:
                        print(f"[WARN] could not set camera properties: {e}")
                    print(f"[OK] Camera opened on index {idx}")
                    return cap
                else:
                    try:
                        cap.release()
                    except:
                        pass
            except Exception as e:
                print(f"[WARN] open_camera idx {idx} failed: {e}")
        print("[ERROR] Could not open camera")
        return None

# Background face recognition
def background_face_recognition():
    global recognition_running, stop_streaming, latest_frame
    print('[INFO] recognition worker started')
    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(default_frame, 'No Camera', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cap = open_camera()
    if cap is None:
        print('[WARN] worker: no camera initially')

    consecutive_failures = 0
    frame_counter = 0
    while recognition_running and not stop_streaming:
        try:
            if cap is None:
                cap = open_camera()
                if cap is None:
                    time.sleep(0.5)
                    continue

            # Prefer grab/retrieve to drop queued frames and reduce latency
            try:
                cap.grab()
                ret, frame = cap.retrieve()
            except Exception:
                ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                frame = default_frame.copy()
                if consecutive_failures > 20:
                    print('[WARN] too many read failures, reopening camera')
                    try:
                        cap.release()
                    except:
                        pass
                    cap = None
                    consecutive_failures = 0
                    continue
            else:
                consecutive_failures = 0
                frame = cv2.flip(frame, 1)
                frame = brightness_controller.adjust_brightness(frame, 80)

                run_recognition = (frame_counter % 5) == 0
                names = []

                if run_recognition and HAVE_FACE_RECOG and known_face_encodings:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    try:
                        face_locations_small = face_recognition.face_locations(rgb_small)
                        encodings_small = face_recognition.face_encodings(rgb_small, face_locations_small)
                    except Exception as e:
                        face_locations_small = []
                        encodings_small = []

                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    for (t, r, b, l), enc in zip(face_locations_small, encodings_small):
                        top, right, bottom, left = [int(x / 0.25) for x in (t, r, b, l)]

                        try:
                            dists = face_recognition.face_distance(known_face_encodings, enc)
                            if len(dists) > 0:
                                best_idx = int(np.argmin(dists))
                                best_dist = float(dists[best_idx])
                                if best_dist <= FACE_MATCH_THRESHOLD:
                                    name = known_face_names[best_idx]
                                else:
                                    name = 'Unknown'
                            else:
                                name = 'Unknown'
                        except Exception:
                            name = 'Unknown'

                        if name != 'Unknown':
                            with attendance_lock:
                                recent = any(r['name'] == name and (datetime.now() - datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S')).total_seconds() < 60 for r in attendance_data)
                                if not recent:
                                    attendance_data.append({
                                        'name': name,
                                        'timestamp': current_time,
                                        'course': student_info.get(name, {}).get('course', 'Unknown'),
                                        'year': student_info.get(name, {}).get('year', 'Unknown')
                                    })
                                    if len(attendance_data) > MAX_ATTENDANCE:
                                        del attendance_data[:-MAX_ATTENDANCE]

                        names.append((left, top, right, bottom, name))

                for (left, top, right, bottom, name) in names:
                    try:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, max(top - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except Exception:
                        pass

                frame_counter += 1

            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            # enable progressive if available to smooth streaming
            try:
                encode_params += [int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
            except Exception:
                pass
            # Downscale for streaming to reduce bandwidth/latency
            try:
                h, w = frame.shape[:2]
                stream_frame = frame
                if w > 1280:
                    new_h = int(h * 1280 / w)
                    stream_frame = cv2.resize(frame, (1280, new_h))
            except Exception:
                stream_frame = frame
            _, jpeg = cv2.imencode('.jpg', stream_frame, encode_params)
            with latest_frame_lock:
                latest_frame = jpeg.tobytes()
                # notify streamers that a new frame is available
                try:
                    latest_frame_cond.notify_all()
                except Exception:
                    pass

            time.sleep(0.03)
        except Exception as e:
            print(f'[ERROR] worker loop: {e}')
            try:
                if cap is not None:
                    cap.release()
            except:
                pass
            cap = None
            time.sleep(0.5)

    try:
        if cap is not None:
            cap.release()
    except:
        pass
    print('[INFO] recognition worker exiting')

# Streaming generator
def generate_frames():
    global latest_frame
    try:
        while recognition_running and not stop_streaming:
            # wait for a new frame (or timeout) to avoid fixed-rate sleep
            with latest_frame_cond:
                latest_frame_cond.wait(timeout=0.1)
                frame = latest_frame
            if frame:
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Cache-Control: no-cache\r\n\r\n' + frame + b'\r\n')
                except GeneratorExit:
                    break
                except Exception as e:
                    print(f'[WARN] stream yield: {e}')
                    break
    except Exception as e:
        print(f'[DEBUG] generate_frames: {e}')
    finally:
        return

# Auth and rate limiting

def require_auth(authorization: str = Header(None)):
    expected = os.getenv("API_TOKEN")
    if expected:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")
        token = authorization.split(" ", 1)[1]
        if token != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")
    return True


def rate_limiter(request: Request):
    ip = request.client.host if request and request.client else "unknown"
    now = time.time()
    window = float(os.getenv("RATE_WINDOW", "1.0"))
    max_req = int(os.getenv("RATE_MAX", "10"))
    if max_req <= 0:
        return True
    with rate_lock:
        count, start = rate_buckets.get(ip, (0, now))
        if now - start > window:
            count, start = 0, now
        count += 1
        rate_buckets[ip] = (count, start)
        if count > max_req:
            raise HTTPException(status_code=429, detail="Too Many Requests")
    return True

# Endpoints
@app.get('/')
def index():
    return {'message': 'Face recognition backend running'}

@app.get('/status')
def get_status():
    return JSONResponse({
        'recognition_running': recognition_running,
        'camera_active': active_camera is not None,
        'status': 'running' if recognition_running else 'stopped'
    })

@app.get('/camera_status')
def camera_status():
    return JSONResponse({
        'running': bool(recognition_running),
        'has_frame': bool(latest_frame is not None),
        'camera_active': active_camera is not None
    })

# Health endpoints
@app.get('/health/live')
def health_live():
    return {'status': 'ok'}

@app.get('/health/ready')
def health_ready():
    return {
        'recognition_running': recognition_running,
        'camera_active': active_camera is not None,
        'has_frame': latest_frame is not None
    }

@app.get('/video')
def video(_: bool = Depends(require_auth)):
    global active_streams
    with stream_lock:
        if active_streams >= MAX_STREAMS:
            return JSONResponse({'error': 'Too many concurrent streams'}, status_code=429)
        active_streams += 1

    def client_stream():
        try:
            for part in generate_frames():
                yield part
        finally:
            global active_streams
            with stream_lock:
                active_streams = max(0, active_streams - 1)

    return StreamingResponse(
        client_stream(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache'}
    )

@app.get('/snapshot')
def snapshot(_: bool = Depends(require_auth)):
    with latest_frame_lock:
        if latest_frame is not None:
            return StreamingResponse(iter([latest_frame]), media_type='image/jpeg', headers={'Cache-Control': 'no-cache'})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, 'Starting up...', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, jpeg = cv2.imencode('.jpg', frame)
    return StreamingResponse(iter([jpeg.tobytes()]), media_type='image/jpeg', headers={'Cache-Control': 'no-cache'})

@app.get('/attendance')
def get_attendance(_: bool = Depends(require_auth)):
    with attendance_lock:
        return JSONResponse({'attendance': attendance_data, 'count': len(attendance_data)})

@app.get('/brightness')
def get_brightness():
    return JSONResponse({'current_adjustment': brightness_controller.current_brightness, 'dark_threshold': DARK_THRESHOLD})

@app.post('/start')
def start_simple(_: bool = Depends(require_auth), __: bool = Depends(rate_limiter)):
    # Call the handler directly with the resolved dependencies
    return start_recognition(_, __)

@app.post('/stop')
def stop_simple(_: bool = Depends(require_auth), __: bool = Depends(rate_limiter)):
    # Call the handler directly with the resolved dependencies
    return stop_recognition(_, __)

@app.post('/start_recognition')
def start_recognition(_: bool = Depends(require_auth), __: bool = Depends(rate_limiter)):
    global recognition_running, recognition_thread, stop_streaming
    with state_lock:
        if recognition_running:
            return {'status': 'already_running'}
        recognition_running = True
        stop_streaming = False

    recognition_thread = threading.Thread(target=background_face_recognition, daemon=True)
    recognition_thread.start()

    start = time.time()
    while time.time() - start < 6.0:
        with latest_frame_lock:
            if latest_frame is not None:
                return {'status': 'started', 'ready': True}
        time.sleep(0.1)

    return {'status': 'started', 'ready': False}

@app.post('/stop_recognition')
def stop_recognition(_: bool = Depends(require_auth), __: bool = Depends(rate_limiter)):
    global recognition_running, stop_streaming
    with state_lock:
        recognition_running = False
        stop_streaming = True

    if recognition_thread is not None:
        try:
            recognition_thread.join(timeout=2.0)
        except Exception:
            pass

    # wake up any waiting streamers
    with latest_frame_lock:
        try:
            latest_frame_cond.notify_all()
        except Exception:
            pass

    release_camera()
    return {'status': 'stopped'}

@app.on_event('shutdown')
def on_shutdown():
    global recognition_running, stop_streaming
    with state_lock:
        recognition_running = False
        stop_streaming = True
    if recognition_thread is not None:
        try:
            recognition_thread.join(timeout=1.0)
        except Exception:
            pass
    # wake up any waiting streamers
    with latest_frame_lock:
        try:
            latest_frame_cond.notify_all()
        except Exception:
            pass
    release_camera()

if __name__ == '__main__':
    print('='*50)
    print('Face Attendance System Backend')
    print('API: http://localhost:8000')
    print('='*50)
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=False)
