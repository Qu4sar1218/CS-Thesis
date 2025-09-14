import cv2
import face_recognition
import numpy as np
import mediapipe as mp
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import uvicorn

# -------------------------------
# FastAPI Setup
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load known faces
# -------------------------------
known_face_files = [
    ("C:/CS Thesis/Images/Maam A.png", "Adoree Ramos"),
    ("C:/CS Thesis/Images/Maam Abhie.jpeg", "Abhiekay Lavastida"),
    ("C:/CS Thesis/Images/Akes.jpeg", "Jhon Webster P. Fortuna"),
    ("C:/CS Thesis/Images/Urie.jpeg", "John Uriel F. Medina")
]

known_face_encodings = []
known_face_names = []

print("Loading face encodings...")
for file, name in known_face_files:
    try:
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

# -------------------------------
# Student info
# -------------------------------
student_info = {
    "Adoree Ramos": {"course": "BSIT", "year": "3rd Year", "section": "A",
                     "schedule": ["8:00 - Math", "10:00 - Programming", "1:00 - Networks"]},
    "Abhiekay Lavastida": {"course": "BSCS", "year": "2nd Year", "section": "B",
                           "schedule": ["9:00 - DB Systems", "11:00 - Web Dev", "2:00 - AI Basics"]},
    "Jhon Webster P. Fortuna": {"course": "BSIS", "year": "4th Year", "section": "C",
                                "schedule": ["7:30 - Thesis", "10:30 - Capstone", "3:00 - Cloud Computing"]},
    "John Uriel F. Medina": {"course": "BSPsych", "year": "1st Year", "section": "D",
                             "schedule": ["7:30 - IDk", "10:30 - Idk", "3:00 - Ewan IDK"]}
    
}

# -------------------------------
# Mediapipe for blink detection
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.2

def eye_aspect_ratio(landmarks, eye_points):
    p2_p6 = np.linalg.norm(landmarks[eye_points[1]] - landmarks[eye_points[5]])
    p3_p5 = np.linalg.norm(landmarks[eye_points[2]] - landmarks[eye_points[4]])
    p1_p4 = np.linalg.norm(landmarks[eye_points[0]] - landmarks[eye_points[3]])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

# -------------------------------
# Brightness Detection and Adjustment
# -------------------------------
DARK_THRESHOLD = 60  # Brightness threshold (0-255)
MIN_BRIGHTNESS = -64  # Minimum brightness adjustment
MAX_BRIGHTNESS = 100  # Maximum brightness adjustment
BRIGHTNESS_STEP = 10  # Step size for brightness adjustment

class BrightnessController:
    def __init__(self):
        self.current_brightness = 0
        self.target_brightness = 0
        self.last_adjustment_time = 0
        self.adjustment_cooldown = 2.0  # Wait 2 seconds between adjustments
        
    def calculate_frame_brightness(self, frame):
        """Calculate average brightness of the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def calculate_face_brightness(self, frame, face_locations):
        """Calculate average brightness in face regions"""
        if not face_locations:
            return self.calculate_frame_brightness(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_brightnesses = []
        
        for (top, right, bottom, left) in face_locations:
            # Ensure coordinates are within frame bounds
            top = max(0, min(top, frame.shape[0]-1))
            bottom = max(0, min(bottom, frame.shape[0]-1))
            left = max(0, min(left, frame.shape[1]-1))
            right = max(0, min(right, frame.shape[1]-1))
            
            if right > left and bottom > top:
                face_region = gray[top:bottom, left:right]
                if face_region.size > 0:
                    face_brightnesses.append(np.mean(face_region))
        
        return np.mean(face_brightnesses) if face_brightnesses else self.calculate_frame_brightness(frame)
    
    def should_adjust_brightness(self, brightness, has_face):
        """Determine if brightness adjustment is needed"""
        current_time = time.time()
        
        # Only adjust if cooldown period has passed
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return False, 0
        
        # If face detected and brightness is too low, increase brightness
        if has_face and brightness < DARK_THRESHOLD:
            needed_adjustment = min(BRIGHTNESS_STEP, MAX_BRIGHTNESS - self.current_brightness)
            if needed_adjustment > 0:
                return True, needed_adjustment
        
        # If no face or brightness is good, gradually return to normal
        elif self.current_brightness > 0:
            needed_adjustment = max(-BRIGHTNESS_STEP, MIN_BRIGHTNESS - self.current_brightness)
            if needed_adjustment < 0:
                return True, needed_adjustment
        
        return False, 0
    
    def adjust_brightness(self, frame, adjustment=None):
        """Apply brightness adjustment to frame"""
        if adjustment is not None:
            self.current_brightness = max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, 
                                        self.current_brightness + adjustment))
            self.last_adjustment_time = time.time()
        
        if self.current_brightness == 0:
            return frame
        
        # Convert to float to avoid overflow
        adjusted = frame.astype(np.float32)
        adjusted = adjusted + self.current_brightness
        
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)

# Initialize brightness controller
brightness_controller = BrightnessController()

# -------------------------------
# Tracking states
# -------------------------------
liveness_states = {}
attendance = {}
last_scanned_student = None
detected_name = "Unknown"
student_details = {}
current_brightness_level = 0

# -------------------------------
# Test endpoint
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Face Attendance System API is running!"}

@app.get("/test")
def test_endpoint():
    return {"status": "Server is working", "timestamp": datetime.now().isoformat()}

# -------------------------------
# Background recognition thread
# -------------------------------
recognition_thread = None
processing_frame = None
processing_lock = threading.Lock()
recognition_results = {"locations": [], "names": [], "last_update": 0}

def background_face_recognition():
    global processing_frame, recognition_results, detected_name, student_details, last_scanned_student
    
    while True:
        with processing_lock:
            frame_to_process = processing_frame.copy() if processing_frame is not None else None
            processing_frame = None
        
        if frame_to_process is not None:
            try:
                # Process at smaller resolution for speed
                small_frame = cv2.resize(frame_to_process, (160, 120))
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                
                temp_locations, temp_names = [], []
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if len(distances) > 0:
                        best_match = np.argmin(distances)
                        name = known_face_names[best_match] if distances[best_match] < 0.45 else "Unknown"
                    else:
                        name = "Unknown"
                    
                    top, right, bottom, left = face_location
                    scaled_location = (top * 4, right * 4, bottom * 4, left * 4)
                    temp_locations.append(scaled_location)
                    temp_names.append(name)
                    
                    detected_name = name
                    student_details = student_info.get(name, {})
                    
                    if name not in liveness_states:
                        liveness_states[name] = False
                    if name in student_info and liveness_states.get(name, False):
                        if name not in attendance:
                            attendance[name] = datetime.now().strftime("%H:%M:%S")
                            last_scanned_student = name
                
                with processing_lock:
                    recognition_results.update({
                        "locations": temp_locations,
                        "names": temp_names,
                        "last_update": time.time()
                    })
                    
            except Exception as e:
                print(f"[ERROR] Background recognition failed: {e}")
        
        time.sleep(0.1)

def generate_frames():
    global processing_frame, recognition_results, recognition_thread, current_brightness_level
    
    if recognition_thread is None or not recognition_thread.is_alive():
        recognition_thread = threading.Thread(target=background_face_recognition, daemon=True)
        recognition_thread.start()
    
    # Improved camera initialization with multiple fallback methods
    cap = None
    
    # Method 1: Try with DirectShow (Windows)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if ret:
                print("[OK] Camera opened successfully with DirectShow")
            else:
                cap.release()
                cap = None
        else:
            cap = None
    except Exception as e:
        cap = None
    
    # Method 2: Try without DirectShow
    if cap is None:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret:
                    print("[OK] Camera opened successfully without DirectShow")
                else:
                    cap.release()
                    cap = None
            else:
                cap = None
        except Exception as e:
            cap = None
    
    # Method 3: Try different camera indices
    if cap is None:
        for i in range(4):  # Try camera indices 0, 1, 2, 3
            try:
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = test_cap.read()
                    if ret:
                        cap = test_cap
                        print(f"[OK] Camera opened successfully with index {i}")
                        break
                    else:
                        test_cap.release()
                else:
                    test_cap.release()
            except Exception as e:
                continue
    
    # Method 4: Try with different backends
    if cap is None:
        backends = [cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_GSTREAMER]
        backend_names = ["MSMF", "V4L2", "GSTREAMER"]
        
        for backend, name in zip(backends, backend_names):
            try:
                test_cap = cv2.VideoCapture(0, backend)
                if test_cap.isOpened():
                    ret, test_frame = test_cap.read()
                    if ret:
                        cap = test_cap
                        print(f"[OK] Camera opened successfully with {name} backend")
                        break
                    else:
                        test_cap.release()
                else:
                    test_cap.release()
            except:
                continue
    
    # If still no camera, create placeholder
    if cap is None or not cap.isOpened():
        print("[ERROR] Cannot open any camera")
        
        placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        while True:
            cv2.putText(placeholder_frame, "No Camera Available", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(placeholder_frame, "Check camera permissions", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(placeholder_frame, "Close other camera apps", (90, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(placeholder_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (200, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', placeholder_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
    
    # Set camera properties with error handling
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    except Exception as e:
        print(f"[WARN] Could not set camera properties: {e}")
    
    frame_count = 0
    consecutive_failures = 0
    max_failures = 10
    
    print("[OK] Camera initialized successfully, streaming video...")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures:
                    print("[ERROR] Too many consecutive frame read failures")
                    cap.release()
                    # Try to reinitialize camera
                    time.sleep(2)
                    cap = cv2.VideoCapture(0)
                    consecutive_failures = 0
                    if not cap.isOpened():
                        print("[ERROR] Could not reinitialize camera")
                        break
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            
            frame = cv2.flip(frame, 1)
            
            # Get current face detection results
            with processing_lock:
                current_locations = recognition_results["locations"].copy()
                current_names = recognition_results["names"].copy()
                last_update = recognition_results["last_update"]
            
            # Clear old results
            if time.time() - last_update > 2.0:
                current_locations, current_names = [], []
            
            # Check brightness and adjust if needed
            has_faces = len(current_locations) > 0
            if has_faces:
                brightness_level = brightness_controller.calculate_face_brightness(frame, current_locations)
            else:
                brightness_level = brightness_controller.calculate_frame_brightness(frame)
            
            # Determine if brightness adjustment is needed
            should_adjust, adjustment = brightness_controller.should_adjust_brightness(brightness_level, has_faces)
            
            # Apply brightness adjustment
            frame = brightness_controller.adjust_brightness(frame, adjustment if should_adjust else None)
            current_brightness_level = brightness_controller.current_brightness
            
            # Convert to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            
            # Send frame for recognition processing every 3rd frame
            if frame_count % 3 == 0:
                with processing_lock:
                    processing_frame = rgb_frame.copy()
            
            # Blink detection
            blink_detected = False
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    ih, iw, _ = frame.shape
                    points = np.array([[lm.x * iw, lm.y * ih] for lm in landmarks.landmark])
                    left_ear = eye_aspect_ratio(points, LEFT_EYE)
                    right_ear = eye_aspect_ratio(points, RIGHT_EYE)
                    ear = (left_ear + right_ear) / 2.0
                    if ear < BLINK_THRESHOLD:
                        blink_detected = True
                        with processing_lock:
                            for name in recognition_results["names"]:
                                if name in liveness_states:
                                    liveness_states[name] = True
            
            # Draw face rectangles and names
            for i, (top, right, bottom, left) in enumerate(current_locations):
                if i < len(current_names):
                    name = current_names[i]
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    top, bottom = max(0, min(int(top), frame.shape[0]-1)), max(0, min(int(bottom), frame.shape[0]-1))
                    left, right = max(0, min(int(left), frame.shape[1]-1)), max(0, min(int(right), frame.shape[1]-1))
                    
                    if right > left and bottom > top:
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, name, (left, max(top - 10, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add basic frame info
            cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add brightness status indicator
            brightness_color = (0, 255, 0) if brightness_level >= DARK_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, "DARK" if brightness_level < DARK_THRESHOLD else "OK", 
                       (frame.shape[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, brightness_color, 2)
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if buffer is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.02)
            
        except Exception as e:
            print(f"[ERROR] Error in video generation loop: {e}")
            time.sleep(0.1)
            continue
    
    # Cleanup
    if cap is not None:
        cap.release()

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# -------------------------------
# JSON endpoints
# -------------------------------
@app.get("/status")
def get_status():
    return JSONResponse({
        "detected_name": detected_name,
        "student_details": student_details,
        "liveness": liveness_states.get(detected_name, False),
        "last_scanned": last_scanned_student,
        "brightness_level": current_brightness_level,
        "auto_brightness_active": current_brightness_level != 0
    })

@app.get("/attendance")
def get_attendance():
    return JSONResponse({
        "attendance": attendance,
        "total_present": len(attendance)
    })

@app.get("/brightness")
def get_brightness_info():
    return JSONResponse({
        "current_adjustment": current_brightness_level,
        "dark_threshold": DARK_THRESHOLD,
        "auto_adjustment_active": current_brightness_level != 0,
        "brightness_range": {"min": MIN_BRIGHTNESS, "max": MAX_BRIGHTNESS}
    })

# -------------------------------
# Run the server
# -------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Face Attendance System Server Starting...")
    print("Enhanced with Auto Brightness Adjustment")
    print("=" * 50)
    print("API Docs: http://localhost:8000/docs")
    print("Video Stream: http://localhost:8000/video")
    print("Status: http://localhost:8000/status")
    print("Attendance: http://localhost:8000/attendance")
    print("Brightness Info: http://localhost:8000/brightness")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")