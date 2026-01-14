import os
import glob
import pickle
import platform
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import jwt

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename

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
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))

# Separate folder for student face training data only
FACE_DATA_DIR = os.path.join(PROJECT_ROOT, "StudentFaceData")
ENCODINGS_DIR = os.path.join(BACKEND_ROOT, "data", "encodings")
LOGS_DIR = os.path.join(BACKEND_ROOT, "logs")

os.makedirs(ENCODINGS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ============================================================================
# STORAGE FUNCTIONS - For face training
# ============================================================================

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def save_image_to_storage(student_id, image_data):
    """
    Save image locally for face training.

    Args:
        student_id: Unique student identifier
        image_data: Raw image bytes

    Returns:
        str: Path where image was saved
    """
    student_folder = os.path.join(FACE_DATA_DIR, str(student_id))
    os.makedirs(student_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(student_folder, f"{timestamp}.jpg")

    with open(image_path, "wb") as f:
        f.write(image_data)

    return image_path

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_encode_face(image_path):
    """
    Detect face in image and generate encoding.

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (success: bool, encoding: np.array or error_message: str)
    """
    try:
        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Detect face locations
        face_locations = face_recognition.face_locations(image, model="hog")

        if len(face_locations) == 0:
            return False, "No face detected in the image"

        if len(face_locations) > 1:
            return False, f"Multiple faces detected ({len(face_locations)}). Please ensure only one face is visible"

        # Generate face encoding
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if len(face_encodings) == 0:
            return False, "Could not generate face encoding"

        return True, face_encodings[0]

    except Exception as e:
        return False, f"Error processing image: {str(e)}"

# ============================================================================
# END OF STORAGE FUNCTIONS
# ============================================================================

# Password hashing
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# Security
security = HTTPBearer()

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

known_face_encodings, known_face_names, known_face_ids = [], [], []
encodings_lock = threading.Lock()

# Attendance tracking
attendance_records = []
attendance_lock = threading.Lock()

# Recently recognized student tracking
recently_recognized = None
recently_recognized_lock = threading.Lock()

# Database
client: AsyncIOMotorClient = None
db = None

async def connect_to_mongodb():
    """Connect to MongoDB."""
    global client, db
    try:
        # Replace with your MongoDB connection string
        client = AsyncIOMotorClient("mongodb://localhost:27017")
        db = client["InterACTS"]
        logger.info("✅ Connected to MongoDB")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        raise

async def close_mongodb_connection():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        logger.info("✅ MongoDB connection closed")

# Authentication utilities
def verify_password(plain_password, hashed_password):
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password."""
    return pwd_context.hash(password)

def create_access_token(data: dict):
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def save_attendance_to_db(record: dict):
    """Save attendance record to database."""
    try:
        attendance_collection = db.attendance
        result = await attendance_collection.insert_one(record)
        logger.info(f"✅ Attendance saved to DB: {record['name']}")
    except Exception as e:
        logger.error(f"❌ Failed to save attendance to DB: {e}")

# Pydantic Models
class UserBase(BaseModel):
    username: str
    email: str
    role: str  # admin, teacher, student

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: str = Field(alias="_id")

class StudentBase(BaseModel):
    student_id: str
    first_name: str
    last_name: str
    email: str
    course: str
    year: str
    face_encodings: Optional[List[List[float]]] = None
    hashed_password: Optional[str] = None

class StudentCreate(StudentBase):
    pass

class Student(StudentBase):
    id: str = Field(alias="_id")

class TeacherBase(BaseModel):
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    email: str
    department: str
    teacher_id: Optional[str] = None  # Auto-generated
    hashed_password: Optional[str] = None

class TeacherCreate(TeacherBase):
    pass

class Teacher(TeacherBase):
    id: str = Field(alias="_id")

class ClassBase(BaseModel):
    class_code: str
    class_name: str
    teacher_id: str
    schedule: str  # e.g., "MWF 9:00-10:00"
    room: str
    courses: List[str] = []  # List of courses this class covers

class ClassCreate(ClassBase):
    pass

class Class(ClassBase):
    id: str = Field(alias="_id")
    enrolled_students: List[str] = []

class AttendanceBase(BaseModel):
    student_id: str
    class_id: str
    date: str
    check_in_time: Optional[str] = None
    check_out_time: Optional[str] = None
    status: str  # present, late, absent

class AttendanceCreate(AttendanceBase):
    pass

class Attendance(AttendanceBase):
    id: str = Field(alias="_id")

class EventBase(BaseModel):
    name: str
    description: str
    date: str
    location: str
    price: Optional[float] = 0.0

class EventCreate(EventBase):
    pass

class Event(EventBase):
    id: str = Field(alias="_id")

class ReceiptBase(BaseModel):
    student_id: str
    event_id: str
    transaction_id: str
    receipt_image: str  # Base64 encoded image
    status: str  # pending, verified, rejected
    submitted_at: str
    verified_at: Optional[str] = None
    verified_by: Optional[str] = None

class ReceiptCreate(BaseModel):
    student_id: str
    event_id: str
    transaction_id: str = Field(..., pattern=r'^\d{6}$', description="Transaction ID must be exactly 6 digits")
    receipt_image: str

class Receipt(ReceiptBase):
    id: str = Field(alias="_id")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

FACE_MATCH_THRESHOLD = 0.5  # Lower threshold for better recognition
PROCESS_EVERY_N_FRAMES = 2
JPEG_QUALITY = 70
FRAME_SCALE = 0.5  # Scale down for faster processing

# App
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Database events
@app.on_event("startup")
async def startup_event():
    await connect_to_mongodb()
    await load_faces_from_db()
    logger.info("🚀 Server started - loading faces from database")

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongodb_connection()


# === FACE LOADING === #
async def load_faces_from_db():
    global known_face_encodings, known_face_names, known_face_ids
    logger.info("🔄 Loading saved face encodings from database...")

    with encodings_lock:
        known_face_encodings.clear()
        known_face_names.clear()
        known_face_ids.clear()

        try:
            students_collection = db.students
            async for student in students_collection.find({"face_encodings": {"$exists": True, "$ne": []}}):
                student_id = student["student_id"]
                first_name = student.get("first_name", "")
                last_name = student.get("last_name", "")
                full_name = f"{first_name} {last_name}".strip()
                enc_list = student.get("face_encodings", [])
                loaded_count = 0
                for enc in enc_list:
                    if isinstance(enc, list) and len(enc) == 128:
                        known_face_encodings.append(np.array(enc))
                        known_face_names.append(full_name)
                        known_face_ids.append(student_id)
                        loaded_count += 1
                    else:
                        logger.warning(f"Skipping invalid encoding for {student_id}: {type(enc)} len={len(enc) if isinstance(enc, list) else 'N/A'}")
                logger.info(f"✅ Loaded {loaded_count}/{len(enc_list)} encodings for: {full_name} ({student_id})")
        except Exception as e:
            logger.error(f"❌ Failed to load face encodings from database: {e}")

    logger.info(f"✅ Loaded {len(known_face_encodings)} known face encodings total from database")

def load_faces_from_disk():
    """Fallback function to load from disk if needed."""
    global known_face_encodings, known_face_names, known_face_ids
    logger.info("🔄 Loading saved face encodings from disk (fallback)...")
    logger.info(f"📂 Looking for encodings in: {ENCODINGS_DIR}")

    encs = glob.glob(os.path.join(ENCODINGS_DIR, "*.pkl"))
    logger.info(f"📄 Found {len(encs)} .pkl files")

    with encodings_lock:
        known_face_encodings.clear()
        known_face_names.clear()
        known_face_ids.clear()

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
                        known_face_names.append(name)  # For disk fallback, name is the filename (student_id)
                        known_face_ids.append(name)   # For disk fallback, id is also the filename
                        loaded_count += 1
                    else:
                        logger.warning(f"Skipping invalid encoding shape: {getattr(enc, 'shape', 'unknown')} for {name}")
                logger.info(f"✅ Loaded {loaded_count}/{len(enc_list)} encodings for: {name}")
            except Exception as e:
                logger.warning(f"❌ Failed to load {file}: {e}")

    logger.info(f"✅ Loaded {len(known_face_encodings)} known face encodings total")





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
                    student_id = "Unknown"
                    if len(known_face_encodings) > 0:
                        dists = face_recognition.face_distance(known_face_encodings, enc)
                        if len(dists) > 0:
                            idx = np.argmin(dists)
                            if dists[idx] <= FACE_MATCH_THRESHOLD:
                                name = known_face_names[idx]
                                student_id = known_face_ids[idx]

                                # Record attendance in memory and database
                                current_time = datetime.now().strftime('%H:%M:%S')

                                with attendance_lock:
                                    # Check if student already recorded today
                                    today_records = [r for r in attendance_records if r['student_id'] == student_id and r['date'] == datetime.now().strftime('%Y-%m-%d')]
                                    if not today_records:
                                        record = {
                                            'name': name,
                                            'student_id': student_id,
                                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            'date': datetime.now().strftime('%Y-%m-%d'),
                                            'time': current_time,
                                            'course': 'Unknown',
                                            'year': 'Unknown',
                                            'status': 'present'
                                        }
                                        attendance_records.append(record)

                                        # Save to database asynchronously
                                        import asyncio
                                        asyncio.run_coroutine_threadsafe(
                                            save_attendance_to_db(record),
                                            asyncio.get_event_loop()
                                        )

                                # Update recently recognized student
                                with recently_recognized_lock:
                                    global recently_recognized
                                    recently_recognized = {
                                        'name': name,
                                        'student_id': student_id,
                                        'time': current_time,
                                        'date': datetime.now().strftime('%Y-%m-%d'),
                                        'course': 'Unknown',
                                        'year': 'Unknown'
                                    }

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
    last_frame = None
    while recognition_running:
        with latest_frame_cond:
            latest_frame_cond.wait(timeout=0.1)  # Shorter timeout for more responsive streaming
            if latest_frame:
                last_frame = latest_frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame + b"\r\n"
            elif last_frame:
                # Send the last known frame to keep the stream alive
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last_frame + b"\r\n"


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


@app.get("/frame")
def get_frame():
    """Get the latest frame as a simple JPEG response."""
    if latest_frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=404)

    from fastapi.responses import Response
    return Response(content=latest_frame, media_type="image/jpeg")


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
async def reload_faces():
    await load_faces_from_db()
    return {"status": "reloaded from database"}

@app.post("/set-mode")
def set_mode(mode_data: dict):
    """Set the current recognition mode."""
    global current_mode, current_event_id
    mode = mode_data.get("mode")
    event_id = mode_data.get("event_id")

    if mode not in ["class", "events"]:
        raise HTTPException(status_code=400, detail="Invalid mode")

    current_mode = mode
    current_event_id = event_id
    return {"message": f"Mode set to {mode}", "event_id": event_id}




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

@app.get("/recently-recognized")
def get_recently_recognized():
    """Get the most recently recognized student."""
    with recently_recognized_lock:
        return {"recently_recognized": recently_recognized}

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


# === AUTHENTICATION ROUTES === #
@app.post("/auth/login", response_model=Token)
async def login(user_credentials: dict):
    """Login user and return access token."""
    username = user_credentials["username"]
    password = user_credentials["password"]

    # Check if username contains '@' - treat as email for admin/teacher
    if '@' in username:
        # Check users collection by email (for admin)
        user = await db.users.find_one({"email": username})
        if user and verify_password(password, user["hashed_password"]):
            access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
            return {"access_token": access_token, "token_type": "bearer"}

        # Check teachers collection by email (for teachers)
        teacher = await db.teachers.find_one({"email": username})
        if teacher and verify_password(password, teacher.get("hashed_password", "")):
            access_token = create_access_token(data={"sub": username, "role": "teacher"})
            first_name = teacher.get('first_name', '')
            last_name = teacher.get('last_name', '')
            logger.info(f"✅ Teacher login successful for {username}: first_name='{first_name}', last_name='{last_name}'")
            return {"access_token": access_token, "token_type": "bearer", "first_name": first_name, "last_name": last_name, "user_id": teacher.get("teacher_id")}

        # Check teachers collection by teacher_id (for teachers logging in with teacher_id)
        if not '@' in username and username.isdigit() and len(username) == 6:
            teacher = await db.teachers.find_one({"teacher_id": username})
            if teacher and verify_password(password, teacher.get("hashed_password", "")):
                access_token = create_access_token(data={"sub": username, "role": "teacher"})
                first_name = teacher.get('first_name', '')
                last_name = teacher.get('last_name', '')
                logger.info(f"✅ Teacher login successful for {username}: first_name='{first_name}', last_name='{last_name}'")
                return {"access_token": access_token, "token_type": "bearer", "first_name": first_name, "last_name": last_name, "user_id": teacher.get("teacher_id")}
    else:
        # Check students collection by student_id
        student = await db.students.find_one({"student_id": username})
        if student and verify_password(password, student.get("hashed_password", "")):
            access_token = create_access_token(data={"sub": username, "role": "student"})
            full_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
            return {"access_token": access_token, "token_type": "bearer", "full_name": full_name}

        # Check teachers collection by username (for teachers with usernames like 'teacher1')
        teacher = await db.teachers.find_one({"username": username})
        if teacher and verify_password(password, teacher.get("hashed_password", "")):
            access_token = create_access_token(data={"sub": username, "role": "teacher"})
            first_name = teacher.get('first_name', '')
            last_name = teacher.get('last_name', '')
            return {"access_token": access_token, "token_type": "bearer", "first_name": first_name, "last_name": last_name}

        # Check users collection by username (for admin with username like 'admin')
        user = await db.users.find_one({"username": username})
        if user and verify_password(password, user["hashed_password"]):
            access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
            return {"access_token": access_token, "token_type": "bearer"}

    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/auth/register")
async def register_user(user: UserCreate):
    """Register a new user."""
    existing_user = await db.users.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    user_dict.pop("password")

    result = await db.users.insert_one(user_dict)
    return {"message": "User created successfully", "user_id": str(result.inserted_id)}


# === STUDENT MANAGEMENT === #
@app.post("/students")
async def create_student(student: StudentCreate):
    """Create a new student."""
    existing_student = await db.students.find_one({"student_id": student.student_id})
    if existing_student:
        raise HTTPException(status_code=400, detail="Student ID already exists")

    # Set initial password as student_id (hashed)
    student_dict = student.dict()
    student_dict["hashed_password"] = get_password_hash(student.student_id)

    result = await db.students.insert_one(student_dict)
    return {"message": "Student created successfully", "student_id": str(result.inserted_id)}

@app.get("/students")
async def get_students():
    """Get all students."""
    students = []
    async for student in db.students.find():
        student["_id"] = str(student["_id"])
        students.append(student)
    return {"students": students}

@app.get("/students/{student_id}")
async def get_student(student_id: str):
    """Get student by ID."""
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student["_id"] = str(student["_id"])
    return student

@app.put("/students/{student_id}")
async def update_student(student_id: str, student_update: dict):
    """Update student information."""
    result = await db.students.update_one(
        {"student_id": student_id},
        {"$set": student_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Student updated successfully"}

@app.post("/students/{student_id}/face-encodings")
async def save_face_encodings(student_id: str, image: UploadFile = File(...)):
    """Process uploaded image and save face encodings for a student."""
    # Check if student exists
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Validate file
    if not image.filename or not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG, JPEG, and PNG are allowed.")

    try:
        # Read image data
        image_data = await image.read()

        # Save image to storage
        image_path = save_image_to_storage(student_id, image_data)

        # Detect and encode face
        success, result = detect_and_encode_face(image_path)

        if not success:
            # Clean up saved image if face detection failed
            if os.path.exists(image_path):
                os.remove(image_path)
            raise HTTPException(status_code=400, detail=result)

        # Convert numpy array to list for MongoDB storage
        encoding_list = result.tolist()

        # Get existing encodings or initialize empty list
        existing_encodings = student.get("face_encodings", [])
        if not isinstance(existing_encodings, list):
            existing_encodings = []

        # Add new encoding
        existing_encodings.append(encoding_list)

        # Update student with new encodings
        result = await db.students.update_one(
            {"student_id": student_id},
            {"$set": {"face_encodings": existing_encodings}}
        )

        logger.info(f"✅ Face encoding saved for student {student_id}")

        # Reload faces in recognition system
        await load_faces_from_db()

        return {"message": "Face encoding saved successfully", "total_encodings": len(existing_encodings)}

    except Exception as e:
        logger.error(f"❌ Error processing face encoding for {student_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.delete("/students/{student_id}")
async def delete_student(student_id: str):
    """Delete a student."""
    result = await db.students.delete_one({"student_id": student_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Student deleted successfully"}


# === TEACHER MANAGEMENT === #
@app.post("/teachers")
async def create_teacher(teacher: TeacherCreate):
    """Create a new teacher."""
    # Auto-generate 6-digit teacher ID
    import random
    while True:
        teacher_id = f"{random.randint(100000, 999999)}"
        existing_teacher = await db.teachers.find_one({"teacher_id": teacher_id})
        if not existing_teacher:
            break

    # Set password as teacher_id (hashed)
    teacher_dict = teacher.dict()
    teacher_dict["teacher_id"] = teacher_id
    teacher_dict["hashed_password"] = get_password_hash(teacher_id)

    result = await db.teachers.insert_one(teacher_dict)
    return {"message": "Teacher created successfully", "teacher_id": teacher_id}

@app.get("/teachers")
async def get_teachers():
    """Get all teachers."""
    teachers = []
    async for teacher in db.teachers.find():
        teacher["_id"] = str(teacher["_id"])
        teachers.append(teacher)
    return {"teachers": teachers}

@app.get("/teachers/{teacher_id}")
async def get_teacher(teacher_id: str):
    """Get teacher by ID or email."""
    teacher = await db.teachers.find_one({"$or": [{"teacher_id": teacher_id}, {"email": teacher_id}]})
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found")
    teacher["_id"] = str(teacher["_id"])
    return teacher

@app.put("/teachers/{teacher_id}")
async def update_teacher(teacher_id: str, teacher_update: dict):
    """Update teacher information."""
    result = await db.teachers.update_one(
        {"teacher_id": teacher_id},
        {"$set": teacher_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Teacher not found")
    return {"message": "Teacher updated successfully"}

@app.delete("/teachers/{teacher_id}")
async def delete_teacher(teacher_id: str):
    """Delete a teacher."""
    result = await db.teachers.delete_one({"teacher_id": teacher_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Teacher not found")
    return {"message": "Teacher deleted successfully"}


# === CLASS MANAGEMENT === #
@app.post("/classes")
async def create_class(class_data: ClassCreate):
    """Create a new class."""
    existing_class = await db.classes.find_one({"class_code": class_data.class_code})
    if existing_class:
        raise HTTPException(status_code=400, detail="Class code already exists")

    result = await db.classes.insert_one(class_data.dict())
    return {"message": "Class created successfully", "class_id": str(result.inserted_id)}

@app.get("/courses")
async def get_courses():
    """Get all available courses and strands from database."""
    courses_collection = db.courses

    # Get all courses from database
    courses = []
    async for course in courses_collection.find().sort("level", 1).sort("code", 1):
        courses.append({
            "code": course["code"],
            "name": course["name"],
            "level": course["level"]
        })

    # Also include any additional courses from existing students (for backward compatibility)
    async for student in db.students.find():
        if student.get("course") and not any(c["code"] == student["course"] for c in courses):
            courses.append({
                "code": student["course"],
                "name": student["course"],  # Use code as name for legacy courses
                "level": "unknown"
            })

    return {"courses": courses}

@app.get("/classes")
async def get_classes():
    """Get all classes."""
    classes = []
    async for class_doc in db.classes.find():
        class_doc["_id"] = str(class_doc["_id"])
        classes.append(class_doc)
    return {"classes": classes}

@app.get("/classes/teacher/{teacher_id}")
async def get_classes_by_teacher(teacher_id: str):
    """Get all classes for a specific teacher."""
    # First, try to find teacher by teacher_id
    teacher = await db.teachers.find_one({"teacher_id": teacher_id})
    if not teacher:
        # If not found, try to find by email
        teacher = await db.teachers.find_one({"email": teacher_id})
        if not teacher:
            # If still not found, return empty
            return {"classes": []}
    actual_teacher_id = teacher.get("teacher_id")
    classes = []
    async for class_doc in db.classes.find({"teacher_id": actual_teacher_id}):
        class_doc["_id"] = str(class_doc["_id"])
        classes.append(class_doc)
    return {"classes": classes}

@app.get("/classes/{class_id}")
async def get_class(class_id: str):
    """Get class by ID."""
    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=404, detail="Class not found")
    class_doc["_id"] = str(class_doc["_id"])
    return class_doc

@app.put("/classes/{class_id}")
async def update_class(class_id: str, class_update: dict):
    """Update class information."""
    result = await db.classes.update_one(
        {"_id": ObjectId(class_id)},
        {"$set": class_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Class not found")
    return {"message": "Class updated successfully"}

@app.delete("/classes/{class_id}")
async def delete_class(class_id: str):
    """Delete a class."""
    result = await db.classes.delete_one({"_id": ObjectId(class_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Class not found")
    return {"message": "Class deleted successfully"}

@app.post("/classes/{class_id}/enroll")
async def enroll_student(class_id: str, student_id: str):
    """Enroll a student in a class."""
    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=404, detail="Class not found")

    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    if student_id not in class_doc.get("enrolled_students", []):
        await db.classes.update_one(
            {"_id": ObjectId(class_id)},
            {"$push": {"enrolled_students": student_id}}
        )

    return {"message": "Student enrolled successfully"}


# === ATTENDANCE MANAGEMENT === #
@app.post("/attendance/check-in")
async def check_in(attendance_data: dict):
    """Manual check-in for attendance."""
    student_id = attendance_data["student_id"]
    class_id = attendance_data["class_id"]

    # Check if student exists
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Check if class exists
    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=404, detail="Class not found")

    current_time = datetime.now()
    attendance_record = {
        "student_id": student_id,
        "class_id": class_id,
        "date": current_time.strftime('%Y-%m-%d'),
        "check_in_time": current_time.strftime('%H:%M:%S'),
        "status": "present",
        "timestamp": current_time.strftime('%Y-%m-%d %H:%M:%S')
    }

    result = await db.attendance.insert_one(attendance_record)
    return {"message": "Check-in recorded", "attendance_id": str(result.inserted_id)}

@app.post("/attendance/check-out")
async def check_out(attendance_data: dict):
    """Manual check-out for attendance."""
    student_id = attendance_data["student_id"]
    class_id = attendance_data["class_id"]

    current_time = datetime.now()
    result = await db.attendance.update_one(
        {
            "student_id": student_id,
            "class_id": class_id,
            "date": current_time.strftime('%Y-%m-%d'),
            "check_out_time": {"$exists": False}
        },
        {"$set": {"check_out_time": current_time.strftime('%H:%M:%S')}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="No active check-in found")

    return {"message": "Check-out recorded"}


# === EVENT MANAGEMENT === #
@app.post("/events")
async def create_event(event: EventCreate):
    """Create a new event."""
    result = await db.events.insert_one(event.dict())
    return {"message": "Event created successfully", "event_id": str(result.inserted_id)}

@app.get("/events")
async def get_events():
    """Get all events."""
    events = []
    async for event in db.events.find():
        event["_id"] = str(event["_id"])
        events.append(event)
    return {"events": events}

@app.get("/events/{event_id}")
async def get_event(event_id: str):
    """Get event by ID."""
    event = await db.events.find_one({"_id": ObjectId(event_id)})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    event["_id"] = str(event["_id"])
    return event

@app.put("/events/{event_id}")
async def update_event(event_id: str, event_update: dict):
    """Update event information."""
    result = await db.events.update_one(
        {"_id": ObjectId(event_id)},
        {"$set": event_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event updated successfully"}

@app.delete("/events/{event_id}")
async def delete_event(event_id: str):
    """Delete an event."""
    result = await db.events.delete_one({"_id": ObjectId(event_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event deleted successfully"}

# === RECEIPT MANAGEMENT === #
@app.post("/receipts")
async def submit_receipt(receipt: ReceiptCreate):
    """Submit a receipt for verification."""
    # Check if student exists
    student = await db.students.find_one({"student_id": receipt.student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Check if event exists
    event = await db.events.find_one({"_id": ObjectId(receipt.event_id)})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Check if receipt already exists for this student and event
    existing_receipt = await db.receipts.find_one({
        "student_id": receipt.student_id,
        "event_id": receipt.event_id
    })
    if existing_receipt:
        raise HTTPException(status_code=400, detail="Receipt already submitted for this event")

    receipt_dict = receipt.dict()
    receipt_dict["status"] = "pending"
    receipt_dict["submitted_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    result = await db.receipts.insert_one(receipt_dict)
    return {"message": "Receipt submitted successfully", "receipt_id": str(result.inserted_id)}

@app.get("/receipts")
async def get_receipts(status: str = None):
    """Get all receipts, optionally filtered by status."""
    query = {}
    if status:
        query["status"] = status

    receipts = []
    async for receipt in db.receipts.find(query).sort("submitted_at", -1):
        receipt["_id"] = str(receipt["_id"])
        receipts.append(receipt)
    return {"receipts": receipts}

@app.get("/receipts/student/{student_id}")
async def get_student_receipts(student_id: str):
    """Get receipts for a specific student."""
    receipts = []
    async for receipt in db.receipts.find({"student_id": student_id}).sort("submitted_at", -1):
        receipt["_id"] = str(receipt["_id"])
        receipts.append(receipt)
    return {"receipts": receipts}

@app.put("/receipts/{receipt_id}/verify")
async def verify_receipt(receipt_id: str, verification_data: dict):
    """Verify or reject a receipt (admin only)."""
    status = verification_data.get("status")  # "verified" or "rejected"
    verified_by = verification_data.get("verified_by")

    if status not in ["verified", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    result = await db.receipts.update_one(
        {"_id": ObjectId(receipt_id)},
        {"$set": {
            "status": status,
            "verified_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "verified_by": verified_by
        }}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Receipt not found")

    return {"message": f"Receipt {status} successfully"}

@app.get("/students/{student_id}/payment-status/{event_id}")
async def get_payment_status(student_id: str, event_id: str):
    """Get payment status for a student for a specific event."""
    receipt = await db.receipts.find_one({
        "student_id": student_id,
        "event_id": event_id,
        "status": "verified"
    })

    if receipt:
        event = await db.events.find_one({"_id": ObjectId(event_id)})
        return {
            "paid": True,
            "event_name": event["name"] if event else "Unknown Event",
            "verified_at": receipt["verified_at"]
        }
    else:
        return {"paid": False}

# === ANALYTICS ROUTES === #
@app.get("/analytics/attendance-summary")
async def get_attendance_summary(date_from: str = None, date_to: str = None):
    """Get attendance summary statistics."""
    query = {}
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}

    pipeline = [
        {"$match": query},
        {"$group": {
            "_id": {
                "date": "$date",
                "class_id": "$class_id"
            },
            "present_count": {"$sum": 1},
            "total_students": {"$addToSet": "$student_id"}
        }},
        {"$project": {
            "date": "$_id.date",
            "class_id": "$_id.class_id",
            "present_count": 1,
            "enrolled_count": {"$size": "$total_students"}
        }}
    ]

    results = []
    async for doc in db.attendance.aggregate(pipeline):
        results.append(doc)

    return {"summary": results}

@app.get("/analytics/student/{student_id}")
async def get_student_attendance(student_id: str, date_from: str = None, date_to: str = None):
    """Get attendance records for a specific student."""
    query = {"student_id": student_id}
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}

    records = []
    async for record in db.attendance.find(query).sort("date", -1):
        record["_id"] = str(record["_id"])
        records.append(record)

    return {"student_id": student_id, "attendance": records}

@app.get("/analytics/class/{class_id}")
async def get_class_attendance(class_id: str, date_from: str = None, date_to: str = None):
    """Get attendance records for a specific class."""
    query = {"class_id": class_id}
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}

    records = []
    async for record in db.attendance.find(query).sort("date", -1):
        record["_id"] = str(record["_id"])
        records.append(record)

    return {"class_id": class_id, "attendance": records}


# === DATABASE VIEW ROUTES === #
@app.get("/db/collections")
async def get_db_collections():
    """Get list of all collections in the database."""
    try:
        collections = await db.list_collection_names()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"❌ Failed to list collections: {e}")
        return JSONResponse({"error": "Failed to list collections"}, status_code=500)

@app.get("/db/{collection}")
async def get_collection_data(collection: str, limit: int = 10):
    """Get data from a specific collection."""
    try:
        collection_obj = db[collection]
        documents = []
        async for doc in collection_obj.find().limit(limit):
            doc["_id"] = str(doc["_id"])
            documents.append(doc)
        return {"collection": collection, "documents": documents, "limit": limit}
    except Exception as e:
        logger.error(f"❌ Failed to fetch data from {collection}: {e}")
        return JSONResponse({"error": f"Failed to fetch data from {collection}"}, status_code=500)


# === STARTUP === #
# Note: Database startup is handled above, this is for backward compatibility
# This event is now redundant but kept for compatibility


# === MAIN ENTRY === #
if __name__ == "__main__":
    import uvicorn
    logger.info(f"✅ Server listening at http://{HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)