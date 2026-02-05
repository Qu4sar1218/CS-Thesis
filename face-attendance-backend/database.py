import os
import random
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import jwt

# Config
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

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Globals
known_face_encodings, known_face_names, known_face_ids, known_face_courses, known_face_years = [], [], [], [], []
encodings_lock = threading.Lock()

current_mode = "class"  # Default to class mode
current_event_id = None
current_class_id = None

# Logging
log_file = os.path.join(LOGS_DIR, f"server_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger("face-attendance")

# Password hashing
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# Name formatting helper
def format_student_name(first_name, middle_name, last_name):
    """Format student name as: Firstname Lastname M."""
    if not first_name or not last_name:
        return f"{first_name or ''} {last_name or ''}".strip()

    # Capitalize each word in first_name and last_name
    def capitalize(s):
        return ' '.join(word.capitalize() for word in s.split())

    capitalized_first = capitalize(first_name)
    capitalized_last = capitalize(last_name)

    last_parts = capitalized_last.split()
    if len(last_parts) > 1:
        # Handle multiple words in last name
        middle_initial = f" {middle_name[0].upper()}." if middle_name else ""
        return f"{capitalized_first} {last_parts[0]}{middle_initial} {' '.join(last_parts[1:])}".strip()
    else:
        # Standard format
        middle_initial = f" {middle_name[0].upper()}." if middle_name else ""
        return f"{capitalized_first}{middle_initial} {capitalized_last}".strip()

db = None

async def connect_to_mongodb():
    """Connect to MongoDB."""
    global client, db
    try:
        # Replace with your MongoDB connection string
        client = AsyncIOMotorClient("mongodb://localhost:27017")
        db = client["InterACTS"]
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
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

async def update_attendance_status(student_id: str, class_id: str, status: str):
    """Update attendance status for a student in a class."""
    try:
        current_date = datetime.now().strftime('%Y-%m-%d')
        attendance_collection = db.attendance

        # Update existing absent record to present
        result = await attendance_collection.update_one(
            {
                "student_id": student_id,
                "class_id": class_id,
                "date": current_date,
                "status": "absent"
            },
            {"$set": {"status": status}}
        )

        if result.modified_count > 0:
            logger.info(f"✅ Updated attendance status for {student_id} in class {class_id} to {status}")
        else:
            logger.warning(f"⚠️ No absent record found to update for {student_id} in class {class_id}")

    except Exception as e:
        logger.error(f"❌ Failed to update attendance status for {student_id}: {e}")

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
    middle_name: Optional[str] = None
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

# Security
security = HTTPBearer()

# Router
router = APIRouter()

# === AUTHENTICATION ROUTES === #
@router.post("/auth/login", response_model=Token)
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
            full_name = format_student_name(
                student.get('first_name', ''),
                student.get('middle_name', ''),
                student.get('last_name', '')
            )
            return {"access_token": access_token, "token_type": "bearer", "full_name": full_name, "course": student.get("course", ""), "year": student.get("year", ""), "user_id": student.get("student_id")}

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

@router.post("/auth/register")
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
