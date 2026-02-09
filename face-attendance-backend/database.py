import os
import random
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import jwt

from models import (
    UserBase, UserCreate, User,
    StudentBase, StudentCreate, Student,
    TeacherBase, TeacherCreate, Teacher,
    ClassBase, ClassCreate, Class,
    AttendanceBase, AttendanceCreate, Attendance,
    EventBase, EventCreate, Event,
    ReceiptBase, ReceiptCreate, Receipt,
    Token, TokenData
)
from utils import format_student_name, is_class_scheduled_today

# Create the router
router = APIRouter()

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

class DatabaseManager:
    """Database manager class for handling MongoDB connections."""

    def __init__(self):
        self.database = None
        self.client = None

    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient("mongodb://localhost:27017")
            self.database = self.client["InterACTS"]
            print("✅ Connected to MongoDB")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise

    async def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")

db_manager = DatabaseManager()

def get_database():
    """Get the database instance from the manager."""
    return db_manager.database

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

@router.get("/courses")
async def get_courses():
    """Get all available courses and strands from database."""
    db = get_database()

    # Get all courses from database
    courses = []
    async for course in db.courses.find().sort("level", 1).sort("code", 1):
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

# === TEACHER MANAGEMENT === #
@router.post("/teachers")
async def create_teacher(teacher: TeacherCreate):
    """Create a new teacher."""
    db = get_database()

    # Auto-generate 6-digit teacher ID
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

@router.get("/teachers")
async def get_teachers():
    """Get all teachers."""
    db = get_database()

    teachers = []
    async for teacher in db.teachers.find():
        teacher["_id"] = str(teacher["_id"])
        teachers.append(teacher)
    return {"teachers": teachers}

@router.get("/teachers/{teacher_id}")
async def get_teacher(teacher_id: str):
    """Get teacher by ID or email."""
    db = get_database()

    teacher = await db.teachers.find_one({"$or": [{"teacher_id": teacher_id}, {"email": teacher_id}]})
    if not teacher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Teacher not found")
    teacher["_id"] = str(teacher["_id"])
    return teacher

@router.put("/teachers/{teacher_id}")
async def update_teacher(teacher_id: str, teacher_update: dict):
    """Update teacher information."""
    db = get_database()

    result = await db.teachers.update_one(
        {"teacher_id": teacher_id},
        {"$set": teacher_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Teacher not found")
    return {"message": "Teacher updated successfully"}

@router.delete("/teachers/{teacher_id}")
async def delete_teacher(teacher_id: str):
    """Delete a teacher."""
    db = get_database()

    result = await db.teachers.delete_one({"teacher_id": teacher_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Teacher not found")
    return {"message": "Teacher deleted successfully"}

# === CLASS MANAGEMENT === #
@router.post("/classes")
async def create_class(class_data: ClassCreate):
    """Create a new class."""
    db = get_database()

    existing_class = await db.classes.find_one({"class_code": class_data.class_code})
    if existing_class:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Class code already exists")

    result = await db.classes.insert_one(class_data.dict())
    return {"message": "Class created successfully", "class_id": str(result.inserted_id)}

@router.get("/classes")
async def get_classes():
    """Get all classes."""
    db = get_database()

    classes = []
    async for class_doc in db.classes.find():
        class_doc["_id"] = str(class_doc["_id"])
        classes.append(class_doc)
    return {"classes": classes}

@router.get("/classes/teacher/{teacher_id}")
async def get_classes_by_teacher(teacher_id: str):
    """Get all classes for a specific teacher."""
    db = get_database()

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
        # Add accessibility flag based on schedule
        schedule = class_doc.get("schedule", "")
        class_doc["accessible"] = is_class_scheduled_today(schedule)
        classes.append(class_doc)
    return {"classes": classes}

@router.get("/classes/{class_id}")
async def get_class(class_id: str):
    """Get class by ID."""
    db = get_database()

    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")
    class_doc["_id"] = str(class_doc["_id"])
    return class_doc

@router.put("/classes/{class_id}")
async def update_class(class_id: str, class_update: dict):
    """Update class information."""
    db = get_database()

    result = await db.classes.update_one(
        {"_id": ObjectId(class_id)},
        {"$set": class_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")
    return {"message": "Class updated successfully"}

@router.delete("/classes/{class_id}")
async def delete_class(class_id: str):
    """Delete a class."""
    db = get_database()

    result = await db.classes.delete_one({"_id": ObjectId(class_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")
    return {"message": "Class deleted successfully"}

@router.post("/classes/{class_id}/enroll")
async def enroll_student(class_id: str, data: dict):
    """Enroll a student in a class."""
    db = get_database()

    student_id = data.get("student_id")
    if not student_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="student_id is required")

    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")

    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    if student_id not in class_doc.get("enrolled_students", []):
        await db.classes.update_one(
            {"_id": ObjectId(class_id)},
            {"$push": {"enrolled_students": student_id}}
        )

    return {"message": "Student enrolled successfully"}

# === ATTENDANCE MANAGEMENT === #
@router.post("/attendance/initialize-class/{class_id}")
async def initialize_class_attendance(class_id: str):
    """Initialize attendance records for all enrolled students in a class as absent for today."""
    db = get_database()

    # Check if class exists
    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")

    enrolled_students = class_doc.get("enrolled_students", [])
    if not enrolled_students:
        return {"message": "No enrolled students found for this class"}

    current_date = datetime.now().strftime('%Y-%m-%d')
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    attendance_collection = db.attendance
    initialized_count = 0

    for student_id in enrolled_students:
        # Check if student already has a record for today
        existing_record = await attendance_collection.find_one({
            "student_id": student_id,
            "class_id": class_id,
            "date": current_date
        })

        if not existing_record:
            # Create absent record
            attendance_record = {
                "student_id": student_id,
                "class_id": class_id,
                "date": current_date,
                "status": "absent",
                "timestamp": current_timestamp,
                "subject": class_doc.get("class_name", "Unknown Subject")
            }
            await attendance_collection.insert_one(attendance_record)
            initialized_count += 1

    return {"message": f"Initialized attendance for {initialized_count} students"}

@router.post("/attendance/check-in")
async def check_in(attendance_data: dict):
    """Manual check-in for attendance."""
    db = get_database()

    student_id = attendance_data["student_id"]
    class_id = attendance_data["class_id"]

    # Check if student exists
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    # Check if class exists
    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")

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

@router.post("/attendance/check-out")
async def check_out(attendance_data: dict):
    """Manual check-out for attendance."""
    db = get_database()

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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active check-in found")

    return {"message": "Check-out recorded"}

# === EVENT MANAGEMENT === #
@router.post("/events")
async def create_event(event: EventCreate):
    """Create a new event."""
    db = get_database()

    result = await db.events.insert_one(event.dict())
    return {"message": "Event created successfully", "event_id": str(result.inserted_id)}

@router.get("/events")
async def get_events():
    """Get all events."""
    db = get_database()

    events = []
    async for event in db.events.find():
        event["_id"] = str(event["_id"])
        events.append(event)
    return {"events": events}

@router.get("/events/{event_id}")
async def get_event(event_id: str):
    """Get event by ID."""
    db = get_database()

    event = await db.events.find_one({"_id": ObjectId(event_id)})
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    event["_id"] = str(event["_id"])
    return event

@router.put("/events/{event_id}")
async def update_event(event_id: str, event_update: dict):
    """Update event information."""
    db = get_database()

    result = await db.events.update_one(
        {"_id": ObjectId(event_id)},
        {"$set": event_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return {"message": "Event updated successfully"}

@router.delete("/events/{event_id}")
async def delete_event(event_id: str):
    """Delete an event."""
    db = get_database()

    result = await db.events.delete_one({"_id": ObjectId(event_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return {"message": "Event deleted successfully"}

# === RECEIPT MANAGEMENT === #
@router.post("/receipts")
async def submit_receipt(receipt: ReceiptCreate):
    """Submit a receipt for verification."""
    db = get_database()

    # Check if student exists
    student = await db.students.find_one({"student_id": receipt.student_id})
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    # Check if event exists
    event = await db.events.find_one({"_id": ObjectId(receipt.event_id)})
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")

    # Check if receipt already exists for this student and event
    existing_receipt = await db.receipts.find_one({
        "student_id": receipt.student_id,
        "event_id": receipt.event_id
    })
    if existing_receipt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Receipt already submitted for this event")

    receipt_dict = receipt.dict()
    receipt_dict["status"] = "pending"
    receipt_dict["submitted_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    result = await db.receipts.insert_one(receipt_dict)
    return {"message": "Receipt submitted successfully", "receipt_id": str(result.inserted_id)}

@router.get("/receipts")
async def get_receipts(status: str = None):
    """Get all receipts, optionally filtered by status."""
    db = get_database()

    query = {}
    if status:
        query["status"] = status

    receipts = []
    async for receipt in db.receipts.find(query).sort("submitted_at", -1):
        receipt["_id"] = str(receipt["_id"])
        receipts.append(receipt)
    return {"receipts": receipts}

@router.get("/receipts/student/{student_id}")
async def get_student_receipts(student_id: str):
    """Get receipts for a specific student."""
    db = get_database()

    receipts = []
    async for receipt in db.receipts.find({"student_id": student_id}).sort("submitted_at", -1):
        receipt["_id"] = str(receipt["_id"])
        receipts.append(receipt)
    return {"receipts": receipts}

@router.put("/receipts/{receipt_id}/verify")
async def verify_receipt(receipt_id: str, verification_data: dict):
    """Verify or reject a receipt (admin only)."""
    db = get_database()

    status = verification_data.get("status")  # "verified" or "rejected"
    verified_by = verification_data.get("verified_by")

    if status not in ["verified", "rejected"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status")

    result = await db.receipts.update_one(
        {"_id": ObjectId(receipt_id)},
        {"$set": {
            "status": status,
            "verified_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "verified_by": verified_by
        }}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Receipt not found")

    return {"message": f"Receipt {status} successfully"}

@router.delete("/receipts/{receipt_id}")
async def delete_receipt(receipt_id: str):
    """Delete a receipt (admin only)."""
    db = get_database()

    result = await db.receipts.delete_one({"_id": ObjectId(receipt_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Receipt not found")
    return {"message": "Receipt deleted successfully"}

# === ANALYTICS ROUTES === #
@router.get("/analytics/attendance-summary")
async def get_attendance_summary(date_from: str = None, date_to: str = None):
    """Get attendance summary statistics."""
    db = get_database()

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

@router.get("/analytics/student/{student_id}")
async def get_student_attendance(student_id: str, date_from: str = None, date_to: str = None):
    """Get attendance records for a specific student."""
    db = get_database()

    query = {"student_id": student_id}
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}

    records = []
    async for record in db.attendance.find(query).sort("date", -1):
        record["_id"] = str(record["_id"])
        records.append(record)

    return {"student_id": student_id, "attendance": records}

@router.get("/analytics/student/{student_id}/insights")
async def get_student_attendance_insights(student_id: str):
    """Get comprehensive attendance insights for a student."""
    db = get_database()

    # Get total attendance summary
    pipeline = [
        {"$match": {"student_id": student_id}},
        {"$group": {
            "_id": None,
            "total_sessions": {"$sum": 1},
            "present_count": {"$sum": {"$cond": [{"$eq": ["$status", "present"]}, 1, 0]}},
            "absent_count": {"$sum": {"$cond": [{"$eq": ["$status", "absent"]}, 1, 0]}}
        }}
    ]

    summary_result = await db.attendance.aggregate(pipeline).to_list(length=1)
    summary = summary_result[0] if summary_result else {"total_sessions": 0, "present_count": 0, "absent_count": 0}

    total_sessions = summary["total_sessions"]
    attendance_percentage = (summary["present_count"] / total_sessions * 100) if total_sessions > 0 else 0

    # Determine attendance status
    if attendance_percentage >= 90:
        status = "Good Standing"
    elif 75 <= attendance_percentage < 90:
        status = "Warning"
    else:  # At Risk
        status = "At Risk"

    # Get subject-based breakdown
    subject_pipeline = [
        {"$match": {"student_id": student_id}},
        {"$lookup": {
            "from": "classes",
            "localField": "class_id",
            "foreignField": "_id",
            "as": "class_info"
        }},
        {"$unwind": {"path": "$class_info", "preserveNullAndEmptyArrays": True}},
        {"$group": {
            "_id": {
                "subject": {"$ifNull": ["$class_info.class_name", "$subject"]},
                "class_id": "$class_id"
            },
            "total_sessions": {"$sum": 1},
            "present_count": {"$sum": {"$cond": [{"$eq": ["$status", "present"]}, 1, 0]}},
            "absent_count": {"$sum": {"$cond": [{"$eq": ["$status", "absent"]}, 1, 0]}}
        }},
        {"$project": {
            "subject": "$_id.subject",
            "total_sessions": 1,
            "present_count": 1,
            "absent_count": 1,
            "attendance_percentage": {
                "$multiply": [{"$divide": ["$present_count", "$total_sessions"]}, 100]
            }
        }},
        {"$sort": {"subject": 1}}
    ]

    subject_breakdown = []
    async for doc in db.attendance.aggregate(subject_pipeline):
        subject_breakdown.append({
            "subject": doc["subject"] or "Unknown Subject",
            "attendance_percentage": round(doc["attendance_percentage"], 1),
            "present_count": doc["present_count"],
            "absent_count": doc["absent_count"]
        })

    # Get face recognition activity log (assuming face_logs collection exists)
    face_logs = []
    try:
        face_logs_collection = db.face_logs
        async for log in face_logs_collection.find({"student_id": student_id}).sort("timestamp", -1).limit(10):
            face_logs.append({
                "date": log.get("date", ""),
                "time": log.get("time", ""),
                "subject": log.get("subject", "N/A"),
                "result": log.get("result", "Unknown")
            })
    except:
        # If face_logs collection doesn't exist, use attendance records with recognition data
        recognition_pipeline = [
            {"$match": {"student_id": student_id, "status": "present"}},
            {"$lookup": {
                "from": "classes",
                "localField": "class_id",
                "foreignField": "_id",
                "as": "class_info"
            }},
            {"$unwind": {"path": "$class_info", "preserveNullAndEmptyArrays": True}},
            {"$project": {
                "date": 1,
                "time": {"$ifNull": ["$check_in_time", "N/A"]},
                "subject": {"$ifNull": ["$class_info.class_name", "$subject"]},
                "result": "Verified"  # Assuming present means verified
            }},
            {"$sort": {"timestamp": -1}},
            {"$limit": 10}
        ]

        async for doc in db.attendance.aggregate(recognition_pipeline):
            face_logs.append({
                "date": doc["date"],
                "time": doc["time"],
                "subject": doc["subject"] or "Unknown Subject",
                "result": doc["result"]
            })

    # Generate smart feedback
    feedback = generate_attendance_feedback(attendance_percentage, status)

    return {
        "attendance_summary": {
            "total_sessions": total_sessions,
            "present_count": summary["present_count"],
            "absent_count": summary["absent_count"],
            "attendance_percentage": round(attendance_percentage, 1),
            "status": status
        },
        "subject_breakdown": subject_breakdown,
        "face_recognition_logs": face_logs,
        "smart_feedback": feedback
    }

def generate_attendance_feedback(attendance_percentage: float, status: str) -> str:
    """Generate rule-based attendance feedback message."""
    if status == "Good Standing":
        return "Excellent attendance. Keep it up!"
    elif status == "Warning":
        return "Your attendance is in the warning zone. Please maintain regular attendance."
    else:  # At Risk
        return "Your attendance is at risk. Please attend classes regularly to avoid academic penalties."

@router.get("/analytics/class/{class_id}")
async def get_class_attendance(class_id: str, date_from: str = None, date_to: str = None):
    """Get attendance records for a specific class."""
    db = get_database()

    query = {"class_id": class_id}
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}

    records = []
    async for record in db.attendance.find(query).sort("date", -1):
        record["_id"] = str(record["_id"])
        records.append(record)

    return {"class_id": class_id, "attendance": records}

# === DATABASE VIEW ROUTES === #
@router.get("/db/collections")
async def get_db_collections():
    """Get list of all collections in the database."""
    db = get_database()

    try:
        collections = await db.list_collection_names()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"❌ Failed to list collections: {e}")
        return JSONResponse({"error": "Failed to list collections"}, status_code=500)

@router.get("/db/{collection}")
async def get_collection_data(collection: str, limit: int = 10):
    """Get data from a specific collection."""
    db = get_database()

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
