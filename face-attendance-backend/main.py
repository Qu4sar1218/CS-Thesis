import os
import glob
import pickle
import platform
import asyncio
import threading
import time
import logging
import queue
import re
import uuid
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
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))

# Separate folder for student face training data only
FACE_DATA_DIR = os.path.join(PROJECT_ROOT, "StudentFaceData")
ENCODINGS_DIR = os.path.join(BACKEND_ROOT, "data", "encodings")
# Keep logs outside the project tree so uvicorn --reload watchers do not
# retrigger on every log write ("1 change detected" loops).
LOGS_DIR = os.path.join(os.path.expanduser("~"), ".face-attendance", "logs")

os.makedirs(ENCODINGS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ============================================================================
# STORAGE FUNCTIONS - For face training
# ============================================================================

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def save_image_to_storage(student_id, image_data, position=None):
    """
    Save image locally for face training.
    Args:
        student_id: Unique student identifier
        image_data: Raw image bytes
        position: Optional position identifier (front, left, right, up, down)
    Returns:
        str: Path where image was saved
    """
    student_folder = os.path.join(FACE_DATA_DIR, str(student_id))
    os.makedirs(student_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Include position prefix in filename if provided
    if position:
        image_filename = f"{student_id}_{position}_{timestamp}.jpg"
    else:
        image_filename = f"{timestamp}.jpg"
    
    image_path = os.path.join(student_folder, image_filename)

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

# ============================================================================
# FACE TRAINING VALIDATION
# ============================================================================

# Required positions for face training completion
REQUIRED_FACE_POSITIONS = ['front', 'left', 'right', 'up', 'down']

def get_face_training_positions(student_id: str) -> dict:
    """
    Get the captured face training positions for a student.
    Analyzes images in the student's folder to determine which positions have been captured.
    
    Args:
        student_id: Unique student identifier
        
    Returns:
        dict: {
            'front': bool,   # Front-facing (Center)
            'left': bool,    # Face turned left
            'right': bool,   # Face turned right
            'up': bool,      # Face looking up
            'down': bool,    # Face looking down
            'completed': bool  # All positions captured
        }
    """
    student_folder = os.path.join(FACE_DATA_DIR, str(student_id))
    
    positions = {
        'front': False,
        'left': False,
        'right': False,
        'up': False,
        'down': False
    }
    
    if not os.path.exists(student_folder):
        return {**positions, 'completed': False}
    
    # Check for images with position prefixes
    # Image naming format: studentID_position_number.jpg
    # Positions: front, left, right, up, down
    try:
        for filename in os.listdir(student_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Check for position prefixes (support both front and center for backward compatibility)
            if filename.startswith(f"{student_id}_front") or filename.startswith(f"{student_id}_center"):
                positions['front'] = True
            elif filename.startswith(f"{student_id}_left"):
                positions['left'] = True
            elif filename.startswith(f"{student_id}_right"):
                positions['right'] = True
            elif filename.startswith(f"{student_id}_up"):
                positions['up'] = True
            elif filename.startswith(f"{student_id}_down"):
                positions['down'] = True
    except Exception as e:
        logger.warning(f"Error reading face training positions for {student_id}: {e}")
    
    # Check if all positions are captured
    positions['completed'] = all(positions.values())
    
    return positions


def is_face_training_complete(student_id: str) -> bool:
    """
    Check if face training is complete for a student.
    Face training is complete only when all 5 positions are captured.
    
    Args:
        student_id: Unique student identifier
        
    Returns:
        bool: True if all positions are captured, False otherwise
    """
    positions = get_face_training_positions(student_id)
    return positions.get('completed', False)


# ============================================================================
# END OF FACE TRAINING VALIDATION
# ============================================================================

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
active_camera_index = None
DEFAULT_CAMERA_INDEX = 0  # Use default camera (index 0) - more reliable on Windows
preferred_camera_index = DEFAULT_CAMERA_INDEX
preferred_device_id = None  # Store the preferred deviceId for camera selection

known_face_encodings, known_face_names, known_face_ids, known_face_courses, known_face_years = [], [], [], [], []
known_face_first_names, known_face_middle_names, known_face_last_names = [], [], []
encodings_lock = threading.Lock()

# Attendance queue for background processing
attendance_queue = queue.Queue()

# Attendance tracking
attendance_records = []
attendance_lock = threading.Lock()

# Recently recognized student tracking
recently_recognized = None
recently_recognized_lock = threading.Lock()
last_attendance_attempt_by_student = {}
attendance_attempt_lock = threading.Lock()

# Mode tracking
current_mode = "class"  # Default to class mode
current_event_id = None
current_class_id = None
current_class_schedule = None

# Attendance IN/OUT mode tracking
attendance_mode = "IN"  # Default to IN mode (for admin to track time in/time out)
attendance_mode_lock = threading.Lock()

# Legacy monitoring lock (kept for backward compatibility but no longer used)
monitoring_lock = threading.Lock()
# Legacy monitoring variables (kept for backward compatibility but no longer used)
monitoring_mode = False
monitoring_session_id = None
monitoring_start_time = None
monitoring_previous_class_id = None
monitoring_revalidation_seconds = 120
monitoring_timer_thread = None
monitoring_students_pending = []
monitoring_finalize_status = "ABSENT"

def is_class_scheduled_today(schedule: str) -> bool:
    """
    Check if a class is scheduled for today based on its schedule string.

    Args:
        schedule: Schedule string like "MWF 9:00-10:00"

    Returns:
        bool: True if scheduled for today, False otherwise
    """
    if not schedule:
        return False

    # Split schedule into days and time parts
    parts = schedule.split()
    if not parts:
        return False

    days_str = parts[0]  # e.g., "MWF" or "ThF"

    # Day codes mapping (weekday() returns 0=Monday, 1=Tuesday, etc.)
    day_codes = ['M', 'T', 'W', 'Th', 'F', 'S', 'Su']
    today_weekday = datetime.now().weekday()
    today_code = day_codes[today_weekday]

    # Parse days_str into individual day codes
    possible_days = ['Su', 'Th', 'M', 'T', 'W', 'F', 'S']  # Check longer codes first
    scheduled_days = []
    i = 0
    while i < len(days_str):
        found = False
        for day in possible_days:
            if days_str.startswith(day, i):
                scheduled_days.append(day)
                i += len(day)
                found = True
                break
        if not found:
            i += 1  # Skip invalid character

    # Check if today_code is in scheduled_days
    return today_code in scheduled_days


def parse_class_start_time(schedule: str) -> Optional[datetime]:
    """
    Parse the start time from a class schedule string.

    Args:
        schedule: Schedule string like "MWF 9:00-10:00"

    Returns:
        datetime: Today's date with the start time, or None if parsing fails
    """
    if not schedule:
        return None

    try:
        # Split schedule into parts
        parts = schedule.split()
        if len(parts) < 2:
            return None

        # Get the time part (e.g., "9:00-10:00")
        time_range = parts[1]
        if '-' not in time_range:
            return None

        # Split to get start time (e.g., "9:00")
        start_time_str = time_range.split('-')[0]

        # Parse the time
        today = datetime.now()
        start_time = datetime.strptime(start_time_str, "%H:%M")

        # Combine today's date with the start time
        return datetime(
            today.year,
            today.month,
            today.day,
            start_time.hour,
            start_time.minute
        )
    except Exception as e:
        logger.warning(f"Failed to parse class start time from schedule '{schedule}': {e}")
        return None


def determine_attendance_status(scan_time: datetime, class_start_time: datetime, grace_period_minutes: int = 15) -> str:
    """
    Determine attendance status based on scan time and class start time.

    Args:
        scan_time: The time when the student scanned
        class_start_time: The scheduled start time of the class
        grace_period_minutes: Number of minutes after start time to allow (default: 15)

    Returns:
        str: "present" if on or before grace period, "late" if after
    """
    # Calculate the end of the grace period
    grace_period_end = class_start_time + timedelta(minutes=grace_period_minutes)

    # If scan time is on or before the grace period end, mark as present
    if scan_time <= grace_period_end:
        return "present"
    else:
        return "late"


def get_class_section_signature(class_doc: dict) -> str:
    """
    Build a best-effort section signature for class-to-class comparison.
    Priority: explicit section -> courses list -> room.
    """
    if not class_doc:
        return ""

    section = (class_doc.get("section") or "").strip().lower()
    if section:
        return section

    courses = class_doc.get("courses") or []
    if isinstance(courses, list) and courses:
        normalized = [str(course).strip().lower() for course in courses if str(course).strip()]
        if normalized:
            return "|".join(sorted(normalized))

    return (class_doc.get("room") or "").strip().lower()


def parse_timestamp_to_datetime(value: str) -> Optional[datetime]:
    """Parse attendance timestamp strings safely."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


async def find_previous_consecutive_class(current_class_id: str, max_gap_minutes: int = 15) -> Optional[dict]:
    """
    Find latest class on the same day that matches section signature and happened recently.
    """
    try:
        class_obj_id = ObjectId(current_class_id)
    except Exception:
        return None

    current_class = await db.classes.find_one({"_id": class_obj_id})
    if not current_class:
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    section_signature = get_class_section_signature(current_class)
    teacher_id = current_class.get("teacher_id")
    if not section_signature:
        return None

    class_candidates = []
    async for class_doc in db.classes.find({"teacher_id": teacher_id}):
        if str(class_doc.get("_id")) == current_class_id:
            continue
        if get_class_section_signature(class_doc) != section_signature:
            continue
        class_candidates.append(class_doc)

    if not class_candidates:
        return None

    class_id_map = {str(doc["_id"]): doc for doc in class_candidates}
    candidate_ids = list(class_id_map.keys())
    latest_scan = {}

    query = {
        "date": today,
        "class_id": {"$in": candidate_ids},
        "mode": {"$ne": "events"},
        "timestamp": {"$exists": True}
    }
    async for attendance_doc in db.attendance.find(query).sort("timestamp", -1):
        cid = str(attendance_doc.get("class_id"))
        if cid in latest_scan:
            continue
        scan_dt = parse_timestamp_to_datetime(attendance_doc.get("timestamp"))
        if scan_dt:
            latest_scan[cid] = scan_dt

    if not latest_scan:
        return None

    now = datetime.now()
    best_class_id = None
    best_dt = None
    for cid, scan_dt in latest_scan.items():
        delta_minutes = (now - scan_dt).total_seconds() / 60
        if delta_minutes < 0 or delta_minutes > max_gap_minutes:
            continue
        if best_dt is None or scan_dt > best_dt:
            best_dt = scan_dt
            best_class_id = cid

    if not best_class_id or not best_dt:
        return None

    previous_class = class_id_map.get(best_class_id, {})
    present_count = await db.attendance.count_documents({
        "date": today,
        "class_id": best_class_id,
        "status": {"$in": ["present", "late", "PRESENT", "LATE"]}
    })

    return {
        "previous_class_id": best_class_id,
        "previous_class_name": previous_class.get("class_name", ""),
        "last_scan_timestamp": best_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "minutes_since_last_scan": int((now - best_dt).total_seconds() // 60),
        "present_count": present_count
    }


async def start_monitoring_revalidation_session(
    class_id: str,
    previous_class_id: str,
    fallback_status: str = "ABSENT"
) -> dict:
    """
    Initialize monitoring mode by carrying over PRESENT/LATE students as pending revalidation.
    """
    global monitoring_mode, monitoring_session_id, monitoring_start_time, monitoring_previous_class_id
    global monitoring_students_pending, monitoring_timer_thread, monitoring_finalize_status

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    timestamp_now = now.strftime("%Y-%m-%d %H:%M:%S")

    try:
        class_obj_id = ObjectId(class_id)
        previous_class_obj_id = ObjectId(previous_class_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid class_id or previous_class_id")

    class_doc = await db.classes.find_one({"_id": class_obj_id})
    if not class_doc:
        raise HTTPException(status_code=404, detail="Class not found")

    previous_class_doc = await db.classes.find_one({"_id": previous_class_obj_id})
    if not previous_class_doc:
        raise HTTPException(status_code=404, detail="Previous class not found")

    # Keep absent baseline for all enrolled students (full-rescan parity, duplicate-safe via upsert).
    enrolled_students = class_doc.get("enrolled_students", [])
    for student_id in enrolled_students:
        await db.attendance.update_one(
            {"student_id": student_id, "class_id": class_id, "date": today},
            {
                "$setOnInsert": {
                    "student_id": student_id,
                    "class_id": class_id,
                    "date": today,
                    "status": "absent",
                    "timestamp": timestamp_now,
                    "subject": class_doc.get("class_name", "Unknown Subject"),
                    "mode": "class"
                }
            },
            upsert=True
        )

    previous_present_records = db.attendance.find({
        "class_id": previous_class_id,
        "date": today,
        "status": {"$in": ["present", "late", "PRESENT", "LATE"]}
    })

    pending_ids = []
    pending_count = 0
    async for prev_record in previous_present_records:
        student_id = prev_record.get("student_id")
        if not student_id:
            continue

        pending_ids.append(student_id)
        await db.attendance.update_one(
            {"student_id": student_id, "class_id": class_id, "date": today},
            {
                "$set": {
                    "status": "PENDING_REVALIDATION",
                    "revalidated": False,
                    "monitoring_session_id": "",
                    "mode": "class",
                    "timestamp": timestamp_now,
                    "subject": class_doc.get("class_name", "Unknown Subject"),
                    "monitoring_source_class_id": previous_class_id
                },
                "$setOnInsert": {
                    "student_id": student_id,
                    "class_id": class_id,
                    "date": today
                }
            },
            upsert=True
        )
        pending_count += 1

    new_session_id = str(uuid.uuid4())
    await db.attendance.update_many(
        {"class_id": class_id, "date": today, "status": "PENDING_REVALIDATION"},
        {"$set": {"monitoring_session_id": new_session_id}}
    )

    with monitoring_lock:
        monitoring_mode = True
        monitoring_session_id = new_session_id
        monitoring_start_time = now
        monitoring_previous_class_id = previous_class_id
        monitoring_students_pending = pending_ids
        monitoring_finalize_status = fallback_status if fallback_status in {"ABSENT", "NEEDS_MANUAL_CONFIRMATION"} else "ABSENT"

    def _monitoring_timer_worker(target_session_id: str, wait_seconds: int):
        time.sleep(wait_seconds)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(finalize_monitoring_revalidation_session(target_session_id))
        except Exception as e:
            logger.error(f"Monitoring finalize failed for {target_session_id}: {e}")
        finally:
            loop.close()

    monitoring_timer_thread = threading.Thread(
        target=_monitoring_timer_worker,
        args=(new_session_id, monitoring_revalidation_seconds),
        daemon=True
    )
    monitoring_timer_thread.start()

    return {
        "monitoring_session_id": new_session_id,
        "pending_students": pending_count,
        "revalidation_seconds": monitoring_revalidation_seconds
    }


async def start_standalone_revalidation_session(
    class_id: str,
    fallback_status: str = "ABSENT"
) -> dict:
    """
    Initialize standalone revalidation mode for a class without requiring a previous class.
    All enrolled students will start as PENDING_REVALIDATION and need to scan again to confirm attendance.
    """
    global monitoring_mode, monitoring_session_id, monitoring_start_time, monitoring_previous_class_id
    global monitoring_students_pending, monitoring_timer_thread, monitoring_finalize_status

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    timestamp_now = now.strftime("%Y-%m-%d %H:%M:%S")

    try:
        class_obj_id = ObjectId(class_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid class_id")

    class_doc = await db.classes.find_one({"_id": class_obj_id})
    if not class_doc:
        raise HTTPException(status_code=404, detail="Class not found")

    # Get enrolled students
    enrolled_students = class_doc.get("enrolled_students", [])
    
    if not enrolled_students:
        return {
            "monitoring_session_id": None,
            "pending_students": 0,
            "revalidation_seconds": monitoring_revalidation_seconds,
            "message": "No enrolled students in this class"
        }

    # Set all enrolled students as PENDING_REVALIDATION
    pending_ids = []
    pending_count = 0
    
    for student_id in enrolled_students:
        pending_ids.append(student_id)
        await db.attendance.update_one(
            {"student_id": student_id, "class_id": class_id, "date": today},
            {
                "$set": {
                    "status": "PENDING_REVALIDATION",
                    "revalidated": False,
                    "monitoring_session_id": "",  # Will be updated below
                    "mode": "class",
                    "timestamp": timestamp_now,
                    "subject": class_doc.get("class_name", "Unknown Subject"),
                    "revalidation_type": "standalone"  # Flag to indicate standalone revalidation
                },
                "$setOnInsert": {
                    "student_id": student_id,
                    "class_id": class_id,
                    "date": today
                }
            },
            upsert=True
        )
        pending_count += 1

    new_session_id = str(uuid.uuid4())
    await db.attendance.update_many(
        {"class_id": class_id, "date": today, "status": "PENDING_REVALIDATION"},
        {"$set": {"monitoring_session_id": new_session_id}}
    )

    with monitoring_lock:
        monitoring_mode = True
        monitoring_session_id = new_session_id
        monitoring_start_time = now
        monitoring_previous_class_id = None  # No previous class in standalone mode
        monitoring_students_pending = pending_ids
        monitoring_finalize_status = fallback_status if fallback_status in {"ABSENT", "NEEDS_MANUAL_CONFIRMATION"} else "ABSENT"

    def _monitoring_timer_worker(target_session_id: str, wait_seconds: int):
        time.sleep(wait_seconds)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(finalize_monitoring_revalidation_session(target_session_id))
        except Exception as e:
            logger.error(f"Monitoring finalize failed for {target_session_id}: {e}")
        finally:
            loop.close()

    monitoring_timer_thread = threading.Thread(
        target=_monitoring_timer_worker,
        args=(new_session_id, monitoring_revalidation_seconds),
        daemon=True
    )
    monitoring_timer_thread.start()

    return {
        "monitoring_session_id": new_session_id,
        "pending_students": pending_count,
        "revalidation_seconds": monitoring_revalidation_seconds,
        "mode": "standalone"
    }


async def finalize_monitoring_revalidation_session(session_id: str) -> dict:
    """Finalize pending students for a monitoring session after revalidation window ends."""
    global monitoring_mode, monitoring_session_id, monitoring_start_time, monitoring_previous_class_id
    global monitoring_students_pending

    with monitoring_lock:
        if not monitoring_session_id or monitoring_session_id != session_id:
            return {"finalized": False, "reason": "Session not active"}
        fallback_status = monitoring_finalize_status
        monitoring_mode = False
        monitoring_start_time = None
        monitoring_previous_class_id = None
        pending_ids = list(monitoring_students_pending)
        monitoring_students_pending = []
        monitoring_session_id = None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = await db.attendance.update_many(
        {"monitoring_session_id": session_id, "status": "PENDING_REVALIDATION", "revalidated": False},
        {
            "$set": {
                "status": fallback_status,
                "monitoring_resolved_at": now,
                "monitoring_resolution": "timer_elapsed"
            }
        }
    )
    return {"finalized": True, "updated_count": result.modified_count, "pending_ids": pending_ids}


def parse_event_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Parse event date and time into a datetime object."""
    if not date_str or not time_str:
        return None

    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(f"{date_str} {time_str}", fmt)
        except ValueError:
            continue
    return None


def parse_event_date(date_str: str) -> Optional[datetime]:
    """Parse event date string to datetime."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


async def mark_event_absences_if_ended(event_id: str, event_doc: dict, now: Optional[datetime] = None) -> int:
    """
    Backfill absences for verified students once the event has ended.
    Returns the number of inserted absent records.
    """
    now = now or datetime.now()
    start_dt = parse_event_datetime(event_doc.get("date"), event_doc.get("start_time"))
    end_dt = parse_event_datetime(event_doc.get("date"), event_doc.get("end_time"))
    if not start_dt or not end_dt or now <= end_dt:
        return 0

    event_date = start_dt.strftime("%Y-%m-%d")
    inserted_count = 0
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    async for receipt in db.receipts.find({"event_id": event_id, "status": "verified"}):
        student_id = receipt.get("student_id")
        if not student_id:
            continue
        result = await db.attendance.update_one(
            {
                "student_id": student_id,
                "event_id": event_id,
                "date": event_date
            },
            {
                "$setOnInsert": {
                    "student_id": student_id,
                    "event_id": event_id,
                    "date": event_date,
                    "timestamp": timestamp,
                    "mode": "events",
                    "status": "absent",
                    "event_name": event_doc.get("name", "Unknown Event"),
                    "course": "",
                    "year": ""
                }
            },
            upsert=True
        )
        if result.upserted_id:
            inserted_count += 1

    return inserted_count


async def process_event_attendance_scan(scan_record: dict) -> dict:
    """Validate and persist an event attendance scan."""
    event_id = scan_record.get("event_id")
    student_id = scan_record.get("student_id")
    scan_dt = scan_record.get("scan_datetime_obj") or datetime.now()

    if not event_id:
        return {"recorded": False, "reason": "No active event selected", "status": "blocked"}

    try:
        event_obj_id = ObjectId(event_id)
    except Exception:
        return {"recorded": False, "reason": "Invalid event id", "status": "blocked"}

    event_doc = await db.events.find_one({"_id": event_obj_id})
    if not event_doc:
        return {"recorded": False, "reason": "Event not found", "status": "blocked"}

    start_dt = parse_event_datetime(event_doc.get("date"), event_doc.get("start_time"))
    end_dt = parse_event_datetime(event_doc.get("date"), event_doc.get("end_time"))
    grace_period = int(event_doc.get("grace_period_minutes", 15))

    if not start_dt or not end_dt:
        return {
            "recorded": False,
            "reason": "Event schedule is incomplete (start/end time required)",
            "status": "blocked"
        }

    if end_dt <= start_dt:
        return {"recorded": False, "reason": "Invalid event schedule", "status": "blocked"}

    # First check if any receipt exists for this student and event
    receipt = await db.receipts.find_one({
        "student_id": student_id,
        "event_id": event_id
    })
    
    if not receipt:
        return {
            "recorded": False,
            "reason": "No receipt submitted for this event",
            "status": "no_receipt"
        }

    # Check receipt status
    receipt_status = receipt.get("status")
    if receipt_status == "verified":
        # Receipt is verified, allow attendance
        pass  # Continue to attendance recording
    elif receipt_status == "pending":
        return {
            "recorded": False,
            "reason": "Receipt pending verification",
            "status": "receipt_pending"
        }
    elif receipt_status == "rejected":
        return {
            "recorded": False,
            "reason": "Receipt was rejected",
            "status": "receipt_rejected"
        }
    else:
        return {
            "recorded": False,
            "reason": "Receipt not verified",
            "status": "blocked"
        }

    if scan_dt < start_dt:
        return {
            "recorded": False,
            "reason": "Event has not started yet",
            "status": "blocked"
        }

    if scan_dt > end_dt:
        await mark_event_absences_if_ended(event_id, event_doc, now=scan_dt)
        return {
            "recorded": False,
            "reason": "Event already ended",
            "status": "absent"
        }

    grace_end = start_dt + timedelta(minutes=grace_period)
    attendance_status = "present" if scan_dt <= grace_end else "late"
    event_date = start_dt.strftime("%Y-%m-%d")
    time_str = scan_dt.strftime("%H:%M:%S")
    timestamp = scan_dt.strftime("%Y-%m-%d %H:%M:%S")

    persisted_record = {
        "student_id": student_id,
        "name": scan_record.get("name"),
        "event_id": event_id,
        "event_name": event_doc.get("name", "Unknown Event"),
        "mode": "events",
        "date": event_date,
        "check_in_time": time_str,
        "time": time_str,
        "timestamp": timestamp,
        "course": scan_record.get("course", ""),
        "year": scan_record.get("year", ""),
        "status": attendance_status
    }

    result = await db.attendance.update_one(
        {
            "student_id": student_id,
            "event_id": event_id,
            "date": event_date
        },
        {"$setOnInsert": persisted_record},
        upsert=True
    )

    if not result.upserted_id:
        return {
            "recorded": False,
            "reason": "Attendance already recorded for today",
            "status": attendance_status
        }

    return {"recorded": True, "record": persisted_record, "status": attendance_status}

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
        logger.info("âœ… Connected to MongoDB")
    except Exception as e:
        logger.error(f"âŒ Failed to connect to MongoDB: {e}")
        raise

async def close_mongodb_connection():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        logger.info("âœ… MongoDB connection closed")

# Authentication utilities
def verify_password(plain_password, hashed_password):
    """Verify a password against its hash."""
    if not hashed_password:
        return False
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        # If hash is invalid, check if it's plain text (for migration)
        if hashed_password == plain_password:
            logger.warning("Plain text password detected, treating as valid for migration")
            return True
        return False

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


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate bearer token and return authenticated user payload."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    subject = (payload.get("sub") or "").strip()
    role = (payload.get("role") or "").strip().lower()
    if not subject or not role:
        raise HTTPException(status_code=401, detail="Invalid authentication payload")

    return {"sub": subject, "role": role}


def require_roles(allowed_roles: List[str]):
    """Dependency factory for role-based access control."""
    normalized_allowed = {role.strip().lower() for role in allowed_roles if role and role.strip()}

    async def _role_dependency(current_user: dict = Depends(get_current_user)):
        current_role = (current_user.get("role") or "").strip().lower()
        if current_role not in normalized_allowed:
            raise HTTPException(status_code=403, detail="Only admin or teacher can perform this action")
        return current_user

    return _role_dependency

async def save_attendance_to_db(record: dict):
    """Save attendance record to database."""
    try:
        attendance_collection = db.attendance
        result = await attendance_collection.insert_one(record)
        logger.info(f"âœ… Attendance saved to DB: {record['name']}")
    except Exception as e:
        logger.error(f"âŒ Failed to save attendance to DB: {e}")

async def update_attendance_status(student_id: str, class_id: str, status: str, record: Optional[dict] = None):
    """Upsert attendance status for a student in a class."""
    try:
        current_date = datetime.now().strftime('%Y-%m-%d')
        attendance_collection = db.attendance

        record_payload = record or {}
        set_fields = {
            "status": status,
            "timestamp": record_payload.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            "check_in_time": record_payload.get("check_in_time", datetime.now().strftime('%H:%M:%S')),
            "mode": record_payload.get("mode", "class")
        }
        if record_payload:
            set_fields["course"] = record_payload.get("course", "")
            set_fields["year"] = record_payload.get("year", "")
            set_fields["name"] = record_payload.get("name", "")
            if "subject" in record_payload:
                set_fields["subject"] = record_payload.get("subject")
            if "revalidated" in record_payload:
                set_fields["revalidated"] = bool(record_payload.get("revalidated"))
            if "monitoring_session_id" in record_payload:
                set_fields["monitoring_session_id"] = record_payload.get("monitoring_session_id")
            if "revalidated_at" in record_payload:
                set_fields["revalidated_at"] = record_payload.get("revalidated_at")
            if "monitoring_resolved_at" in record_payload:
                set_fields["monitoring_resolved_at"] = record_payload.get("monitoring_resolved_at")
            if "monitoring_resolution" in record_payload:
                set_fields["monitoring_resolution"] = record_payload.get("monitoring_resolution")

        result = await attendance_collection.update_one(
            {
                "student_id": student_id,
                "class_id": class_id,
                "date": current_date
            },
            {
                "$set": set_fields,
                "$setOnInsert": {
                    "student_id": student_id,
                    "class_id": class_id,
                    "date": current_date
                }
            },
            upsert=True
        )

        if result.modified_count > 0:
            logger.info(f"Updated attendance status for {student_id} in class {class_id} to {status}")
        elif result.upserted_id:
            logger.info(f"Created attendance record for {student_id} in class {class_id} as {status}")
        else:
            logger.info(f"Attendance status unchanged for {student_id} in class {class_id}")

    except Exception as e:
        logger.error(f"Failed to update attendance status for {student_id}: {e}")

def record_attendance(student_id, name, course, year, first_name="", middle_name="", last_name=""):
    """Record attendance without blocking the recognition thread."""
    global attendance_mode
    
    now_dt = datetime.now()
    current_time = now_dt.strftime('%H:%M:%S')
    current_date = now_dt.strftime('%Y-%m-%d')
    current_timestamp = now_dt.strftime('%Y-%m-%d %H:%M:%S')

    if current_mode == "events":
        queue_record = {
            "mode": "events",
            "event_id": current_event_id,
            "name": name,
            "student_id": student_id,
            "timestamp": current_timestamp,
            "date": current_date,
            "time": current_time,
            "check_in_time": current_time,
            "course": course,
            "year": year,
            "scan_datetime_obj": now_dt
        }
        with recently_recognized_lock:
            recently_recognized = {
                "name": name,
                "student_id": student_id,
                "event_id": current_event_id,
                "time": current_time,
                "date": current_date,
                "course": course,
                "year": year,
                "status": "processing",
                "message": "Processing attendance..."
            }
        attendance_queue.put(queue_record)
        return

    # Get current attendance mode (IN or OUT)
    with attendance_mode_lock:
        current_attendance_mode = attendance_mode

    class_start = parse_class_start_time(current_class_schedule or "")
    status = determine_attendance_status(now_dt, class_start, 15) if class_start else "present"
    message = None  # Initialize message for recently recognized
    is_monitoring_revalidation = False
    active_monitoring_session = None

    with monitoring_lock:
        if monitoring_mode and monitoring_session_id and student_id in monitoring_students_pending:
            is_monitoring_revalidation = True
            active_monitoring_session = monitoring_session_id
            status = "PRESENT"
            monitoring_students_pending.remove(student_id)

    with attendance_lock:
        today_records = [
            r for r in attendance_records
            if r.get('student_id') == student_id
            and r.get('date') == current_date
            and r.get('class_id') == current_class_id
        ]

        # Check for existing attendance records for this student today
        has_time_in = any(r.get('attendance_type') == 'TIME_IN' for r in today_records)
        has_time_out = any(r.get('attendance_type') == 'TIME_OUT' for r in today_records)

        if current_attendance_mode == "IN":
            # IN Mode - Record TIME IN
            if has_time_in:
                message = "Already Timed In"
                status = "already_timed_in"
            else:
                record = {
                    'mode': 'class',
                    'name': name,
                    'student_id': student_id,
                    'class_id': current_class_id,
                    'timestamp': current_timestamp,
                    'date': current_date,
                    'time': current_time,
                    'check_in_time': current_time,
                    'course': course,
                    'year': year,
                    'status': status,
                    'attendance_type': 'TIME_IN'
                }
                if is_monitoring_revalidation and active_monitoring_session:
                    record["revalidated"] = True
                    record["revalidated_at"] = current_timestamp
                    record["monitoring_session_id"] = active_monitoring_session
                attendance_records.append(record)
                attendance_queue.put(record)
                message = "Time In Recorded"
        else:
            # OUT Mode - Record TIME OUT
            if not has_time_in:
                message = "Must Time In First"
                status = "no_time_in"
            elif has_time_out:
                message = "Already Timed Out"
                status = "already_timed_out"
            else:
                record = {
                    'mode': 'class',
                    'name': name,
                    'student_id': student_id,
                    'class_id': current_class_id,
                    'timestamp': current_timestamp,
                    'date': current_date,
                    'time': current_time,
                    'check_out_time': current_time,
                    'course': course,
                    'year': year,
                    'status': 'present',
                    'attendance_type': 'TIME_OUT'
                }
                attendance_records.append(record)
                attendance_queue.put(record)
                message = "Time Out Recorded"
                # Auto-reset to IN mode after successful TIME OUT
                with attendance_mode_lock:
                    attendance_mode = "IN"
                    logger.info(f"Auto-reset attendance mode to IN after TIME OUT for student {student_id}")

    with recently_recognized_lock:
        # Determine attendance_type based on current_attendance_mode
        attendance_type = 'TIME_IN' if current_attendance_mode == 'IN' else 'TIME_OUT'
        recently_recognized = {
            'name': name,
            'student_id': student_id,
            'time': current_time,
            'date': current_date,
            'course': course,
            'year': year,
            'status': status,
            'message': message,
            'attendance_mode': current_attendance_mode,
            'attendance_type': attendance_type
        }


def should_attempt_attendance(student_id: str, now_ts: float, cooldown_seconds: float) -> bool:
    """Rate-limit attendance processing per student to reduce recognition jitter."""
    with attendance_attempt_lock:
        last_ts = last_attendance_attempt_by_student.get(student_id)
        if last_ts is not None and (now_ts - last_ts) < cooldown_seconds:
            return False
        last_attendance_attempt_by_student[student_id] = now_ts
        return True

def attendance_worker():
    """Background worker to save attendance to database without blocking."""
    # Create a single event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        while True:
            try:
                record = attendance_queue.get(timeout=1)
                if record is None:  # Poison pill to stop
                    break

                try:
                    if record.get("mode") == "events":
                        result = loop.run_until_complete(process_event_attendance_scan(record))
                        with recently_recognized_lock:
                            # Get status from result - if recorded=True use status, if recorded=False also use status (could be present/late for already recorded, or blocked for errors)
                            # Use correct status: if recorded=True use status, if recorded=False also use status (could be present/late for already recorded, or blocked for errors)
                            attendance_status = result.get("status", "present") if result.get("recorded") else result.get("status", "blocked")
                            recently_recognized = {
                                "name": record.get("name"),
                                "student_id": record.get("student_id"),
                                "event_id": record.get("event_id"),
                                "time": record.get("time"),
                                "date": record.get("date"),
                                "course": record.get("course"),
                                "year": record.get("year"),
                                "status": attendance_status,
                                "message": result.get("reason") if not result.get("recorded") else "Attendance recorded"
                            }

                        if result.get("recorded") and result.get("record"):
                            with attendance_lock:
                                attendance_records.append(result["record"])
                        elif result.get("status") in ["present", "late"]:
                            # Even if already recorded, append to attendance_records for display
                            # Create a record from the queue data
                            existing_record = {
                                "mode": "events",
                                "name": record.get("name"),
                                "student_id": record.get("student_id"),
                                "event_id": record.get("event_id"),
                                "date": record.get("date"),
                                "time": record.get("time"),
                                "check_in_time": record.get("check_in_time"),
                                "course": record.get("course"),
                                "year": record.get("year"),
                                "status": result.get("status"),
                                "timestamp": record.get("timestamp")
                            }
                            with attendance_lock:
                                attendance_records.append(existing_record)
                            # Also update recently_recognized to show success status for already recorded attendance
                            with recently_recognized_lock:
                                recently_recognized = {
                                    "name": record.get("name"),
                                    "student_id": record.get("student_id"),
                                    "event_id": record.get("event_id"),
                                    "time": record.get("time"),
                                    "date": record.get("date"),
                                    "course": record.get("course"),
                                    "year": record.get("year"),
                                    "status": result.get("status"),
                                    "message": "Attendance already recorded"
                                }
                    else:
                        loop.run_until_complete(
                            update_attendance_status(
                                record['student_id'],
                                record.get('class_id') or current_class_id,
                                record.get('status', 'present'),
                                record
                            )
                        )
                except Exception as e:
                    logger.error(f"Error processing attendance record: {e}")
                    if record.get("mode") == "events":
                        with recently_recognized_lock:
                            recently_recognized = {
                                "name": record.get("name"),
                                "student_id": record.get("student_id"),
                                "event_id": record.get("event_id"),
                                "time": record.get("time"),
                                "date": record.get("date"),
                                "course": record.get("course"),
                                "year": record.get("year"),
                                "status": "blocked",
                                "message": "Attendance processing failed"
                            }
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in attendance worker: {e}")
    finally:
        loop.close()

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
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    grace_period_minutes: int = Field(default=15, ge=0, le=180)
    late_limit_hours: int = Field(default=1, ge=0)
    absent_after_hours: int = Field(default=2, ge=0)
    location: str
    price: Optional[float] = 0.0

class EventCreate(EventBase):
    pass

class Event(EventBase):
    id: str = Field(alias="_id")

class EventScheduleUpdate(BaseModel):
    start_time: str
    end_time: str
    grace_period_minutes: int = Field(default=15, ge=0, le=180)

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

# Enrollment Models
class EnrollmentBase(BaseModel):
    student_id: str
    class_id: str
    enrolled_at: Optional[str] = None
    enrolled_by: Optional[str] = None

class EnrollmentCreate(BaseModel):
    student_id: str
    class_id: str

class Enrollment(EnrollmentBase):
    id: str = Field(alias="_id")

class SubjectEnrollmentRequest(BaseModel):
    class_id: str
    enrolled_by: Optional[str] = "system"

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

FACE_MATCH_THRESHOLD = 0.48  # Stricter threshold — reduces false positives significantly
PROCESS_EVERY_N_FRAMES_ACTIVE = 5  # Process more frequently when faces are present
PROCESS_EVERY_N_FRAMES_IDLE = 15   # Slower cadence when no faces visible
JPEG_QUALITY = 40  # Much lower - much smaller files for faster transfer
FRAME_SCALE = 0.5  # 50% scale — better detail for recognition vs 25%
FRAME_RATE_LIMIT = 15 # Very low FPS for maximum stability
DETECTION_RESOLUTION = (320, 240)  # Lower resolution for faster face detection
ATTENDANCE_RETRY_COOLDOWN_SECONDS = 10.0  # Prevent repeated attendance processing on the same face
RECOGNITION_CONFIRM_FRAMES = 3  # Require N consecutive matching frames before recording attendance
RECOGNITION_VOTE_RATIO = 0.6    # Minimum ratio of encodings that must agree on a match (voting)

# App       
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Database events
@app.on_event("startup")
async def startup_event():
    await connect_to_mongodb()
    await load_faces_from_db()
    # Create unique index for enrollments collection to prevent duplicates
    await create_enrollment_indexes()
    # Start the attendance worker thread
    attendance_worker_thread = threading.Thread(target=attendance_worker, daemon=True)
    attendance_worker_thread.start()
    logger.info("ðŸš€ Server started - loading faces from database")

@app.on_event("shutdown")
async def shutdown_event():
    attendance_queue.put(None)  # Stop worker thread
    await close_mongodb_connection()

async def create_enrollment_indexes():
    """Create indexes for the enrollments collection."""
    try:
        # Create unique index on student_id + class_id to prevent duplicates
        await db.enrollments.create_index(
            [("student_id", 1), ("class_id", 1)],
            unique=True,
            name="student_class_unique"
        )
        logger.info("âœ… Created unique index on enrollments collection (student_id + class_id)")
    except Exception as e:
        logger.warning(f"âš ï¸ Could not create enrollment index: {e}")


# === FACE LOADING === #
async def load_faces_from_db():
    global known_face_encodings, known_face_names, known_face_ids, known_face_courses, known_face_years
    global known_face_first_names, known_face_middle_names, known_face_last_names
    logger.info("ðŸ”„ Loading saved face encodings from database...")

    with encodings_lock:
        known_face_encodings.clear()
        known_face_names.clear()
        known_face_ids.clear()
        known_face_courses.clear()
        known_face_years.clear()
        known_face_first_names.clear()
        known_face_middle_names.clear()
        known_face_last_names.clear()

        try:
            students_collection = db.students
            async for student in students_collection.find({"face_encodings": {"$exists": True, "$ne": None}}):
                student_id = student["student_id"]
                first_name = student.get("first_name", "")
                middle_name = student.get("middle_name", "")
                last_name = student.get("last_name", "")
                full_name = format_student_name(first_name, middle_name, last_name)
                course = student.get("course", "Unknown")
                year = student.get("year", "Unknown")
                enc_list = student.get("face_encodings", [])
                # Safety check: handle None values
                if enc_list is None:
                    enc_list = []
                loaded_count = 0
                for enc in enc_list:
                    if isinstance(enc, list) and len(enc) == 128:
                        known_face_encodings.append(np.array(enc))
                        known_face_names.append(full_name)
                        known_face_ids.append(student_id)
                        known_face_courses.append(course)
                        known_face_years.append(year)
                        known_face_first_names.append(first_name)
                        known_face_middle_names.append(middle_name)
                        known_face_last_names.append(last_name)
                        loaded_count += 1
                    else:
                        logger.warning(f"Skipping invalid encoding for {student_id}: {type(enc)} len={len(enc) if isinstance(enc, list) else 'N/A'}")
                logger.info(f"âœ… Loaded {loaded_count}/{len(enc_list)} encodings for: {full_name} ({student_id}) - Course: {course}, Year: {year}")
        except Exception as e:
            logger.error(f"âŒ Failed to load face encodings from database: {e}")

    logger.info(f"âœ… Loaded {len(known_face_encodings)} known face encodings total from database")

def load_faces_from_disk():
    """Fallback function to load from disk if needed."""
    global known_face_encodings, known_face_names, known_face_ids
    logger.info("ðŸ”„ Loading saved face encodings from disk (fallback)...")
    logger.info(f"ðŸ“‚ Looking for encodings in: {ENCODINGS_DIR}")

    encs = glob.glob(os.path.join(ENCODINGS_DIR, "*.pkl"))
    logger.info(f"ðŸ“„ Found {len(encs)} .pkl files")

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
                logger.info(f"âœ… Loaded {loaded_count}/{len(enc_list)} encodings for: {name}")
            except Exception as e:
                logger.warning(f"âŒ Failed to load {file}: {e}")

    logger.info(f"âœ… Loaded {len(known_face_encodings)} known face encodings total")


# === CAMERA === #
def release_camera():
    global active_camera, active_camera_index, preferred_device_id
    if active_camera:
        try:
            active_camera.release()
        except Exception as e:
            logger.warning(f"Error releasing camera: {e}")
        active_camera = None
        active_camera_index = None
        preferred_device_id = None


def get_camera_backends_for_platform(prefer_dshow: bool = False) -> List[int]:
    """
    Choose camera backends by platform.
    On many Windows setups, DSHOW by index emits warnings/fails; keep it as fallback.
    """
    if platform.system() == "Windows":
        allow_dshow = os.getenv("CAMERA_ALLOW_DSHOW", "0").strip().lower() in {"1", "true", "yes", "on"}
        allow_cap_any = os.getenv("CAMERA_ALLOW_CAP_ANY", "0").strip().lower() in {"1", "true", "yes", "on"}
        backends: List[int] = [cv2.CAP_MSMF]
        if allow_cap_any:
            backends.append(cv2.CAP_ANY)
        if allow_dshow:
            if prefer_dshow:
                return [cv2.CAP_DSHOW] + [b for b in backends if b != cv2.CAP_DSHOW]
            backends.append(cv2.CAP_DSHOW)
        return backends
    return [cv2.CAP_ANY]


def backend_code_to_name(backend: int) -> str:
    """Map OpenCV backend constants to readable names."""
    names = {
        cv2.CAP_ANY: "ANY",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_DSHOW: "DSHOW",
    }
    return names.get(backend, str(backend))


def detect_available_cameras(max_index: int = 4) -> List[dict]:
    """
    Detect available cameras and return their details including names.
    On Windows, tries multiple backends (DSHOW, MSMF) to find cameras.
    Added timeout to prevent system freezing.
    """
    cameras = []
    used_names = set()
    tested_indices = set()
    
    # Use platform-specific backend order (MSMF/ANY first on Windows).
    backends = get_camera_backends_for_platform(prefer_dshow=False)
    
    for backend in backends:
        # For all backends, probe 0..max_index.
        index_range = range(max_index + 1)
        
        for i in index_range:
            if i in tested_indices:
                continue
            
            cap = None
            try:
                # Try to open camera with this backend
                cap = cv2.VideoCapture(i, backend)
                
                if cap and cap.isOpened():
                    # Set buffer size to reduce latency
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    tested_indices.add(i)
                    
                    # Try to get the backend name
                    backend_name = cap.getBackendName() if cap else "unknown"
                    
                    # Create device name based on index and backend
                    if backend == cv2.CAP_DSHOW:
                        device_name = f"USB Camera ({i})" if i > 0 else "Default Camera"
                    elif backend == cv2.CAP_MSMF:
                        device_name = f"Webcam {i} (MSMF)"
                    else:
                        device_name = f"Camera {i}"
                    
                    # Create a unique name if needed
                    base_name = device_name
                    counter = 1
                    while device_name in used_names:
                        device_name = f"{base_name} #{counter}"
                        counter += 1
                    used_names.add(device_name)
                    
                    cameras.append({
                        "index": i,
                        "name": device_name,
                        "backend": backend_name
                    })
                    logger.info(f"Found camera at index {i} with backend {backend_name}")
            except Exception as e:
                logger.warning(f"Error probing camera {i} with backend {backend}: {e}")
            finally:
                # Always release camera handle to prevent resource leaks
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
    
    # Always include camera 0 as fallback even if not detected
    if not any(cam["index"] == 0 for cam in cameras):
        cameras.insert(0, {
            "index": 0,
            "name": "Camera 0 (Default)",
            "backend": "unknown"
        })
    
    logger.info(f"Detected {len(cameras)} cameras: {[c['name'] for c in cameras]}")
    return cameras


def _apply_camera_settings(cap: cv2.VideoCapture) -> None:
    """Apply stable, low-latency camera settings."""
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Favor stability/smoothness over resolution to avoid frame stalls on Windows drivers.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 24)


def open_camera():
    global active_camera, active_camera_index, preferred_camera_index, preferred_device_id

    # Avoid broad index probing here; probing itself can trigger backend warnings
    # on Windows drivers. Open directly with preferred/default index first.
    logger.info("Attempting to open camera...")

    # Priority: Use preferred_camera_index if set, otherwise DEFAULT_CAMERA_INDEX
    forced_cap = None
    camera_to_try = preferred_camera_index if preferred_camera_index is not None else DEFAULT_CAMERA_INDEX

    # Try selected index with platform backend order.
    for backend in get_camera_backends_for_platform(prefer_dshow=False):
        try:
            backend_label = backend_code_to_name(backend)
            logger.info(f"Trying to open camera index {camera_to_try} with backend {backend_label}...")
            forced_cap = cv2.VideoCapture(camera_to_try, backend)
            if forced_cap and forced_cap.isOpened():
                _apply_camera_settings(forced_cap)
                active_camera = forced_cap
                active_camera_index = camera_to_try
                backend_name = forced_cap.getBackendName() if forced_cap else "unknown"
                logger.info(f"SUCCESS: Camera opened at index={camera_to_try}, backend={backend_name}")
                return forced_cap
            elif forced_cap:
                forced_cap.release()
                forced_cap = None
        except Exception as e:
            logger.warning(f"Failed to open camera {camera_to_try} with backend {backend}: {e}")

    # If preferred_device_id is set, try using deviceId
    if preferred_device_id:
        try:
            logger.info(f"Trying to open camera with deviceId: {preferred_device_id}")
            cap = cv2.VideoCapture(preferred_device_id)
            if cap.isOpened():
                _apply_camera_settings(cap)
                active_camera_index = None
                logger.info(f"SUCCESS: Camera opened using deviceId: {preferred_device_id}")
                return cap
        except Exception as e:
            logger.warning(f"Failed to open camera with deviceId {preferred_device_id}: {e}")
    
    # Try minimal fallback indices only (preferred index, then index 0).
    candidate_indices = [camera_to_try]
    if 0 not in candidate_indices:
        candidate_indices.append(0)
    
    backends_to_try = get_camera_backends_for_platform(prefer_dshow=False)
    backend_names = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "ANY"}
    
    for i in candidate_indices:
        for backend in backends_to_try:
            try:
                logger.info(f"Trying camera index {i} with backend {backend_names.get(backend, 'unknown')}...")
                cap = cv2.VideoCapture(i, backend)
                if cap and cap.isOpened():
                    _apply_camera_settings(cap)
                    active_camera = cap
                    active_camera_index = i
                    backend_name = cap.getBackendName() if cap else "unknown"
                    logger.info(f"SUCCESS: Camera opened at index {i} with backend {backend_name}")
                    return cap
                elif cap:
                    cap.release()
            except Exception as e:
                logger.warning(f"Failed to open camera at index {i} with backend {backend_names.get(backend, 'unknown')}: {e}")
    
    logger.error("No camera could be opened after trying all methods")
    return None

# Shared variables for detection thread communication
detected_faces = []
detected_names = {}
detection_lock = threading.Lock()
pending_detection = False
detection_frame = None
detection_thread = None

def detection_worker():
    """Separate thread for face detection - runs asynchronously to not block video stream."""
    global detected_faces, detected_names, pending_detection, detection_frame, recognition_running, stop_streaming
    
    while recognition_running and not stop_streaming:
        # Check if there's a frame to process
        frame_to_process = None
        with detection_lock:
            if pending_detection and detection_frame is not None:
                frame_to_process = detection_frame.copy()
        if frame_to_process is None:
            time.sleep(0.01)
            continue
        
        # Process the frame for face detection
        if HAVE_FACE_RECOG and frame_to_process is not None:
            try:
                small_frame = cv2.resize(frame_to_process, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                faces_small = face_recognition.face_locations(rgb_small, model="hog")
                faces = [(int(top/FRAME_SCALE), int(right/FRAME_SCALE),
                         int(bottom/FRAME_SCALE), int(left/FRAME_SCALE))
                         for (top, right, bottom, left) in faces_small]
                
                names = {}
                
                if faces_small and len(known_face_encodings) > 0:
                    encs = face_recognition.face_encodings(rgb_small, faces_small, num_jitters=2)
                    
                    for face_idx, ((top, right, bottom, left), enc) in enumerate(zip(faces, encs)):
                        name = "Unknown"
                        student_id = None
                        dists = face_recognition.face_distance(known_face_encodings, enc)
                        
                        if len(dists) > 0:
                            # --- Voting: tally votes per unique student_id ---
                            vote_counts = {}
                            vote_min_dist = {}
                            for i, dist in enumerate(dists):
                                if dist <= FACE_MATCH_THRESHOLD:
                                    sid = known_face_ids[i]
                                    vote_counts[sid] = vote_counts.get(sid, 0) + 1
                                    if sid not in vote_min_dist or dist < vote_min_dist[sid]:
                                        vote_min_dist[sid] = dist

                            if vote_counts:
                                # Pick the student with the most votes (break ties by distance)
                                best_sid = max(vote_counts, key=lambda s: (vote_counts[s], -vote_min_dist[s]))
                                total_encodings_for_best = sum(1 for sid in known_face_ids if sid == best_sid)
                                vote_ratio = vote_counts[best_sid] / max(total_encodings_for_best, 1)

                                if vote_ratio >= RECOGNITION_VOTE_RATIO or vote_counts[best_sid] >= 2:
                                    # Find the index with the best distance for this student
                                    best_idx = min(
                                        (i for i, sid in enumerate(known_face_ids) if sid == best_sid),
                                        key=lambda i: dists[i]
                                    )
                                    name = known_face_names[best_idx]
                                    student_id = best_sid
                                    course = known_face_courses[best_idx]
                                    year = known_face_years[best_idx]
                                    first_name = known_face_first_names[best_idx]
                                    middle_name = known_face_middle_names[best_idx]
                                    last_name = known_face_last_names[best_idx]

                                    # --- Frame confirmation: require N consecutive matches ---
                                    if not hasattr(detection_worker, '_confirm_buffer'):
                                        detection_worker._confirm_buffer = {}
                                    buf = detection_worker._confirm_buffer
                                    prev = buf.get(face_idx)
                                    if prev and prev['student_id'] == student_id:
                                        buf[face_idx]['count'] = prev['count'] + 1
                                    else:
                                        buf[face_idx] = {'student_id': student_id, 'count': 1}

                                    if buf[face_idx]['count'] >= RECOGNITION_CONFIRM_FRAMES:
                                        if should_attempt_attendance(student_id, time.time(), ATTENDANCE_RETRY_COOLDOWN_SECONDS):
                                            record_attendance(student_id, name, course, year,
                                                            first_name, middle_name, last_name)
                                else:
                                    # Clear confirmation buffer if vote not confident enough
                                    if hasattr(detection_worker, '_confirm_buffer'):
                                        detection_worker._confirm_buffer.pop(face_idx, None)
                            else:
                                if hasattr(detection_worker, '_confirm_buffer'):
                                    detection_worker._confirm_buffer.pop(face_idx, None)
                        
                        names[face_idx] = name
                
                # Update shared variables
                with detection_lock:
                    detected_faces = faces
                    detected_names = names
                    pending_detection = False
            except Exception as e:
                logger.warning(f"Detection error: {e}")
                with detection_lock:
                    pending_detection = False
        else:
            time.sleep(0.01)


# === RECOGNITION THREAD === #
def recognition_loop():
    """Optimized recognition loop with minimal lag - detection runs in separate thread."""
    global latest_frame, recognition_running, stop_streaming, detected_faces, detected_names, pending_detection, detection_frame
    global latest_frame, recognition_running, stop_streaming

    logger.info("ðŸŽ¬ Starting recognition loop...")
    cap = open_camera()
    if not cap:
        logger.error("âŒ Failed to open camera")
        recognition_running = False
        return

    frame_count = 0
    last_faces = []  # Cache face locations
    last_names = {}  # Cache recognized names
    
    # For frame rate limiting - control output frame rate
    frame_interval = 1.0 / FRAME_RATE_LIMIT if FRAME_RATE_LIMIT > 0 else 0
    last_frame_time = time.time()

    while recognition_running and not stop_streaming:
        # Frame rate limiting - control how fast frames are processed/sent
        if frame_interval > 0:
            elapsed = time.time() - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        last_frame_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        # Adaptive detection cadence:
        # - Faster while faces are present (accuracy/tracking)
        # - Slower while idle (smoothness/CPU)
        detect_every_n = PROCESS_EVERY_N_FRAMES_ACTIVE if last_faces else PROCESS_EVERY_N_FRAMES_IDLE
        if HAVE_FACE_RECOG and frame_count % detect_every_n == 0:
            # Non-blocking detection lock - skip if lock is held
            if detection_lock.acquire(blocking=False):
                try:
                    if not pending_detection:
                        detection_frame = frame  # No copy needed here, worker will copy
                        pending_detection = True
                finally:
                    detection_lock.release()
        
        # Get latest detection results (non-blocking)
        with detection_lock:
            last_faces = detected_faces.copy()
            last_names = detected_names.copy()

        # Draw faces every frame using cached locations (smooth visualization)
        for face_idx, (top, right, bottom, left) in enumerate(last_faces):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            name = last_names.get(face_idx, "")
            if name:
                cv2.putText(frame, name, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_count += 1

        # Encode and update frame
        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        with latest_frame_lock:
            latest_frame = jpeg.tobytes()
            latest_frame_cond.notify_all()

    release_camera()
    logger.info("ðŸ›‘ Recognition stopped")


# === STREAM === #
def frame_stream():
    last_frame = None
    consecutive_empty_count = 0
    max_consecutive_empty = 50  # Max ~5 seconds of empty before forcing a frame
    
    while recognition_running:
        with latest_frame_cond:
            # Wait for new frame with timeout
            result = latest_frame_cond.wait(timeout=0.1)
            
            if latest_frame:
                last_frame = latest_frame
                consecutive_empty_count = 0
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame + b"\r\n"
            elif last_frame:
                # Send the last known frame to keep the stream alive
                consecutive_empty_count += 1
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last_frame + b"\r\n"
                
                # If we've sent many consecutive empty frames, try to get a fresh frame
                if consecutive_empty_count >= max_consecutive_empty:
                    # Force a small delay to allow camera to recover
                    time.sleep(0.05)
                    consecutive_empty_count = 0
            else:
                # No frame yet - send a placeholder or wait
                consecutive_empty_count += 1
                if consecutive_empty_count >= 10:  # After 1 second, try to restart camera
                    # This helps if camera initialization failed
                    logger.warning("No frame received for extended period, stream may be initializing...")


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
    """Start face recognition."""
    global recognition_running, stop_streaming, latest_frame
    global detected_faces, detected_names, pending_detection, detection_frame, detection_thread
    
    if recognition_running:
        return {"status": "already_running"}

    logger.info("🚀 Starting face recognition...")
    recognition_running = True
    stop_streaming = False
    with latest_frame_lock:
        latest_frame = None
    with detection_lock:
        detected_faces = []
        detected_names = {}
        pending_detection = False
        detection_frame = None
    with attendance_attempt_lock:
        last_attendance_attempt_by_student.clear()

    try:
        thread = threading.Thread(target=recognition_loop, daemon=True)
        thread.start()
        if HAVE_FACE_RECOG:
            detection_thread = threading.Thread(target=detection_worker, daemon=True)
            detection_thread.start()
        else:
            logger.warning("face_recognition is not available; live stream will run without detection/recognition")
        logger.info("✅ Recognition thread started")
    except Exception as e:
        logger.error(f"❌ Failed to start recognition thread: {e}")
        recognition_running = False
        return {"status": "failed", "error": str(e)}

    return {"status": "started"}


@app.post("/stop")
def stop():
    global recognition_running, stop_streaming, latest_frame
    global monitoring_mode, monitoring_session_id, monitoring_start_time, monitoring_previous_class_id, monitoring_students_pending
    recognition_running = False
    stop_streaming = True
    with latest_frame_lock:
        latest_frame = None
    with monitoring_lock:
        monitoring_mode = False
        monitoring_session_id = None
        monitoring_start_time = None
        monitoring_previous_class_id = None
        monitoring_students_pending = []
    with attendance_attempt_lock:
        last_attendance_attempt_by_student.clear()
    release_camera()
    return {"status": "stopped"}


@app.post("/reload_faces")
async def reload_faces():
    await load_faces_from_db()
    return {"status": "reloaded from database"}

@app.post("/set-mode")
async def set_mode(mode_data: dict):
    """Set the current recognition mode."""
    global current_mode, current_event_id, current_class_id, current_class_schedule
    global monitoring_mode, monitoring_session_id, monitoring_start_time, monitoring_previous_class_id, monitoring_students_pending
    mode = mode_data.get("mode")
    event_id = mode_data.get("event_id")

    if mode not in ["class", "events", "hallway"]:
        raise HTTPException(status_code=400, detail="Invalid mode")

    if mode == "class":
        # For class mode, event_id is actually class_id
        class_id = event_id
        if not class_id:
            raise HTTPException(status_code=400, detail="class_id is required for class mode")

        # Check if class exists
        class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
        if not class_doc:
            raise HTTPException(status_code=404, detail="Class not found")

        # Removed: Check if class is scheduled today - now allows any day
        # if not is_class_scheduled_today(class_doc.get("schedule", "")):
        #     raise HTTPException(status_code=403, detail="This class is not scheduled for today")

        current_class_id = class_id
        current_class_schedule = class_doc.get("schedule", "")
        current_event_id = None
    elif mode == "events":
        # For events mode
        if not event_id:
            raise HTTPException(status_code=400, detail="event_id is required for events mode")
        try:
            event_object_id = ObjectId(event_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid event_id format")

        event_doc = await db.events.find_one({"_id": event_object_id})
        if not event_doc:
            raise HTTPException(status_code=404, detail="Event not found")
        await mark_event_absences_if_ended(event_id, event_doc)
        current_class_id = None
        current_class_schedule = None
        current_event_id = event_id
        with monitoring_lock:
            monitoring_mode = False
            monitoring_session_id = None
            monitoring_start_time = None
            monitoring_previous_class_id = None
            monitoring_students_pending = []
    else:
        current_class_id = None
        current_class_schedule = None
        current_event_id = None
        with monitoring_lock:
            monitoring_mode = False
            monitoring_session_id = None
            monitoring_start_time = None
            monitoring_previous_class_id = None
            monitoring_students_pending = []

    current_mode = mode
    return {"message": f"Mode set to {mode}", "event_id": event_id}




@app.get("/status")
def get_status():
    """Get current system status."""
    with monitoring_lock:
        active_monitoring = {
            "enabled": monitoring_mode,
            "session_id": monitoring_session_id,
            "started_at": monitoring_start_time.strftime("%Y-%m-%d %H:%M:%S") if monitoring_start_time else None,
            "previous_class_id": monitoring_previous_class_id,
            "pending_count": len(monitoring_students_pending)
        }

    return {
        "status": "running" if recognition_running else "stopped",
        "recognition_running": recognition_running,
        "camera_active": active_camera is not None,
        "faces_loaded": len(known_face_names),
        "current_mode": current_mode,
        "current_event_id": current_event_id,
        "current_class_id": current_class_id,
        "monitoring": active_monitoring,
        "attendance_mode": attendance_mode
    }


@app.post("/attendance-mode")
def set_attendance_mode(payload: dict):
    """Set the attendance mode (IN or OUT)."""
    global attendance_mode
    mode = payload.get("mode", "IN").upper()
    if mode not in ["IN", "OUT"]:
        raise HTTPException(status_code=400, detail="Mode must be IN or OUT")
    attendance_mode = mode
    logger.info(f"Attendance mode set to: {mode}")
    return {"mode": attendance_mode, "message": f"Attendance mode set to {mode}"}


@app.get("/attendance-mode")
def get_attendance_mode():
    """Get the current attendance mode."""
    return {"mode": attendance_mode}


@app.get("/camera_status")
def camera_status():
    """Get camera status and frame availability."""
    return {
        "camera_active": active_camera is not None,
        "active_camera_index": active_camera_index,
        "preferred_camera_index": preferred_camera_index,
        "has_frame": latest_frame is not None,
        "recognition_running": recognition_running
    }


@app.get("/cameras")
def list_cameras():
    """List available cameras with their details."""
    cameras = detect_available_cameras()
    # Convert to frontend format
    camera_list = [{"index": cam["index"], "label": cam["name"]} for cam in cameras]
    return {
        "cameras": camera_list,
        "active_camera_index": active_camera_index,
        "preferred_camera_index": preferred_camera_index
    }


@app.post("/camera/select")
def select_camera(payload: dict):
    """Set preferred camera by camera index for the next recognition start."""
    global preferred_camera_index, preferred_device_id
    
    camera_index = payload.get("camera_index")
    camera_name = payload.get("camera_name")

    # If camera_name is provided, try to match it to a camera index
    if camera_name:
        available_cameras = detect_available_cameras()
        matched_camera = None
        for cam in available_cameras:
            if cam["name"] == camera_name:
                matched_camera = cam
                break
        
        if matched_camera:
            preferred_camera_index = matched_camera["index"]
            preferred_device_id = None
            return {
                "message": f"Preferred camera set to: {camera_name} (index {preferred_camera_index})",
                "camera_index": preferred_camera_index,
                "camera_name": camera_name,
                "preferred_camera_index": preferred_camera_index,
                "restart_required": bool(recognition_running)
            }
        else:
            raise HTTPException(status_code=400, detail=f"Camera '{camera_name}' not found")
    
    # If camera_index is provided, use it directly
    if camera_index is not None:
        try:
            camera_index = int(camera_index)
        except Exception:
            raise HTTPException(status_code=400, detail="camera_index must be an integer")

        # Validate camera index is in reasonable range (0-10)
        if camera_index < 0 or camera_index > 10:
            raise HTTPException(status_code=400, detail="camera_index must be between 0 and 10")

        preferred_camera_index = camera_index
        preferred_device_id = None
        return {
            "message": f"Preferred camera set to index {camera_index}",
            "camera_index": camera_index,
            "preferred_camera_index": preferred_camera_index,
            "restart_required": bool(recognition_running)
        }
    
    # If both are None, clear preferences
    preferred_camera_index = None
    preferred_device_id = None
    return {"message": "Camera preference cleared", "preferred_camera_index": None, "camera_index": None}


@app.post("/camera/test")
def test_camera(payload: dict):
    """
    Test opening a specific camera index.
    This endpoint allows manual testing of camera indices to find the USB camera.
    Tries multiple backends on Windows to work around MSMF issues.
    """
    camera_index = payload.get("camera_index")
    
    if camera_index is None:
        raise HTTPException(status_code=400, detail="camera_index is required")
    
    try:
        camera_index = int(camera_index)
    except Exception:
        raise HTTPException(status_code=400, detail="camera_index must be an integer")
    
    # Try different backends with platform-appropriate priority.
    backends_to_try = get_camera_backends_for_platform(prefer_dshow=False)
    backend_names = {cv2.CAP_ANY: "CAP_ANY", cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF"}
    
    cap = None
    for backend in backends_to_try:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            
            if cap and cap.isOpened():
                _apply_camera_settings(cap)
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Success! Get some info
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    backend_name = cap.getBackendName()
                    
                    return {
                        "success": True,
                        "camera_index": camera_index,
                        "message": f"Camera {camera_index} is working!",
                        "info": {
                            "resolution": f"{int(width)}x{int(height)}",
                            "backend": backend_name
                        }
                    }
                else:
                    # Frame read failed, but camera opened - try next backend
                    cap.release()
                    cap = None
                    continue
            else:
                # Couldn't open with this backend, try next
                if cap:
                    cap.release()
                    cap = None
                continue
        except Exception as e:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = None
            continue
    
    return {
        "success": False,
        "camera_index": camera_index,
        "message": f"Camera {camera_index} could not be opened with any backend. This is a Windows/OpenCV limitation - try using deviceId instead of index."
    }


@app.get("/attendance")
def get_attendance():
    """Get current attendance records."""
    with attendance_lock:
        return {"attendance": attendance_records.copy()}

@app.get("/recently-recognized")
def get_recently_recognized():
    """Get the most recently recognized student with database attendance status for class mode."""
    with recently_recognized_lock:
        result = recently_recognized
    
    # If in class mode and we have a recognized student, fetch their database attendance status
    if result and current_mode == "class" and current_class_id:
        student_id = result.get("student_id")
        if student_id and db:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    today = datetime.now().strftime("%Y-%m-%d")
                    db_record = loop.run_until_complete(
                        db.attendance.find_one({
                            "student_id": student_id,
                            "class_id": current_class_id,
                            "date": today
                        })
                    )
                    if db_record:
                        # Add database status to the result
                        result = {
                            **result,
                            "db_status": db_record.get("status"),
                            "db_attendance_type": db_record.get("attendance_type"),
                            "db_timestamp": db_record.get("timestamp")
                        }
                finally:
                    loop.close()
            except Exception as e:
                logger.warning(f"Error fetching database attendance status: {e}")
    
    return {"recently_recognized": result}

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
        logger.error(f"âŒ Failed to fetch attendance from DB: {e}")
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
@app.options("/auth/login")
async def login_options():
    return {"message": "OK"}

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
            logger.info(f"âœ… Teacher login successful for {username}: first_name='{first_name}', last_name='{last_name}'")
            return {"access_token": access_token, "token_type": "bearer", "first_name": first_name, "last_name": last_name, "user_id": teacher.get("teacher_id")}

        # Check teachers collection by teacher_id (for teachers logging in with teacher_id)
        if not '@' in username and username.isdigit() and len(username) == 6:
            teacher = await db.teachers.find_one({"teacher_id": username})
            if teacher and verify_password(password, teacher.get("hashed_password", "")):
                access_token = create_access_token(data={"sub": username, "role": "teacher"})
                first_name = teacher.get('first_name', '')
                last_name = teacher.get('last_name', '')
                logger.info(f"âœ… Teacher login successful for {username}: first_name='{first_name}', last_name='{last_name}'")
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

# Face Training API Endpoints
@app.get("/students/{student_id}/face-training-status")
async def get_face_training_status(student_id: str):
    """
    Get the face training status for a student.
    Returns which positions have been captured and if training is complete.
    """
    # Check if student exists
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Get face training positions
    positions = get_face_training_positions(student_id)
    
    return {
        "student_id": student_id,
        "positions": positions,
        "completed": positions.get('completed', False)
    }


@app.post("/students/{student_id}/validate-registration")
async def validate_student_registration(student_id: str):
    """
    Validate if a student can complete registration based on face training completion.
    Returns success if face training is complete (all 5 positions captured).
    Returns error if face training is incomplete.
    """
    # Check if student exists
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Get face training positions
    positions = get_face_training_positions(student_id)
    
    if not positions.get('completed', False):
        # Get missing positions
        missing_positions = []
        for pos in REQUIRED_FACE_POSITIONS:
            if not positions.get(pos, False):
                missing_positions.append(pos)
        
        return {
            "valid": False,
            "message": "Face training incomplete. Please capture all required face positions.",
            "missing_positions": missing_positions,
            "captured_positions": {k: v for k, v in positions.items() if k != 'completed'}
        }
    
    return {
        "valid": True,
        "message": "Face training complete. Registration can proceed."
    }


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
    # Hash password if provided
    if "password" in student_update:
        student_update["hashed_password"] = get_password_hash(student_update.pop("password"))

    result = await db.students.update_one(
        {"student_id": student_id},
        {"$set": student_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Student updated successfully"}

@app.post("/students/{student_id}/face-encodings")
async def save_face_encodings(
    student_id: str, 
    image: UploadFile = File(...),
    position: Optional[str] = Form(None)
):
    """
    Process uploaded image and save face encodings for a student.
    Optionally accepts a 'position' parameter to track face position (front, left, right, up, down).
    
    This endpoint can be used either:
    1. After student is created (student exists in DB)
    2. Before student is created (for pre-registration face training)
    
    When used for pre-registration, images are saved to disk but not to DB yet.
    """
    # Check if student exists (optional - allow pre-registration uploads)
    student = await db.students.find_one({"student_id": student_id})
    is_preregistration = student is None

    # Validate file
    if not image.filename or not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG, JPEG, and PNG are allowed.")

    # Validate position if provided
    valid_positions = ['front', 'left', 'right', 'up', 'down', 'center']
    if position and position.lower() not in valid_positions:
        # Normalize 'center' to 'front' for consistency
        if position.lower() == 'center':
            position = 'front'
        else:
            raise HTTPException(status_code=400, detail=f"Invalid position. Must be one of: {', '.join(valid_positions)}")
    
    # Normalize position
    if position == 'center':
        position = 'front'

    try:
        # Read image data
        image_data = await image.read()

        # Save image to storage with position prefix (works for both pre-reg and post-reg)
        image_path = save_image_to_storage(student_id, image_data, position)

        # Try to detect and encode face (optional - may fail during pre-registration)
        # We still save the image even if face detection fails, as long as the image is valid
        encoding_list = None
        success, result = detect_and_encode_face(image_path)
        
        if success and student is not None:
            # Student exists - save encoding to DB
            encoding_list = result.tolist()
            existing_encodings = student.get("face_encodings", [])
            if not isinstance(existing_encodings, list):
                existing_encodings = []
            existing_encodings.append(encoding_list)
            
            await db.students.update_one(
                {"student_id": student_id},
                {"$set": {"face_encodings": existing_encodings}}
            )
            
            logger.info(f"âœ… Face encoding saved for student {student_id}")
            
            # Reload faces in recognition system
            await load_faces_from_db()
            
            return {"message": "Face encoding saved successfully", "total_encodings": len(existing_encodings)}
        elif success:
            # Pre-registration - just return success without saving to DB yet
            logger.info(f"âœ… Face image saved for pre-registration student {student_id} (position: {position})")
            return {
                "message": "Face image saved for pre-registration. Student creation will validate completion.",
                "total_encodings": 0,
                "preregistration": True,
                "position": position
            }
        else:
            # Face detection failed - but image was saved
            logger.warning(f"Face detection failed for pre-registration {student_id}: {result}")
            return {
                "message": f"Image saved but face detection failed: {result}. Please try with a clearer photo.",
                "total_encodings": 0,
                "preregistration": True,
                "position": position,
                "face_detected": False
            }

    except Exception as e:
        logger.error(f"âŒ Error processing face encoding for {student_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.delete("/students/{student_id}")
async def delete_student(student_id: str):
    """Delete a student and all associated face data."""
    # First check if student exists
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Track what was deleted for response
    deleted_items = []
    
    # 1. Delete student's face images folder
    student_images_folder = os.path.join(FACE_DATA_DIR, str(student_id))
    if os.path.exists(student_images_folder):
        try:
            import shutil
            shutil.rmtree(student_images_folder)
            deleted_items.append("face images")
            logger.info(f"✅ Deleted face images folder: {student_images_folder}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete face images folder: {e}")
    else:
        logger.info(f"ℹ️ No face images folder found for student {student_id}")
    
    # 2. Delete pickle encoding file
    encoding_file = os.path.join(ENCODINGS_DIR, f"{student_id}.pkl")
    if os.path.exists(encoding_file):
        try:
            os.remove(encoding_file)
            deleted_items.append("face encodings file")
            logger.info(f"✅ Deleted encoding file: {encoding_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete encoding file: {e}")
    else:
        logger.info(f"ℹ️ No encoding file found for student {student_id}")
    
    # 3. Delete from MongoDB (this also removes face_encodings from the document)
    result = await db.students.delete_one({"student_id": student_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    
    deleted_items.append("database record")
    
    # 4. Reload face encodings to update in-memory data
    await load_faces_from_db()
    logger.info(f"✅ Face encodings reloaded after deleting student {student_id}")
    
    return {
        "message": "Student deleted successfully",
        "deleted_items": deleted_items,
        "student_id": student_id
    }


@app.get("/api/student/schedule/{student_id}")
async def get_student_schedule(student_id: str):
    """Get schedule for a specific student."""
    try:
        # Find all classes where the student is enrolled
        classes = []
        async for class_doc in db.classes.find({"enrolled_students": student_id}):
            classes.append(class_doc)

        if not classes:
            return {"schedule": []}

        schedule_data = []

        for class_doc in classes:
            schedule_str = class_doc.get("schedule", "")
            if not schedule_str:
                continue

            # Parse schedule string like "MWF 9:00-10:00"
            parts = schedule_str.split()
            if len(parts) < 2:
                continue

            days_str = parts[0]
            time_range = parts[1]

            # Split time range
            if '-' not in time_range:
                continue
            start_time, end_time = time_range.split('-', 1)

            # Parse days (same order as is_class_scheduled_today function)
            possible_days = ['Su', 'Th', 'M', 'T', 'W', 'F', 'S']
            day_names = ['Sunday', 'Thursday', 'Monday', 'Tuesday', 'Wednesday', 'Friday', 'Saturday']

            # Parse days_str into individual day codes
            scheduled_days = []
            i = 0
            while i < len(days_str):
                found = False
                for day in possible_days:
                    if days_str.startswith(day, i):
                        scheduled_days.append(day)
                        i += len(day)
                        found = True
                        break
                if not found:
                    i += 1

            # Get teacher information
            teacher_id = class_doc.get("teacher_id", "")
            teacher = await db.teachers.find_one({"teacher_id": teacher_id})
            instructor_name = "Unknown"
            if teacher:
                instructor_name = format_student_name(
                    teacher.get("first_name", ""),
                    teacher.get("middle_name", ""),
                    teacher.get("last_name", "")
                )

            # Create schedule entry for each day
            for day_code in scheduled_days:
                day_index = possible_days.index(day_code)
                day_name = day_names[day_index]

                schedule_data.append({
                    "subject_name": class_doc.get("class_name", "Unknown Subject"),
                    "subject_code": class_doc.get("class_code", ""),
                    "day": day_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "instructor": instructor_name,
                    "room": class_doc.get("room", "TBA")
                })

        return {"schedule": schedule_data}

    except Exception as e:
        logger.error(f"âŒ Error fetching schedule for student {student_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching schedule: {str(e)}")

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
        # Add accessibility flag based on schedule
        schedule = class_doc.get("schedule", "")
        class_doc["accessible"] = is_class_scheduled_today(schedule)
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
async def enroll_student(
    class_id: str,
    data: Optional[dict] = None,
    current_user: dict = Depends(require_roles(["admin", "teacher"]))
):
    """Enroll a student in a class."""
    payload = data or {}
    student_id = (payload.get("student_id") or "").strip()
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id is required")

    try:
        class_obj_id = ObjectId(class_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid class id")

    class_doc = await db.classes.find_one({"_id": class_obj_id})
    if not class_doc:
        raise HTTPException(status_code=404, detail="Class not found")

    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    if student_id not in class_doc.get("enrolled_students", []):
        await db.classes.update_one(
            {"_id": class_obj_id},
            {"$push": {"enrolled_students": student_id}}
        )

    return {"message": "Student enrolled successfully"}


@app.post("/classes/{class_id}/enroll-student")
async def enroll_single_student_to_class(
    class_id: str,
    data: Optional[dict] = None,
    current_user: dict = Depends(require_roles(["admin", "teacher"]))
):
    """
    Enroll a specific student to a class manually.
    This is for irregular students who are not following the full course subject list.
    Requires Admin or Teacher role.
    """
    payload = data or {}
    student_id = (payload.get("student_id") or "").strip()
    enrolled_by = (payload.get("enrolled_by") or current_user.get("sub") or "system").strip()
    
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id is required")

    try:
        class_obj_id = ObjectId(class_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid class id")

    # Validate class exists
    class_doc = await db.classes.find_one({"_id": class_obj_id})
    if not class_doc:
        raise HTTPException(status_code=404, detail="Class not found")

    # Validate student exists
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Check if already enrolled in this class (prevent duplicates)
    if student_id in class_doc.get("enrolled_students", []):
        return {
            "message": "Student already enrolled in this class",
            "already_enrolled": True
        }

    # Add student to class
    class_update = await db.classes.update_one(
        {"_id": class_obj_id},
        {"$addToSet": {"enrolled_students": student_id}}
    )
    if class_update.modified_count == 0:
        return {
            "message": "Student already enrolled in this class",
            "already_enrolled": True
        }
    
    # Also create an enrollment record in the enrollments collection
    try:
        enrollment_record = {
            "student_id": student_id,
            "class_id": class_id,
            "enrolled_by": enrolled_by,
            "enrolled_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "is_manual": True  # Flag to indicate this was manually added (irregular student)
        }
        await db.enrollments.update_one(
            {"student_id": student_id, "class_id": class_id},
            {"$setOnInsert": enrollment_record},
            upsert=True
        )
    except Exception as e:
        # If enrollment collection insert fails, still return success since class update worked
        logger.warning(f"Could not create enrollment record: {e}")

    return {
        "message": "Student successfully enrolled to class",
        "student_id": student_id,
        "class_id": class_id,
        "class_name": class_doc.get("class_name", "Unknown")
    }


@app.post("/enrollments/student/{student_id}/all-subjects")
async def enroll_student_to_all_subjects(student_id: str, data: Optional[dict] = None):
    """
    Enroll a student to all subjects/classes linked to their course/strand.
    This should only be triggered manually by admin, not automatically on student creation.
    
    The system will:
    1. Get the student's assigned course/strand
    2. Find all classes that are linked to that course/strand
    3. Enroll the student to all those classes
    4. Skip any classes where the student is already enrolled
    """
    payload = data or {}
    enrolled_by = (payload.get("enrolled_by") or "system").strip() or "system"
    
    # Get student information
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    student_course = (student.get("course") or "").strip()
    if not student_course:
        raise HTTPException(status_code=400, detail="Student has no course/strand assigned")

    # Find all classes linked to this course/strand (case-insensitive exact match)
    classes_for_course = []
    course_match = re.compile(f"^{re.escape(student_course)}$", re.IGNORECASE)
    async for cls in db.classes.find({"courses": {"$in": [course_match]}}):
        classes_for_course.append(cls)

    if not classes_for_course:
        return {
            "message": f"No classes found for course {student_course}",
            "student_id": student_id,
            "course": student_course,
            "enrolled_count": 0,
            "skipped_count": 0,
            "classes": []
        }

    # Enroll student to each class
    enrolled_count = 0
    skipped_count = 0
    enrollment_results = []

    for cls in classes_for_course:
        class_id = str(cls["_id"])
        class_name = cls.get("class_name", "Unknown")
        enrolled_students = cls.get("enrolled_students", [])

        # Check if already enrolled
        if student_id in enrolled_students:
            skipped_count += 1
            enrollment_results.append({
                "class_id": class_id,
                "class_name": class_name,
                "status": "skipped",
                "reason": "already enrolled"
            })
            continue

        # Enroll student in this class (duplicate-safe)
        class_update = await db.classes.update_one(
            {"_id": ObjectId(class_id)},
            {"$addToSet": {"enrolled_students": student_id}}
        )
        if class_update.modified_count == 0:
            skipped_count += 1
            enrollment_results.append({
                "class_id": class_id,
                "class_name": class_name,
                "status": "skipped",
                "reason": "already enrolled"
            })
            continue

        # Create enrollment record (duplicate-safe via unique index)
        try:
            enrollment_record = {
                "student_id": student_id,
                "class_id": class_id,
                "course": student_course,
                "enrolled_by": enrolled_by,
                "enrolled_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "is_manual": True,
                "is_bulk": True  # Flag to indicate this was bulk enrollment
            }
            await db.enrollments.update_one(
                {"student_id": student_id, "class_id": class_id},
                {"$setOnInsert": enrollment_record},
                upsert=True
            )
        except Exception as e:
            # Log but continue - class enrollment already succeeded.
            logger.warning(f"Could not create enrollment record for class {class_id}: {e}")

        enrolled_count += 1
        enrollment_results.append({
            "class_id": class_id,
            "class_name": class_name,
            "status": "enrolled"
        })

    return {
        "message": f"Enrolled student to {enrolled_count} subjects, skipped {skipped_count} (already enrolled)",
        "student_id": student_id,
        "student_name": format_student_name(
            student.get("first_name", ""),
            student.get("middle_name", ""),
            student.get("last_name", "")
        ),
        "course": student_course,
        "enrolled_count": enrolled_count,
        "skipped_count": skipped_count,
        "total_classes_found": len(classes_for_course),
        "enrollments": enrollment_results
    }


@app.get("/enrollments/student/{student_id}")
async def get_student_enrollments(student_id: str):
    """Get all enrollments for a specific student."""
    # First get student's course info
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Get classes where student is enrolled
    classes = []
    async for cls in db.classes.find({"enrolled_students": student_id}):
        classes.append({
            "class_id": str(cls["_id"]),
            "class_code": cls.get("class_code", ""),
            "class_name": cls.get("class_name", ""),
            "teacher_id": cls.get("teacher_id", ""),
            "schedule": cls.get("schedule", ""),
            "room": cls.get("room", ""),
            "courses": cls.get("courses", [])
        })

    # Also get enrollment records from enrollments collection
    enrollment_records = []
    async for enrollment in db.enrollments.find({"student_id": student_id}):
        enrollment["_id"] = str(enrollment["_id"])
        enrollment_records.append(enrollment)

    return {
        "student_id": student_id,
        "student_name": format_student_name(
            student.get("first_name", ""),
            student.get("middle_name", ""),
            student.get("last_name", "")
        ),
        "course": student.get("course", ""),
        "year": student.get("year", ""),
        "enrolled_classes": classes,
        "enrollment_count": len(classes),
        "enrollment_records": enrollment_records
    }


@app.delete("/enrollments/{enrollment_id}")
async def delete_enrollment(enrollment_id: str, data: dict = {}):
    """
    Remove a student from a specific class/enrollment.
    Requires student_id and class_id to identify the enrollment.
    """
    student_id = data.get("student_id")
    class_id = data.get("class_id")

    if not student_id or not class_id:
        raise HTTPException(status_code=400, detail="student_id and class_id are required")

    # Remove from class's enrolled_students array
    result = await db.classes.update_one(
        {"_id": ObjectId(class_id)},
        {"$pull": {"enrolled_students": student_id}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Class not found")

    # Remove from enrollments collection
    await db.enrollments.delete_one({
        "student_id": student_id,
        "class_id": class_id
    })

    return {"message": "Enrollment removed successfully"}


@app.get("/classes/{class_id}/students")
async def get_class_students(class_id: str):
    """Get all students enrolled in a specific class."""
    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=404, detail="Class not found")

    enrolled_student_ids = class_doc.get("enrolled_students", [])
    
    # Get student details
    students = []
    for student_id in enrolled_student_ids:
        student = await db.students.find_one({"student_id": student_id})
        if student:
            students.append({
                "student_id": student["student_id"],
                "name": format_student_name(
                    student.get("first_name", ""),
                    student.get("middle_name", ""),
                    student.get("last_name", "")
                ),
                "course": student.get("course", ""),
                "year": student.get("year", ""),
                "email": student.get("email", "")
            })

    return {
        "class_id": class_id,
        "class_name": class_doc.get("class_name", ""),
        "students": students,
        "student_count": len(students)
    }


# === ATTENDANCE MANAGEMENT === #
@app.post("/attendance/initialize-inout")
async def initialize_inout_attendance():
    """
    Initialize in-out attendance tracking for hallway mode.
    This sets up the system to track time-in/time-out for students.
    """
    global attendance_mode
    with attendance_mode_lock:
        attendance_mode = "IN"  # Reset to IN mode when initializing
    
    logger.info("In-Out attendance initialized - mode set to IN")
    return {
        "message": "In-Out attendance tracking initialized",
        "mode": "IN"
    }


@app.get("/attendance/monitoring/check/{class_id}")
async def check_monitoring_candidate(class_id: str):
    """Check if a class can use monitoring mode based on recent consecutive class attendance."""
    candidate = await find_previous_consecutive_class(class_id, max_gap_minutes=15)
    if not candidate:
        return {"eligible": False}
    return {"eligible": True, **candidate}


@app.post("/attendance/monitoring/start")
async def start_monitoring_mode(payload: dict):
    """
    Start monitoring mode for a class.
    Required: class_id, previous_class_id
    Optional: fallback_status (ABSENT|NEEDS_MANUAL_CONFIRMATION)
    """
    class_id = (payload.get("class_id") or "").strip()
    previous_class_id = (payload.get("previous_class_id") or "").strip()
    fallback_status = (payload.get("fallback_status") or "ABSENT").strip().upper()

    if not class_id or not previous_class_id:
        raise HTTPException(status_code=400, detail="class_id and previous_class_id are required")
    if fallback_status not in {"ABSENT", "NEEDS_MANUAL_CONFIRMATION"}:
        raise HTTPException(status_code=400, detail="fallback_status must be ABSENT or NEEDS_MANUAL_CONFIRMATION")

    # Safety check: only allow when class is really consecutive today (<=15 min)
    candidate = await find_previous_consecutive_class(class_id, max_gap_minutes=15)
    if not candidate or candidate.get("previous_class_id") != previous_class_id:
        raise HTTPException(status_code=400, detail="Class is not eligible for monitoring mode")

    session_info = await start_monitoring_revalidation_session(class_id, previous_class_id, fallback_status=fallback_status)
    return {"message": "Monitoring mode activated", **session_info}


@app.post("/attendance/monitoring/standalone-start")
async def start_standalone_revalidation_mode(payload: dict):
    """
    Start standalone revalidation mode for a class WITHOUT requiring a previous consecutive class.
    All enrolled students will start as PENDING_REVALIDATION and need to scan again to confirm attendance.
    
    Required: class_id
    Optional: fallback_status (ABSENT|NEEDS_MANUAL_CONFIRMATION)
    """
    class_id = (payload.get("class_id") or "").strip()
    fallback_status = (payload.get("fallback_status") or "ABSENT").strip().upper()

    if not class_id:
        raise HTTPException(status_code=400, detail="class_id is required")
    if fallback_status not in {"ABSENT", "NEEDS_MANUAL_CONFIRMATION"}:
        raise HTTPException(status_code=400, detail="fallback_status must be ABSENT or NEEDS_MANUAL_CONFIRMATION")

    session_info = await start_standalone_revalidation_session(class_id, fallback_status=fallback_status)
    return {"message": "Standalone revalidation mode activated", **session_info}


@app.post("/attendance/monitoring/finalize")
async def finalize_monitoring_mode(payload: dict):
    """Manually finalize an active monitoring session."""
    session_id = (payload.get("monitoring_session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="monitoring_session_id is required")
    result = await finalize_monitoring_revalidation_session(session_id)
    return result


@app.get("/attendance/class/{class_id}/today")
async def get_today_class_attendance(class_id: str):
    """Get today's attendance records for a class from database."""
    today = datetime.now().strftime("%Y-%m-%d")
    records = []
    async for record in db.attendance.find({"class_id": class_id, "date": today}).sort("timestamp", -1):
        record["_id"] = str(record["_id"])
        records.append(record)
    return {"class_id": class_id, "date": today, "attendance": records}


@app.post("/attendance/initialize-class/{class_id}")
async def initialize_class_attendance(class_id: str):
    """Initialize attendance records for all enrolled students in a class as absent for today."""
    # Check if class exists
    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=404, detail="Class not found")

    enrolled_students = class_doc.get("enrolled_students", [])
    if not enrolled_students:
        return {"message": "No enrolled students found for this class"}

    current_date = datetime.now().strftime('%Y-%m-%d')
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    attendance_collection = db.attendance
    initialized_count = 0

    for student_id in enrolled_students:
        result = await attendance_collection.update_one(
            {"student_id": student_id, "class_id": class_id, "date": current_date},
            {
                "$setOnInsert": {
                    "student_id": student_id,
                    "class_id": class_id,
                    "date": current_date,
                    "status": "absent",
                    "timestamp": current_timestamp,
                    "subject": class_doc.get("class_name", "Unknown Subject"),
                    "mode": "class"
                }
            },
            upsert=True
        )
        if result.upserted_id:
            initialized_count += 1

    return {"message": f"Initialized attendance for {initialized_count} students"}

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
    current_date = current_time.strftime('%Y-%m-%d')
    update_result = await db.attendance.update_one(
        {"student_id": student_id, "class_id": class_id, "date": current_date},
        {
            "$set": {
                "check_in_time": current_time.strftime('%H:%M:%S'),
                "status": "present",
                "timestamp": current_time.strftime('%Y-%m-%d %H:%M:%S'),
                "mode": "class"
            },
            "$setOnInsert": {
                "student_id": student_id,
                "class_id": class_id,
                "date": current_date
            }
        },
        upsert=True
    )
    return {
        "message": "Check-in recorded",
        "attendance_id": str(update_result.upserted_id) if update_result.upserted_id else None
    }

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
    event_dict = event.dict()

    if event_dict.get("start_time") and event_dict.get("end_time"):
        start_dt = parse_event_datetime(event_dict["date"], event_dict["start_time"])
        end_dt = parse_event_datetime(event_dict["date"], event_dict["end_time"])
        if not start_dt or not end_dt:
            raise HTTPException(status_code=400, detail="Invalid event start_time/end_time format")
        if end_dt <= start_dt:
            raise HTTPException(status_code=400, detail="end_time must be after start_time")

    result = await db.events.insert_one(event_dict)
    return {"message": "Event created successfully", "event_id": str(result.inserted_id)}

@app.get("/events")
async def get_events():
    """Get all events."""
    events = []
    async for event in db.events.find():
        event["_id"] = str(event["_id"])
        events.append(event)
    return {"events": events}

@app.get("/events/today")
async def get_today_events():
    """Get events scheduled for today."""
    today = datetime.now().date()
    today_events = []

    async for event in db.events.find():
        event_date = parse_event_date(event.get("date", ""))
        if not event_date or event_date.date() != today:
            continue
        event["_id"] = str(event["_id"])
        today_events.append(event)

    today_events.sort(key=lambda e: e.get("start_time") or "23:59")
    return {"events": today_events}

@app.get("/events/{event_id}")
async def get_event(event_id: str):
    """Get event by ID."""
    try:
        event_obj_id = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid event id")

    event = await db.events.find_one({"_id": event_obj_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    event["_id"] = str(event["_id"])
    return event

@app.put("/events/{event_id}")
async def update_event(event_id: str, event_update: dict):
    """Update event information."""
    try:
        event_obj_id = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid event id")

    current_event = await db.events.find_one({"_id": event_obj_id})
    if not current_event:
        raise HTTPException(status_code=404, detail="Event not found")

    merged_event = {**current_event, **event_update}
    if merged_event.get("start_time") and merged_event.get("end_time"):
        start_dt = parse_event_datetime(merged_event.get("date"), merged_event.get("start_time"))
        end_dt = parse_event_datetime(merged_event.get("date"), merged_event.get("end_time"))
        if not start_dt or not end_dt:
            raise HTTPException(status_code=400, detail="Invalid event start_time/end_time format")
        if end_dt <= start_dt:
            raise HTTPException(status_code=400, detail="end_time must be after start_time")

    result = await db.events.update_one(
        {"_id": event_obj_id},
        {"$set": event_update}
    )
    return {"message": "Event updated successfully"}


@app.put("/admin/events/{event_id}/schedule")
async def update_event_schedule(event_id: str, schedule_update: EventScheduleUpdate):
    """Admin API: update event start/end time and grace period."""
    try:
        event_obj_id = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid event id")

    event_doc = await db.events.find_one({"_id": event_obj_id})
    if not event_doc:
        raise HTTPException(status_code=404, detail="Event not found")

    start_dt = parse_event_datetime(event_doc.get("date"), schedule_update.start_time)
    end_dt = parse_event_datetime(event_doc.get("date"), schedule_update.end_time)
    if not start_dt or not end_dt:
        raise HTTPException(status_code=400, detail="Invalid start_time/end_time format. Use HH:MM or HH:MM:SS")
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end_time must be after start_time")

    await db.events.update_one(
        {"_id": event_obj_id},
        {
            "$set": {
                "start_time": schedule_update.start_time,
                "end_time": schedule_update.end_time,
                "grace_period_minutes": schedule_update.grace_period_minutes
            }
        }
    )
    return {"message": "Event schedule updated successfully"}

@app.delete("/events/{event_id}")
async def delete_event(event_id: str):
    """Delete an event."""
    try:
        event_obj_id = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid event id")

    result = await db.events.delete_one({"_id": event_obj_id})
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

async def record_attendance_for_verified_receipt(receipt: dict, event_doc: dict) -> bool:
    """
    Automatically record attendance when a receipt is verified.
    Returns True if attendance was recorded, False otherwise.
    """
    global recently_recognized
    
    student_id = receipt.get("student_id")
    event_id = receipt.get("event_id")
    
    if not student_id or not event_id:
        return False
    
    # Get event details
    event_name = event_doc.get("name", "Unknown Event")
    event_date = event_doc.get("date", "")
    start_time = event_doc.get("start_time", "00:00")
    end_time = event_doc.get("end_time", "23:59")
    
    # Parse event datetime
    start_dt = parse_event_datetime(event_date, start_time)
    end_dt = parse_event_datetime(event_date, end_time)
    
    if not start_dt or not end_dt:
        logger.warning(f"Cannot record attendance: invalid event datetime for event {event_id}")
        return False
    
    now = datetime.now()
    
    # Determine attendance status based on when the receipt is verified
    grace_period = int(event_doc.get("grace_period_minutes", 15))
    
    if now < start_dt:
        # Event hasn't started yet - mark as present (they registered early)
        attendance_status = "present"
    elif now <= end_dt:
        # Event is ongoing - check if within grace period
        grace_end = start_dt + timedelta(minutes=grace_period)
        attendance_status = "present" if now <= grace_end else "late"
    else:
        # Event has ended - mark as present (they verified before event ended)
        attendance_status = "present"
    
    # Format time strings and determine attendance date
    if now <= end_dt:
        check_in_time = now.strftime("%H:%M:%S")
        time_str = now.strftime("%H:%M:%S")
        # Use event date for attendance record
        attendance_date = event_date
    else:
        # If event has ended, use event end time and event date
        check_in_time = end_time
        time_str = end_time
        # Use event date for attendance record
        attendance_date = event_date
    
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    
    # Get student info for attendance record
    student = await db.students.find_one({"student_id": student_id})
    course = student.get("course", "") if student else ""
    year = student.get("year", "") if student else ""
    student_name = format_student_name(
        student.get("first_name", "") if student else "",
        student.get("middle_name", "") if student else "",
        student.get("last_name", "") if student else ""
    ) if student else receipt.get("student_name", "Unknown")
    
    # Check if attendance already recorded (check both current date and event date for backward compatibility)
    existing_record = await db.attendance.find_one({
        "student_id": student_id,
        "event_id": event_id,
        "$or": [{"date": attendance_date}, {"date": current_date}]
    })
    
    if existing_record:
        logger.info(f"Attendance already exists for student {student_id} in event {event_id}")
        # Even if attendance already exists, update recently_recognized to show success status
        with recently_recognized_lock:
            recently_recognized = {
                "name": student_name,
                "student_id": student_id,
                "event_id": event_id,
                "time": check_in_time,
                "date": attendance_date,
                "course": course,
                "year": year,
                "status": existing_record.get("status", "present"),
                "message": "Attendance already recorded"
            }
        return False
    
    # Create attendance record
    attendance_record = {
        "student_id": student_id,
        "name": student_name,
        "event_id": event_id,
        "event_name": event_name,
        "mode": "events",
        "date": attendance_date,
        "check_in_time": check_in_time,
        "time": time_str,
        "timestamp": timestamp,
        "course": course,
        "year": year,
        "status": attendance_status,
        "receipt_verified": True,
        "verified_at": receipt.get("verified_at", timestamp),
        "verified_by_receipt": True  # Flag to indicate attendance was recorded via receipt verification
    }
    
    await db.attendance.insert_one(attendance_record)
    logger.info(f"✅ Auto-recorded attendance for student {student_id} in event {event_id} as {attendance_status}")
    
    # Update recently_recognized to show success status in the frontend GIF
    with recently_recognized_lock:
        recently_recognized = {
            "name": student_name,
            "student_id": student_id,
            "event_id": event_id,
            "time": check_in_time,
            "date": attendance_date,
            "course": course,
            "year": year,
            "status": attendance_status,
            "message": "Attendance verified via receipt"
        }
    
    return True


@app.put("/receipts/{receipt_id}/verify")
async def verify_receipt(receipt_id: str, verification_data: dict):
    """Verify or reject a receipt (admin only)."""
    status = verification_data.get("status")  # "verified" or "rejected"
    verified_by = verification_data.get("verified_by")

    if status not in ["verified", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    # Get the receipt first to check if it exists and to get event info
    receipt = await db.receipts.find_one({"_id": ObjectId(receipt_id)})
    if not receipt:
        raise HTTPException(status_code=404, detail="Receipt not found")
    
    # Store the old status to check if we're changing to verified
    old_status = receipt.get("status")
    
    # Update the receipt status
    verified_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result = await db.receipts.update_one(
        {"_id": ObjectId(receipt_id)},
        {"$set": {
            "status": status,
            "verified_at": verified_at,
            "verified_by": verified_by
        }}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Receipt not found")
    
    # If receipt is being verified for the first time, auto-record attendance
    if status == "verified" and old_status != "verified":
        try:
            # Get event details
            event_id = receipt.get("event_id")
            logger.info(f"Verifying receipt {receipt_id} for event_id: {event_id}")
            
            # Handle both string and ObjectId event_id
            try:
                event_doc = await db.events.find_one({"_id": ObjectId(event_id)})
            except:
                # If ObjectId conversion fails, try as string
                event_doc = await db.events.find_one({"event_id": event_id})
            
            if event_doc:
                logger.info(f"Event found: {event_doc.get('name')}, recording attendance...")
                # Add verified_at to receipt for the attendance record
                receipt["verified_at"] = verified_at
                result = await record_attendance_for_verified_receipt(receipt, event_doc)
                logger.info(f"Attendance recording result: {result}")
            else:
                logger.warning(f"Event not found for receipt {receipt_id} with event_id: {event_id}")
        except Exception as e:
            logger.error(f"Error auto-recording attendance for receipt {receipt_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't fail the verification if attendance recording fails

    return {"message": f"Receipt {status} successfully"}

@app.delete("/receipts/{receipt_id}")
async def delete_receipt(receipt_id: str):
    """Delete a receipt (admin only)."""
    result = await db.receipts.delete_one({"_id": ObjectId(receipt_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return {"message": "Receipt deleted successfully"}

@app.get("/students/{student_id}/payment-status/{event_id}")
async def get_payment_status(student_id: str, event_id: str):
    """Get payment status for a student for a specific event."""
    # First check if any receipt exists for this student and event
    receipt = await db.receipts.find_one({
        "student_id": student_id,
        "event_id": event_id
    })

    if receipt:
        status = receipt.get("status")
        event = await db.events.find_one({"_id": ObjectId(event_id)})
        
        if status == "verified":
            return {
                "paid": True,
                "receipt_status": "verified",
                "event_name": event["name"] if event else "Unknown Event",
                "verified_at": receipt["verified_at"]
            }
        elif status == "pending":
            return {
                "paid": False,
                "receipt_status": "pending",
                "event_name": event["name"] if event else "Unknown Event",
                "message": "Receipt submitted but not yet verified"
            }
        else:  # rejected
            return {
                "paid": False,
                "receipt_status": "rejected",
                "event_name": event["name"] if event else "Unknown Event",
                "message": "Receipt was rejected"
            }
    else:
        # No receipt found at all
        return {
            "paid": False,
            "receipt_status": "not_found",
            "message": "No receipt submitted for this event"
        }

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

@app.get("/analytics/student/{student_id}/insights")
async def get_student_attendance_insights(student_id: str):
    """Get comprehensive attendance insights for a student."""
    # Get total attendance summary - EXCLUDE event attendance (mode: "events")
    # Only count class/subject attendance for the main summary
    pipeline = [
        {"$match": {"student_id": student_id, "mode": {"$ne": "events"}}},
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
    else:
        status = "At Risk"

    # Get subject-based breakdown - EXCLUDE event attendance
    subject_pipeline = [
        {"$match": {"student_id": student_id, "mode": {"$ne": "events"}}},
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
            "subject": doc.get("subject", "Unknown Subject"),
            "attendance_percentage": round(doc.get("attendance_percentage", 0), 1),
            "present_count": doc.get("present_count", 0),
            "absent_count": doc.get("absent_count", 0)
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
                "date": doc.get("date", ""),
                "time": doc.get("time", ""),
                "subject": doc.get("subject", "Unknown Subject"),
                "result": doc.get("result", "Unknown")
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
        logger.error(f"âŒ Failed to list collections: {e}")
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
        logger.error(f"âŒ Failed to fetch data from {collection}: {e}")
        return JSONResponse({"error": f"Failed to fetch data from {collection}"}, status_code=500)


# === STARTUP === #
# Note: Database startup is handled above, this is for backward compatibility
# This event is now redundant but kept for compatibility


# === MAIN ENTRY === #
if __name__ == "__main__":
    import uvicorn
    logger.info(f"âœ… Server listening at http://{HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)