"""
Face recognition utilities and operations.
"""
import os
import glob
import pickle
import threading
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import logging

try:
    import face_recognition
    HAVE_FACE_RECOG = True
except Exception as e:
    print(f"[WARN] face_recognition missing: {e}")
    HAVE_FACE_RECOG = False

from config import settings
from database.connection import get_database

logger = logging.getLogger(__name__)

# Global face data storage
known_face_encodings: List[np.ndarray] = []
known_face_names: List[str] = []
known_face_ids: List[str] = []
known_face_courses: List[str] = []
known_face_years: List[str] = []
encodings_lock = threading.Lock()

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_image_to_storage(student_id: str, image_data: bytes) -> str:
    """
    Save image locally for face training.

    Args:
        student_id: Unique student identifier
        image_data: Raw image bytes

    Returns:
        str: Path where image was saved
    """
    student_folder = os.path.join(settings.face_data_dir, str(student_id))
    os.makedirs(student_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(student_folder, f"{timestamp}.jpg")

    with open(image_path, "wb") as f:
        f.write(image_data)

    return image_path


def detect_and_encode_face(image_path: str) -> Tuple[bool, Any]:
    """
    Detect face in image and generate encoding.

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (success: bool, encoding: np.array or error_message: str)
    """
    if not HAVE_FACE_RECOG:
        return False, "Face recognition library not available"

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


async def load_faces_from_db() -> None:
    """Load saved face encodings from database."""
    global known_face_encodings, known_face_names, known_face_ids, known_face_courses, known_face_years

    logger.info("🔄 Loading saved face encodings from database...")

    with encodings_lock:
        known_face_encodings.clear()
        known_face_names.clear()
        known_face_ids.clear()
        known_face_courses.clear()
        known_face_years.clear()

        try:
            db = get_database()
            students_collection = db.students
            async for student in students_collection.find({"face_encodings": {"$exists": True, "$ne": []}}):
                student_id = student["student_id"]
                first_name = student.get("first_name", "")
                middle_name = student.get("middle_name", "")
                last_name = student.get("last_name", "")

                # Import here to avoid circular imports
                from utils import format_student_name

                full_name = format_student_name(first_name, middle_name, last_name)
                course = student.get("course", "Unknown")
                year = student.get("year", "Unknown")
                enc_list = student.get("face_encodings") or []
                loaded_count = 0
                for enc in enc_list:
                    if isinstance(enc, list) and len(enc) == 128:
                        known_face_encodings.append(np.array(enc))
                        known_face_names.append(full_name)
                        known_face_ids.append(student_id)
                        known_face_courses.append(course)
                        known_face_years.append(year)
                        loaded_count += 1
                    else:
                        logger.warning(f"Skipping invalid encoding for {student_id}: {type(enc)} len={len(enc) if isinstance(enc, list) else 'N/A'}")
                logger.info(f"✅ Loaded {loaded_count}/{len(enc_list)} encodings for: {full_name} ({student_id}) - Course: {course}, Year: {year}")
        except Exception as e:
            logger.error(f"❌ Failed to load face encodings from database: {e}")

    logger.info(f"✅ Loaded {len(known_face_encodings)} known face encodings total from database")


def load_faces_from_disk() -> None:
    """Fallback function to load from disk if needed."""
    global known_face_encodings, known_face_names, known_face_ids

    logger.info("🔄 Loading saved face encodings from disk (fallback)...")
    logger.info(f"📂 Looking for encodings in: {settings.encodings_dir}")

    encs = glob.glob(os.path.join(settings.encodings_dir, "*.pkl"))
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


def recognize_face(face_encoding: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Recognize a face from known encodings.

    Args:
        face_encoding: Face encoding to match

    Returns:
        dict: Recognition result with name, id, course, year or None if not recognized
    """
    if not known_face_encodings:
        return None

    with encodings_lock:
        try:
            # Calculate distances to all known faces
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(distances) == 0:
                return None

            # Find the best match
            idx = np.argmin(distances)
            if distances[idx] <= settings.face_match_threshold:
                return {
                    'name': known_face_names[idx],
                    'student_id': known_face_ids[idx],
                    'course': known_face_courses[idx],
                    'year': known_face_years[idx],
                    'confidence': 1 - distances[idx]  # Convert distance to confidence
                }
        except Exception as e:
            logger.error(f"❌ Error during face recognition: {e}")

    return None
