"""
Student management routes.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from bson import ObjectId

from database.connection import get_database
from database.auth import get_password_hash
from models import Student, StudentCreate
from face_recognition import save_image_to_storage, detect_and_encode_face, allowed_file, load_faces_from_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/")
async def create_student(student: StudentCreate) -> Dict[str, Any]:
    """Create a new student."""
    db = get_database()

    existing_student = await db.students.find_one({"student_id": student.student_id})
    if existing_student:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Student ID already exists")

    # Set username as student_id and password as student_id (hashed)
    from database.auth import get_password_hash
    student_dict = student.dict()
    student_dict["username"] = student.student_id
    student_dict["hashed_password"] = get_password_hash(student.student_id)

    result = await db.students.insert_one(student_dict)
    return {"message": "Student created successfully", "student_id": str(result.inserted_id)}


@router.get("/")
async def get_students() -> Dict[str, List[Dict[str, Any]]]:
    """Get all students."""
    db = get_database()

    students = []
    async for student in db.students.find():
        student["_id"] = str(student["_id"])
        students.append(student)
    return {"students": students}


@router.get("/{student_id}")
async def get_student(student_id: str) -> Dict[str, Any]:
    """Get student by ID."""
    db = get_database()

    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    student["_id"] = str(student["_id"])
    return student


@router.put("/{student_id}")
async def update_student(student_id: str, student_update: Dict[str, Any]) -> Dict[str, str]:
    """Update student information."""
    db = get_database()

    # Hash password if provided
    if "password" in student_update:
        from database import get_password_hash
        student_update["hashed_password"] = get_password_hash(student_update.pop("password"))

    result = await db.students.update_one(
        {"student_id": student_id},
        {"$set": student_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return {"message": "Student updated successfully"}


@router.post("/{student_id}/face-encodings")
async def save_face_encodings(student_id: str, image: UploadFile = File(...)) -> Dict[str, Any]:
    """Process uploaded image and save face encodings for a student."""
    db = get_database()

    # Check if student exists
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    # Validate file
    if not image.filename or not allowed_file(image.filename):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                          detail="Invalid file type. Only JPG, JPEG, and PNG are allowed.")

    try:
        # Read image data
        image_data = await image.read()
        logger.info(f"📁 Read image data for student {student_id}, size: {len(image_data)} bytes")

        # Save image to storage
        image_path = save_image_to_storage(student_id, image_data)
        logger.info(f"💾 Saved image to {image_path}")

        # Detect and encode face
        success, result = detect_and_encode_face(image_path)
        logger.info(f"🔍 Face detection result for {student_id}: success={success}, result={result[:100] if isinstance(result, str) else type(result)}")

        if not success:
            # Log the error but keep the image for debugging
            logger.warning(f"Face detection failed for {student_id}: {result}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result)

        # Convert numpy array to list for MongoDB storage
        encoding_list = result.tolist()
        logger.info(f"📊 Generated encoding with {len(encoding_list)} features")

        # Get existing encodings or initialize empty list
        existing_encodings = student.get("face_encodings", [])
        if not isinstance(existing_encodings, list):
            existing_encodings = []

        # Add new encoding
        existing_encodings.append(encoding_list)

        # Update student with new encodings
        update_result = await db.students.update_one(
            {"student_id": student_id},
            {"$set": {"face_encodings": existing_encodings}}
        )

        logger.info(f"✅ Face encoding saved for student {student_id}, total encodings: {len(existing_encodings)}")

        # Reload faces in recognition system
        await load_faces_from_db()

        return {"message": "Face encoding saved successfully", "total_encodings": len(existing_encodings)}

    except Exception as e:
        logger.error(f"❌ Error processing face encoding for {student_id}: {str(e)}")
        # Change to 400 Bad Request since it's likely an invalid image
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                          detail=f"Error processing image: {str(e)}")


@router.delete("/{student_id}")
async def delete_student(student_id: str) -> Dict[str, str]:
    """Delete a student."""
    db = get_database()

    result = await db.students.delete_one({"student_id": student_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return {"message": "Student deleted successfully"}


@router.get("/api/student/schedule/{student_id}")
async def get_student_schedule(student_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get schedule for a specific student."""
    db = get_database()

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
                from utils import format_student_name
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
        logger.error(f"❌ Error fetching schedule for student {student_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail=f"Error fetching schedule: {str(e)}")


@router.get("/{student_id}/payment-status/{event_id}")
async def get_payment_status(student_id: str, event_id: str) -> Dict[str, Any]:
    """Get payment status for a student for a specific event."""
    db = get_database()

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
