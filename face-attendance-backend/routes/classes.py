"""
Class management routes.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from bson import ObjectId

from database.connection import get_database
from models import Class, ClassCreate
from utils import is_class_scheduled_today
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/")
async def create_class(class_data: ClassCreate) -> Dict[str, Any]:
    """Create a new class."""
    db = get_database()

    existing_class = await db.classes.find_one({"class_code": class_data.class_code})
    if existing_class:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Class code already exists")

    result = await db.classes.insert_one(class_data.dict())
    return {"message": "Class created successfully", "class_id": str(result.inserted_id)}


@router.get("/")
async def get_classes() -> Dict[str, List[Dict[str, Any]]]:
    """Get all classes."""
    db = get_database()

    classes = []
    async for class_doc in db.classes.find():
        class_doc["_id"] = str(class_doc["_id"])
        classes.append(class_doc)
    return {"classes": classes}


@router.get("/teacher/{teacher_id}")
async def get_classes_by_teacher(teacher_id: str) -> Dict[str, List[Dict[str, Any]]]:
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


@router.get("/courses")
async def get_courses() -> Dict[str, List[Dict[str, Any]]]:
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


@router.get("/{class_id}")
async def get_class(class_id: str) -> Dict[str, Any]:
    """Get class by ID."""
    db = get_database()

    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")
    class_doc["_id"] = str(class_doc["_id"])
    return class_doc


@router.put("/{class_id}")
async def update_class(class_id: str, class_update: Dict[str, Any]) -> Dict[str, str]:
    """Update class information."""
    db = get_database()

    result = await db.classes.update_one(
        {"_id": ObjectId(class_id)},
        {"$set": class_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")
    return {"message": "Class updated successfully"}


@router.delete("/{class_id}")
async def delete_class(class_id: str) -> Dict[str, str]:
    """Delete a class."""
    db = get_database()

    result = await db.classes.delete_one({"_id": ObjectId(class_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")
    return {"message": "Class deleted successfully"}


@router.post("/{class_id}/enroll")
async def enroll_student(class_id: str, data: Dict[str, str]) -> Dict[str, str]:
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
