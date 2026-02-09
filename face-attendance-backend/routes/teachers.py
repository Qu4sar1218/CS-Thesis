"""
Teacher management routes.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import random

from database.connection import get_database
from database.auth import get_password_hash
from models import Teacher, TeacherCreate

router = APIRouter()

@router.post("/", response_model=Teacher)
async def create_teacher(teacher: TeacherCreate):
    """Create a new teacher."""
    db = get_database()

    # Auto-generate 6-digit teacher ID
    while True:
        teacher_id = f"{random.randint(100000, 999999)}"
        existing_teacher = await db.teachers.find_one({"teacher_id": teacher_id})
        if not existing_teacher:
            break

    # Set username as email and password as teacher_id (hashed)
    teacher_dict = teacher.dict()
    teacher_dict["username"] = teacher.email
    teacher_dict["teacher_id"] = teacher_id
    teacher_dict["hashed_password"] = get_password_hash(teacher_id)

    result = await db.teachers.insert_one(teacher_dict)
    teacher_dict["_id"] = str(result.inserted_id)
    return Teacher(**teacher_dict)

@router.get("/")
async def get_all_teachers():
    """Get all teachers."""
    db = get_database()
    teachers = await db.teachers.find().to_list(1000)
    for teacher in teachers:
        teacher["_id"] = str(teacher["_id"])
    return {"teachers": [Teacher(**teacher) for teacher in teachers]}

@router.get("/{teacher_id}", response_model=Teacher)
async def get_teacher(teacher_id: str):
    """Get teacher by ID."""
    db = get_database()
    teacher = await db.teachers.find_one({"teacher_id": teacher_id})
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found")
    teacher["_id"] = str(teacher["_id"])
    return Teacher(**teacher)

@router.put("/{teacher_id}", response_model=Teacher)
async def update_teacher(teacher_id: str, teacher_update: TeacherCreate):
    """Update teacher information."""
    db = get_database()
    update_data = teacher_update.dict(exclude_unset=True)
    result = await db.teachers.update_one(
        {"teacher_id": teacher_id},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Teacher not found")
    updated_teacher = await db.teachers.find_one({"teacher_id": teacher_id})
    updated_teacher["_id"] = str(updated_teacher["_id"])
    return Teacher(**updated_teacher)

@router.delete("/{teacher_id}")
async def delete_teacher(teacher_id: str):
    """Delete a teacher."""
    db = get_database()
    result = await db.teachers.delete_one({"teacher_id": teacher_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Teacher not found")
    return {"message": "Teacher deleted successfully"}
