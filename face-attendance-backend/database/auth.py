"""
Authentication utilities for password hashing and JWT management.
"""
import os
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer
import jwt

from config import settings
from database.connection import get_database
from utils import format_student_name

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt", "sha256_crypt"], deprecated="auto")

# Security
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any]) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user and return user data if successful.

    Returns user dict with role information or None if authentication fails.
    """
    db = get_database()
    if db is None:
        logger.error("❌ Database connection is None")
        return None

    # Check if username contains '@' - treat as email for admin/teacher
    if '@' in username:
        logger.info("📧 Checking email-based login")
        # Check users collection by email (for admin)
        user = await db.users.find_one({"email": username})
        if user and verify_password(password, user["hashed_password"]):
            logger.info("✅ Admin login successful")
            return {"username": user["username"], "role": user["role"]}

        # Check teachers collection by email (for teachers)
        teacher = await db.teachers.find_one({"email": username})
        if teacher and verify_password(password, teacher.get("hashed_password", "")):
            logger.info("✅ Teacher login successful")
            return {
                "username": username,
                "role": "teacher",
                "first_name": teacher.get('first_name', ''),
                "last_name": teacher.get('last_name', ''),
                "user_id": teacher.get("teacher_id")
            }

    # Check if username is 6 digits
    elif username.isdigit() and len(username) == 6:
        logger.info("🔢 Checking 6-digit ID login")
        # If starts with '11', treat as student_id
        if username.startswith('11'):
            logger.info("🎓 Checking student login")
            student = await db.students.find_one({"student_id": username})
            if student and verify_password(password, student.get("hashed_password", "")):
                logger.info("✅ Student login successful")
                return {
                    "username": username,
                    "role": "student",
                    "full_name": format_student_name(
                        student.get('first_name', ''),
                        student.get('middle_name', ''),
                        student.get('last_name', '')
                    ),
                    "course": student.get("course", ""),
                    "year": student.get("year", ""),
                    "user_id": student.get("student_id")
                }
        else:
            # Treat as teacher_id
            logger.info("👨‍🏫 Checking teacher login by ID")
            teacher = await db.teachers.find_one({"teacher_id": username})
            if teacher and verify_password(password, teacher.get("hashed_password", "")):
                logger.info("✅ Teacher login successful")
                return {
                    "username": username,
                    "role": "teacher",
                    "first_name": teacher.get('first_name', ''),
                    "last_name": teacher.get('last_name', ''),
                    "user_id": teacher.get("teacher_id")
                }

    else:
        logger.info("🔍 Checking username-based login")
        # Check teachers collection by username (for teachers with usernames like email without @, but unlikely)
        teacher = await db.teachers.find_one({"username": username})
        if teacher and verify_password(password, teacher.get("hashed_password", "")):
            logger.info("✅ Teacher login successful (username)")
            return {
                "username": username,
                "role": "teacher",
                "first_name": teacher.get('first_name', ''),
                "last_name": teacher.get('last_name', '')
            }

        # Check users collection by username (for admin with username like 'admin')
        user = await db.users.find_one({"username": username})
        if user and verify_password(password, user["hashed_password"]):
            logger.info("✅ Admin login successful (username)")
            return {"username": user["username"], "role": user["role"]}

    logger.warning(f"❌ Login failed for username: {username}")
    return None


def generate_teacher_id() -> str:
    """Generate a unique 6-digit teacher ID."""
    return f"{random.randint(100000, 999999)}"
