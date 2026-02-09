"""
Authentication routes.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer

from database.connection import get_database
from database.auth import authenticate_user, create_access_token, get_password_hash, verify_password
from models import UserCreate, Token
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()


@router.post("/register", response_model=Dict[str, str])
async def register_user(user: UserCreate) -> Dict[str, str]:
    """Register a new user."""
    db = get_database()

    existing_user = await db.users.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")

    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    user_dict.pop("password")

    result = await db.users.insert_one(user_dict)
    return {"message": "User created successfully", "user_id": str(result.inserted_id)}


@router.post("/login", response_model=Token)
async def login(user_credentials: Dict[str, str]) -> Token:
    """Login user and return access token."""
    username = user_credentials["username"]
    password = user_credentials["password"]

    logger.info(f"🔐 Login attempt for username: {username}")

    db = get_database()
    if db is None:
        logger.error("❌ Database connection is None")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database connection error")

    # Check if username contains '@' - treat as email for admin/teacher
    if '@' in username:
        logger.info("📧 Checking email-based login")
        # Check users collection by email (for admin)
        user = await db.users.find_one({"email": username})
        if user:
            logger.info(f"👤 Found user in users collection: {user.get('username')}")
            if verify_password(password, user.get("hashed_password", "")):
                logger.info("✅ Admin login successful")
                access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
                return {"access_token": access_token, "token_type": "bearer"}

        # Check teachers collection by email (for teachers)
        teacher = await db.teachers.find_one({"email": username})
        if teacher:
            logger.info(f"👨‍🏫 Found teacher in teachers collection: {teacher.get('teacher_id')}")
            if verify_password(password, teacher.get("hashed_password", "")):
                logger.info("✅ Teacher login successful")
                access_token = create_access_token(data={"sub": username, "role": "teacher"})
                first_name = teacher.get('first_name', '')
                last_name = teacher.get('last_name', '')
                return {"access_token": access_token, "token_type": "bearer", "first_name": first_name, "last_name": last_name, "user_id": teacher.get("teacher_id")}

    # Check if username is 6 digits
    elif username.isdigit() and len(username) == 6:
        logger.info("🔢 Checking 6-digit ID login")
        # If starts with '11', treat as student_id
        if username.startswith('11'):
            logger.info("🎓 Checking student login")
            student = await db.students.find_one({"student_id": username})
            if student:
                logger.info(f"👨‍🎓 Found student: {student.get('student_id')}")
                if verify_password(password, student.get("hashed_password", "")):
                    logger.info("✅ Student login successful")
                    access_token = create_access_token(data={"sub": username, "role": "student"})
                    from utils import format_student_name
                    full_name = format_student_name(
                        student.get('first_name', ''),
                        student.get('middle_name', ''),
                        student.get('last_name', '')
                    )
                    return {"access_token": access_token, "token_type": "bearer", "full_name": full_name, "course": student.get("course", ""), "year": student.get("year", ""), "user_id": student.get("student_id")}
        else:
            # Treat as teacher_id
            logger.info("👨‍🏫 Checking teacher login by ID")
            teacher = await db.teachers.find_one({"teacher_id": username})
            if teacher:
                logger.info(f"👨‍🏫 Found teacher: {teacher.get('teacher_id')}")
                if verify_password(password, teacher.get("hashed_password", "")):
                    logger.info("✅ Teacher login successful")
                    access_token = create_access_token(data={"sub": username, "role": "teacher"})
                    first_name = teacher.get('first_name', '')
                    last_name = teacher.get('last_name', '')
                    return {"access_token": access_token, "token_type": "bearer", "first_name": first_name, "last_name": last_name, "user_id": teacher.get("teacher_id")}

    else:
        logger.info("🔍 Checking username-based login")
        # Check teachers collection by username (for teachers with usernames like email without @, but unlikely)
        teacher = await db.teachers.find_one({"username": username})
        if teacher and verify_password(password, teacher.get("hashed_password", "")):
            logger.info("✅ Teacher login successful (username)")
            access_token = create_access_token(data={"sub": username, "role": "teacher"})
            first_name = teacher.get('first_name', '')
            last_name = teacher.get('last_name', '')
            return {"access_token": access_token, "token_type": "bearer", "first_name": first_name, "last_name": last_name}

        # Check users collection by username (for admin with username like 'admin')
        user = await db.users.find_one({"username": username})
        if user and verify_password(password, user["hashed_password"]):
            logger.info("✅ Admin login successful (username)")
            access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
            return {"access_token": access_token, "token_type": "bearer"}

    logger.warning(f"❌ Login failed for username: {username}")
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")



