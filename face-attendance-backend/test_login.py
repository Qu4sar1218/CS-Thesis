import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def test_login():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]

    print("Testing teacher login with email and teacher_id as password:")
    teacher = await db.teachers.find_one({"email": "DJ@gmail.com"})
    if teacher:
        teacher_id = teacher.get("teacher_id")
        hashed_pw = teacher.get("hashed_password")
        print(f"Teacher ID: {teacher_id}")
        print(f"Hashed PW: {hashed_pw}")
        print(f"Verify teacher_id as password: {verify_password(teacher_id, hashed_pw)}")

    print("\nTesting teacher login with teacher_id as username and password:")
    teacher = await db.teachers.find_one({"teacher_id": "716974"})
    if teacher:
        teacher_id = teacher.get("teacher_id")
        hashed_pw = teacher.get("hashed_password")
        print(f"Teacher ID: {teacher_id}")
        print(f"Hashed PW: {hashed_pw}")
        print(f"Verify teacher_id as password: {verify_password(teacher_id, hashed_pw)}")

    print("\nTesting student login:")
    student = await db.students.find_one({"student_id": "116653"})
    if student:
        student_id = student.get("student_id")
        hashed_pw = student.get("hashed_password")
        print(f"Student ID: {student_id}")
        print(f"Hashed PW: {hashed_pw}")
        print(f"Verify student_id as password: {verify_password(student_id, hashed_pw)}")

    client.close()

if __name__ == "__main__":
    asyncio.run(test_login())
