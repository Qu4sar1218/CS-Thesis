import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check_users():
    """Check users in database."""
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]

    print("Teachers:")
    async for t in db.teachers.find():
        print(f"  ID: {t.get('teacher_id')}, Email: {t.get('email')}, Hashed PW: {t.get('hashed_password')[:10]}...")

    print("\nStudents:")
    async for s in db.students.find():
        print(f"  ID: {s.get('student_id')}, Hashed PW: {s.get('hashed_password')[:10]}...")

    print("\nUsers (Admin):")
    async for u in db.users.find():
        print(f"  Username: {u.get('username')}, Email: {u.get('email')}, Role: {u.get('role')}")

    client.close()

if __name__ == "__main__":
    asyncio.run(check_users())
