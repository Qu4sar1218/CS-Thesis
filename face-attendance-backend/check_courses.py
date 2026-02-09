import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check_courses():
    """Check courses in the database."""
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]
    courses_collection = db.courses

    print("Courses in database:")
    async for course in courses_collection.find().sort("level", 1).sort("code", 1):
        print(f"  {course['code']} - {course['name']} (level: {course['level']})")

    total = await courses_collection.count_documents({})
    print(f"Total courses: {total}")

    client.close()

if __name__ == "__main__":
    asyncio.run(check_courses())
