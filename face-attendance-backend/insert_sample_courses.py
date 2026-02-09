import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def insert_sample_courses():
    """Insert sample courses into the database."""
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]
    courses_collection = db.courses

    # Sample senior high strands
    senior_high_courses = [
        {"code": "STEM", "name": "Science, Technology, Engineering, and Mathematics", "level": "senior_high"},
        {"code": "ABM", "name": "Accountancy, Business, and Management", "level": "senior_high"},
        {"code": "HUMSS", "name": "Humanities and Social Sciences", "level": "senior_high"},
        {"code": "GAS", "name": "General Academic Strand", "level": "senior_high"},
        {"code": "TVL-ICT", "name": "Technical-Vocational-Livelihood - Information and Communications Technology", "level": "senior_high"},
        {"code": "TVL-HE", "name": "Technical-Vocational-Livelihood - Home Economics", "level": "senior_high"},
        {"code": "ARTS", "name": "Arts and Design", "level": "senior_high"},
        {"code": "SPORTS", "name": "Sports Track", "level": "senior_high"}
    ]

    # Sample college courses
    college_courses = [
        {"code": "BSCS", "name": "Bachelor of Science in Computer Science", "level": "college"},
        {"code": "BSIT", "name": "Bachelor of Science in Information Technology", "level": "college"},
        {"code": "BSECE", "name": "Bachelor of Science in Electronics and Communications Engineering", "level": "college"},
        {"code": "BSCE", "name": "Bachelor of Science in Civil Engineering", "level": "college"},
        {"code": "BSME", "name": "Bachelor of Science in Mechanical Engineering", "level": "college"},
        {"code": "BSEE", "name": "Bachelor of Science in Electrical Engineering", "level": "college"},
        {"code": "BSBA", "name": "Bachelor of Science in Business Administration", "level": "college"},
        {"code": "BSA", "name": "Bachelor of Science in Accountancy", "level": "college"},
        {"code": "BSN", "name": "Bachelor of Science in Nursing", "level": "college"},
        {"code": "BSEd", "name": "Bachelor of Science in Education", "level": "college"},
        {"code": "ABCOMM", "name": "Bachelor of Arts in Communication", "level": "college"},
        {"code": "ABPSYCH", "name": "Bachelor of Arts in Psychology", "level": "college"}
    ]

    all_courses = senior_high_courses + college_courses

    # Insert courses if they don't already exist
    inserted_count = 0
    for course in all_courses:
        existing = await courses_collection.find_one({"code": course["code"]})
        if not existing:
            await courses_collection.insert_one(course)
            inserted_count += 1
            print(f"Inserted course: {course['code']} - {course['name']}")
        else:
            print(f"Course already exists: {course['code']}")

    print(f"Total courses inserted: {inserted_count}")
    print(f"Total courses in database: {await courses_collection.count_documents({})}")

    client.close()

if __name__ == "__main__":
    asyncio.run(insert_sample_courses())
