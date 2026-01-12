#!/usr/bin/env python3
"""
Script to populate the courses collection in MongoDB with predefined courses and strands.
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def populate_courses():
    """Populate the courses collection with predefined data."""
    # Connect to MongoDB
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]
    courses_collection = db.courses

    # Clear existing courses
    await courses_collection.delete_many({})

    # Define courses and strands
    courses_data = [
        # College courses
        {"code": "BSIT", "name": "Bachelor of Science in Information Technology", "level": "college"},
        {"code": "BSENTREP", "name": "Bachelor of Science in Entrepreneurship", "level": "college"},
        {"code": "BSOA", "name": "Bachelor of Science in Office Administration", "level": "college"},
        {"code": "BSBA", "name": "Bachelor of Science in Business Administration", "level": "college"},
        {"code": "Btvted", "name": "Bachelor of Technical-Vocational Teacher Education", "level": "college"},
        {"code": "BSCS", "name": "Bachelor of Science in Computer Science", "level": "college"},

        # Senior High School strands
        {"code": "GAS", "name": "General Academic Strand", "level": "senior_high"},
        {"code": "HUMSS", "name": "Humanities and Social Sciences", "level": "senior_high"},
        {"code": "STEM", "name": "Science, Technology, Engineering, and Mathematics", "level": "senior_high"},
        {"code": "ICT", "name": "Information and Communications Technology", "level": "senior_high"},
    ]

    # Insert courses
    result = await courses_collection.insert_many(courses_data)

    print(f"✅ Successfully inserted {len(result.inserted_ids)} courses into the database")

    # Verify insertion
    count = await courses_collection.count_documents({})
    print(f"📊 Total courses in database: {count}")

    # List all courses
    print("\n📋 Courses in database:")
    async for course in courses_collection.find().sort("level", 1).sort("code", 1):
        print(f"  {course['code']} - {course['name']} ({course['level']})")

    # Close connection
    client.close()

if __name__ == "__main__":
    asyncio.run(populate_courses())
