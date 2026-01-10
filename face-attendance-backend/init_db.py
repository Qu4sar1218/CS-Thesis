#!/usr/bin/env python3
"""
Database Initialization Script for Face Attendance System
Run this script to create all required collections in MongoDB Compass
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

def get_password_hash(password):
    """Hash a password."""
    return pwd_context.hash(password)

async def init_database():
    """Initialize the InterACTS database with required collections and sample data."""

    # Connect to MongoDB
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]

    print("🚀 Initializing Face Attendance Database...")

    # Create collections (MongoDB creates them automatically when first used, but we'll add sample data)

    # 1. Users Collection - Sample admin user
    users_collection = db.users
    sample_users = [
        {
            "username": "admin",
            "email": "admin@school.edu",
            "role": "admin",
            "hashed_password": get_password_hash("admin123")
        },
        {
            "username": "teacher1",
            "email": "teacher1@school.edu",
            "role": "teacher",
            "hashed_password": get_password_hash("admin123")
        }
    ]

    # Clear existing and insert sample users
    await users_collection.delete_many({})
    result = await users_collection.insert_many(sample_users)
    print(f"✅ Created users collection with {len(result.inserted_ids)} sample users")

    # 2. Teachers Collection - Sample teacher
    teachers_collection = db.teachers
    sample_teachers = [
        {
            "teacher_id": "T001",
            "first_name": "John",
            "last_name": "Smith",
            "email": "john.smith@school.edu",
            "department": "Computer Science"
        }
    ]

    await teachers_collection.delete_many({})
    result = await teachers_collection.insert_many(sample_teachers)
    print(f"✅ Created teachers collection with {len(result.inserted_ids)} sample teachers")

    # 3. Students Collection - Sample students
    students_collection = db.students
    sample_students = [
        {
            "student_id": "114001",
            "first_name": "Alice",
            "last_name": "Johnson",
            "email": "alice.johnson@student.edu",
            "course": "Computer Science",
            "year": "3rd Year",
            "face_encodings": [],  # Will be populated during face registration
            "hashed_password": get_password_hash("114001")
        },
        {
            "student_id": "114002",
            "first_name": "Bob",
            "last_name": "Wilson",
            "email": "bob.wilson@student.edu",
            "course": "Information Technology",
            "year": "2nd Year",
            "face_encodings": [],
            "hashed_password": get_password_hash("114002")
        }
    ]

    await students_collection.delete_many({})
    result = await students_collection.insert_many(sample_students)
    print(f"✅ Created students collection with {len(result.inserted_ids)} sample students")

    # 4. Classes Collection - Sample class
    classes_collection = db.classes
    sample_classes = [
        {
            "class_code": "CS101",
            "class_name": "Introduction to Programming",
            "teacher_id": "T001",
            "schedule": "MWF 9:00-10:00",
            "room": "Room 101",
            "enrolled_students": ["STU001", "STU002"]
        }
    ]

    await classes_collection.delete_many({})
    result = await classes_collection.insert_many(sample_classes)
    print(f"✅ Created classes collection with {len(result.inserted_ids)} sample classes")

    # 5. Attendance Collection - Sample attendance records
    attendance_collection = db.attendance
    sample_attendance = [
        {
            "student_id": "STU001",
            "class_id": str(result.inserted_ids[0]),  # Reference to the class we just created
            "date": datetime.now().strftime('%Y-%m-%d'),
            "check_in_time": "09:15:00",
            "check_out_time": "10:00:00",
            "status": "present",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    ]

    await attendance_collection.delete_many({})
    result = await attendance_collection.insert_many(sample_attendance)
    print(f"✅ Created attendance collection with {len(result.inserted_ids)} sample records")

    # 6. Events Collection - Sample events
    events_collection = db.events
    sample_events = [
        {
            "name": "Christmas Party",
            "description": "Annual Christmas celebration with games and food",
            "date": "2024-12-25",
            "location": "School Auditorium",
            "price": 500.00
        },
        {
            "name": "Sports Fest",
            "description": "Inter-class sports competition",
            "date": "2025-01-15",
            "location": "School Gymnasium",
            "price": 300.00
        },
        {
            "name": "Symposium",
            "description": "Educational symposium on technology",
            "date": "2025-02-10",
            "location": "Conference Hall",
            "price": 200.00
        },
        {
            "name": "Cultural Night",
            "description": "Celebration of diverse cultures with performances and food",
            "date": "2025-03-05",
            "location": "School Auditorium",
            "price": 400.00
        },
        {
            "name": "Science Fair",
            "description": "Student science projects and demonstrations",
            "date": "2025-03-20",
            "location": "Science Building",
            "price": 150.00
        },
        {
            "name": "Graduation Ball",
            "description": "Formal dance for graduating students",
            "date": "2025-04-15",
            "location": "Grand Ballroom",
            "price": 800.00
        },
        {
            "name": "Book Fair",
            "description": "Annual book fair with local and international publishers",
            "date": "2025-05-10",
            "location": "Library Hall",
            "price": 100.00
        },
        {
            "name": "Music Festival",
            "description": "Student music performances and concerts",
            "date": "2025-06-01",
            "location": "Outdoor Amphitheater",
            "price": 350.00
        }
    ]

    await events_collection.delete_many({})
    event_result = await events_collection.insert_many(sample_events)
    print(f"✅ Created events collection with {len(event_result.inserted_ids)} sample events")

    # 7. Receipts Collection - No sample receipts (will be populated by users)
    receipts_collection = db.receipts
    await receipts_collection.delete_many({})
    print("✅ Created empty receipts collection (no sample data)")

    # List all collections
    collections = await db.list_collection_names()
    print(f"\n📋 Database initialized with collections: {collections}")

    print("\n🎉 Database initialization complete!")
    print("💡 You can now view these collections in MongoDB Compass")
    print("🔐 Sample login credentials:")
    print("   Admin: username='admin', password='admin123'")
    print("   Teacher: username='teacher1', password='admin123'")

    # Close connection
    client.close()

if __name__ == "__main__":
    asyncio.run(init_database())
