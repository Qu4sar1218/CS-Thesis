#!/usr/bin/env python3
"""
Face Attendance System - Random Student Generator
Connects directly to MongoDB 'InterACTS.students' collection.
Uses REAL courses from your 'courses' collection + smart unique ID generation.
Compatible with your exact database schema.

Usage: python generate_random_students.py
"""

import pymongo
import random
import string
from datetime import datetime

# MongoDB connection (matches your backend)
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["InterACTS"]

def get_courses():
    """Get REAL courses from your 'courses' collection"""
    courses = list(db.courses.find({}, {"code":1, "name":1}).limit(50))
    course_codes = [c["code"] for c in courses if c.get("code")]
    if course_codes:
        print(f"✅ Found {len(course_codes)} real courses from DB")
        return course_codes
    print("⚠️ No courses found, using defaults")
    return ["BS Computer Science", "BS Information Technology", "BS Nursing", "BS Accountancy",
            "BS Education", "BS Engineering", "BS Business Administration"]

def get_years():
    """Grade 11-12 + College 1st-4th year"""
    return ["Grade 11", "Grade 12", "1st Year", "2nd Year", "3rd Year", "4th Year"]

# Load real data
COURSES = get_courses()
YEARS = get_years()

# Expanded name pools (more Filipino names to match your system)
FIRST_NAMES = ["John", "Jane", "Michael", "Emily", "David", "Sarah", "Chris", "Lisa", "Mark", "Anna",
               "James", "Mary", "Robert", "Patricia", "Joseph", "Jennifer", "Daniel", "Maria",
               "Juan", "Maria", "Jose", "Ana", "Carlos", "Sofia", "Miguel", "Isabella"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Santos", "Cruz", "Reyes", "Dela Cruz", "Torres", "Ramos", "Bautista"]
MIDDLE_NAMES = ["R", "L", "M", "D", "A", "C", "E", "B", "S"]

def generate_unique_student_id():
    """Generate 6-digit numeric ID (100000-999999), check DB for uniqueness"""
    attempts = 0
    while attempts < 200:
        sid = f"{random.randint(100000, 999999)}"
        if not db.students.find_one({"student_id": sid}):
            return sid
        attempts += 1
    raise Exception("Could not generate unique ID after 200 attempts")

def create_student():
    """Create student matching your EXACT DB schema"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    middle = random.choice(MIDDLE_NAMES) if random.random() > 0.3 else ""
    
    student_id = generate_unique_student_id()
    
    return {
        "student_id": student_id,
        "first_name": first,
        "middle_name": middle,
        "last_name": last,
        "course": random.choice(COURSES),
        "year": random.choice(YEARS),
        "email": f"{first.lower()}.{last.lower()}.{student_id.lower()}@school.edu",
        "guardian_contact": f"+63 9{random.randint(17,99)} {random.randint(100,999)} {random.randint(1000,9999)}"
    }

def main(num_students=50):
    print(f"🚀 Generating {num_students} students with YOUR real data...")
    print(f"📚 Courses: {COURSES[:6]}{'...' if len(COURSES)>6 else ''}")
    print(f"📅 Years: {YEARS}")
    print("-" * 60)
    
    created = 0
    skipped_dupes = 0
    
    for i in range(num_students):
        student = create_student()
        try:
            result = db.students.insert_one(student)
            if result.inserted_id:
                created += 1
                print(f"✅ #{created:2d} {student['first_name']:8s} {student['middle_name'] or '':1s}. {student['last_name']:10s} | {student['student_id']:6s} | {student['course']:25s} {student['year']}")
            else:
                skipped_dupes += 1
                print(f"⚠️  Skip #{i+1} (insert failed): {student['student_id']}")
        except pymongo.errors.DuplicateKeyError:
            skipped_dupes += 1
            print(f"⚠️  Skip #{i+1} (ID exists): {student['student_id']}")
        except Exception as e:
            print(f"❌ #{i+1} Failed {student['student_id']}: {str(e)[:60]}")
    
    total_students = db.students.count_documents({})
    print("\n" + "="*60)
    print(f"✅ SUCCESS! Created: {created} new students")
    print(f"⚠️  Skipped duplicates: {skipped_dupes}")
    print(f"📊 Total students now: {total_students:,}")
    print(f"🎉 Reload StudentList page - pagination ready to test!")
    print("💡 Backend restart optional - DB changes are live.")

if __name__ == "__main__":
    print("=== FACE ATTENDANCE RANDOM STUDENT GENERATOR ===")
    print("🎯 Uses YOUR exact database schema & courses collection")
    print("🔗 Direct MongoDB insert (no API needed)")
    print("🆔 Generates unique 6-char student IDs")
    print()
    
    try:
        num = input("Number of students? [50]: ").strip()
        num_students = int(num) if num else 50
        
        main(num_students)
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
    except Exception as e:
        print(f"💥 Error: {e}")
    finally:
        client.close()
