#!/usr/bin/env python3
"""
Script to populate the students collection in MongoDB with 50 unique fake students.
"""

import asyncio
import random
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# Predefined data for generating students
FIRST_NAMES = [
    "Juan", "Maria", "Jose", "Ana", "Pedro", "Rosa", "Miguel", "Isabel", "Antonio", "Carmen",
    "Francisco", "Dolores", "Luis", "Pilar", "Carlos", "Concepcion", "Manuel", "Victoria", "Jesus", "Mercedes",
    "Angel", "Teresa", "Fernando", "Encarnacion", "Rafael", "Montserrat", "Diego", "Cristina", "Javier", "Antonia",
    "David", "Monica", "Alejandro", "Margarita", "Daniel", "Silvia", "Sergio", "Paz", "Alberto", "Consuelo",
    "Adrian", "Beatriz", "Pablo", "Inmaculada", "Ruben", "Milagros", "Ivan", "Lourdes", "Raul", "Esperanza",
    "Oscar", "Gloria", "Roberto", "Trinidad", "Alvaro", "Purificacion", "Mario", "Asuncion", "Ramon", "Candelaria",
    "Enrique", "Remedios", "Salvador", "Nieves", "Julian", "Luz", "Hugo", "Gracia", "Felix", "Amparo",
    "Eduardo", "Caridad", "Samuel", "Fidelidad", "Nicolas", "Misericordia", "Victor", "Fe", "Santiago", "Esperanza",
    "Mateo", "Carolina", "Lucas", "Gabriela", "Leo", "Valentina", "Sebastian", "Camila", "Mateo", "Sofia"
]

LAST_NAMES = [
    "Garcia", "Rodriguez", "Gonzalez", "Fernandez", "Lopez", "Martinez", "Sanchez", "Perez", "Martin", "Ruiz",
    "Hernandez", "Jimenez", "Diaz", "Moreno", "Munoz", "Alvarez", "Romero", "Navarro", "Torres", "Ramos",
    "Gil", "Ramirez", "Serrano", "Blanco", "Suarez", "Molina", "Morales", "Ortega", "Delgado", "Castro",
    "Ortiz", "Rubio", "Marin", "Sanz", "Iglesias", "Nuñez", "Medina", "Garrido", "Cortez", "Castillo",
    "Santos", "Arias", "Vega", "Flores", "Cabrera", "Campos", "Vargas", "Gomez", "Herrera", "Nieto",
    "Cortes", "Leon", "Guerrero", "Pena", "Prieto", "Vazquez", "Mendez", "Santiago", "Dominguez", "Mora"
]

COURSES = [
    "BSIT", "BSENTREP", "BSOA", "BSBA", "Btvted", "BSCS",
    "GAS", "HUMSS", "STEM", "ICT"
]

YEARS = ["1st", "2nd", "3rd", "4th"]

def generate_unique_student_id(existing_ids):
    """Generate a unique 6-digit student ID."""
    while True:
        student_id = f"{random.randint(100000, 999999)}"
        if student_id not in existing_ids:
            return student_id

def generate_email(first_name, last_name, student_id):
    """Generate a unique email address."""
    # Use first letter of first name + last name + student_id to ensure uniqueness
    base_email = f"{first_name[0].lower()}{last_name.lower()}{student_id}@student.edu"
    return base_email

async def populate_students():
    """Populate the students collection with 50 unique fake students."""
    # Connect to MongoDB
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]
    students_collection = db.students

    # Clear existing students (optional - comment out if you want to keep existing data)
    # await students_collection.delete_many({})

    # Get existing student IDs to avoid duplicates
    existing_students = await students_collection.find({}, {"student_id": 1}).to_list(None)
    existing_ids = {student["student_id"] for student in existing_students}

    students_data = []
    generated_ids = set()

    print("🔄 Generating 50 unique students...")

    while len(students_data) < 10:
        # Generate random data
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        full_name = f"{first_name} {last_name}"

        # Ensure unique student ID
        student_id = generate_unique_student_id(existing_ids | generated_ids)
        generated_ids.add(student_id)

        # Generate email
        email = generate_email(first_name, last_name, student_id)

        # Random course and year
        course = random.choice(COURSES)
        year = random.choice(YEARS)

        # Hash the password (using student_id as password)
        hashed_password = pwd_context.hash(student_id)

        student_data = {
            "student_id": student_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "course": course,
            "year": year,
            "hashed_password": hashed_password,
            "face_encodings": []  # Empty list - will be populated when face images are uploaded
        }

        students_data.append(student_data)

        if len(students_data) % 10 == 0:
            print(f"📝 Generated {len(students_data)} students...")

    # Insert students in batches
    print("💾 Inserting students into database...")
    result = await students_collection.insert_many(students_data)

    print(f"✅ Successfully inserted {len(result.inserted_ids)} students into the database")

    # Verify insertion
    total_count = await students_collection.count_documents({})
    print(f"📊 Total students in database: {total_count}")

    # Show sample students
    print("\n📋 Sample students:")
    sample_students = await students_collection.find().limit(5).to_list(None)
    for student in sample_students:
        print(f"  {student['student_id']} - {student['first_name']} {student['last_name']} ({student['course']} - {student['year']})")

    # Show course distribution
    print("\n📊 Course distribution:")
    pipeline = [
        {"$group": {"_id": "$course", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    course_stats = await students_collection.aggregate(pipeline).to_list(None)
    for stat in course_stats:
        print(f"  {stat['_id']}: {stat['count']} students")

    # Close connection
    client.close()

if __name__ == "__main__":
    asyncio.run(populate_students())
