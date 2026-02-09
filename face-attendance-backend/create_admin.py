import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

async def create_admin():
    """Create admin user if not exists."""
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]
    users_collection = db.users

    admin = await users_collection.find_one({"email": "admin@school.edu"})
    if admin:
        print("Admin user already exists")
    else:
        hashed_password = pwd_context.hash("admin123")
        admin_data = {
            "username": "admin",
            "email": "admin@school.edu",
            "role": "admin",
            "hashed_password": hashed_password
        }
        result = await users_collection.insert_one(admin_data)
        print(f"Admin user created with ID: {result.inserted_id}")

    client.close()

if __name__ == "__main__":
    asyncio.run(create_admin())
