import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check_admin():
    """Check if admin user exists."""
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]
    users_collection = db.users

    admin = await users_collection.find_one({"email": "admin@school.edu"})
    if admin:
        print(f"Admin user found: {admin}")
    else:
        print("Admin user not found")

    client.close()

if __name__ == "__main__":
    asyncio.run(check_admin())
