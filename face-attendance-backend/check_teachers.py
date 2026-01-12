import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check_teachers():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['InterACTS']

    print('Teachers in database:')
    async for teacher in db.teachers.find():
        print(f'Teacher ID: {teacher.get("teacher_id")}')
        print(f'Email: {teacher.get("email")}')
        print(f'First Name: {teacher.get("first_name")}')
        print(f'Last Name: {teacher.get("last_name")}')
        print(f'Full Name would be: {teacher.get("first_name", "")} {teacher.get("last_name", "")}'.strip())
        print('---')

    client.close()

if __name__ == "__main__":
    asyncio.run(check_teachers())
