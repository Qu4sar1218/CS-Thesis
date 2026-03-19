#!/usr/bin/env python3
"""
Migration script: attendance -> class_attendance + new hallway/events collections.
"""

import asyncio
import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DB_URI = "mongodb://localhost:27017"
DB_NAME = "InterACTS"

async def connect_to_mongodb():
    """
    Connect to MongoDB.
    """
    try:
        client = AsyncIOMotorClient(DB_URI)
        db = client[DB_NAME]
        # Test connection
        await db.command('ping')
        logger.info(f"✅ Connected to MongoDB: {DB_NAME}")
        return db
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        raise

async def migrate_attendance_collections():
    """
    Main migration function - idempotent.
    """
    db = await connect_to_mongodb()
    
    collections = await db.list_collection_names()
    logger.info(f"Current collections: {collections}")
    
    # Step 1: Rename attendance -> class_attendance if needed
    if 'attendance' in collections and 'class_attendance' not in collections:
        logger.info("🔄 Renaming 'attendance' -> 'class_attendance'...")
        result = await db.command('renameCollection', 'attendance', 'class_attendance', dropTarget=True)
        logger.info(f"Rename result: {result}")
        logger.info("✅ 'attendance' renamed to 'class_attendance'")
        collections = await db.list_collection_names()  # Refresh
    elif 'attendance' in collections:
        logger.warning("⚠️ 'attendance' still exists - manual cleanup needed")
    else:
        logger.info("ℹ️ No 'attendance' collection found")
    
    # Step 2: Create events_attendance if missing
    if 'events_attendance' not in collections:
        logger.info("📥 Creating 'events_attendance' collection...")
        await db.events_attendance.insert_one({'_migration': f'events_attendance created {datetime.now()}'})
        await db.events_attendance.delete_one({})  # Clean marker
        logger.info("✅ 'events_attendance' created")
    
    # Step 3: Create hallway_attendance if missing
    if 'hallway_attendance' not in collections:
        logger.info("🚶 Creating 'hallway_attendance' collection...")
        await db.hallway_attendance.insert_one({'_migration': f'hallway_attendance created {datetime.now()}'})
        await db.hallway_attendance.delete_one({})  # Clean marker
        logger.info("✅ 'hallway_attendance' created")
    
    # Step 4: Migrate event docs from class_attendance -> events_attendance
    if 'class_attendance' in collections:
        events_docs = await db.class_attendance.find({'mode': 'events'}).to_list(length=None)
        if events_docs:
            logger.info(f"📤 Moving {len(events_docs)} event docs to 'events_attendance'...")
            await db.events_attendance.insert_many(events_docs)
            await db.class_attendance.delete_many({'mode': 'events'})
            logger.info(f"✅ Moved {len(events_docs)} events to 'events_attendance'")
        else:
            logger.info("ℹ️ No event docs to migrate")
    
    final_collections = await db.list_collection_names()
    logger.info(f"Final collections: {final_collections}")
    logger.info("🎉 Migration complete! Ready for code updates.")

async def main():
    try:
        await migrate_attendance_collections()
    except KeyboardInterrupt:
        logger.info("🛑 Migration interrupted by user")
    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        raise
    logger.info("Next: python main.py to test server")

if __name__ == '__main__':
    asyncio.run(main())

