"""
Database connection management.
"""
import logging
from motor.motor_asyncio import AsyncIOMotorClient

from config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager class for handling MongoDB connections."""

    def __init__(self):
        self.database = None
        self.client = None

    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_url)
            self.database = self.client[settings.database_name]
            logger.info("✅ Connected to MongoDB")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise

    async def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_database():
    """Get the database instance from the manager."""
    return db_manager.database
