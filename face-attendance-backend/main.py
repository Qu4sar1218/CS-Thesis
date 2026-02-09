"""
Face Attendance Backend - Main Application
Refactored modular version
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database.connection import db_manager
from face_recognition import load_faces_from_db

# Import route modules
from routes import auth_router, recognition_router, student_router, teacher_router, class_router, events_router, database_views_router, attendance_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(settings.logs_dir, "server.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("🚀 Starting up the application...")
    await db_manager.connect()
    await load_faces_from_db()
    logger.info("✅ Application startup complete")

    yield

    # Shutdown
    logger.info("🛑 Shutting down the application...")
    await db_manager.close()
    logger.info("✅ Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Face Attendance Backend",
    description="AI-powered face recognition attendance system",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3003"],  # Allow frontend origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(recognition_router, tags=["Face Recognition"])
app.include_router(student_router, prefix="/students", tags=["Student Management"])
app.include_router(teacher_router, prefix="/teachers", tags=["Teacher Management"])
app.include_router(class_router, prefix="/classes", tags=["Class Management"])
app.include_router(events_router, prefix="/events", tags=["Event Management"])
app.include_router(attendance_router, prefix="/attendance", tags=["Attendance Management"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "database_connected": db_manager.database is not None,
        "face_recognition_loaded": True  # TODO: Add actual face recognition status check
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"✅ Server listening at http://{settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="warning"
    )
