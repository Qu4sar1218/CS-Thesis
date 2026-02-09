"""
Configuration management using environment variables and Pydantic settings.
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server Configuration
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=True, env="DEBUG")

    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY", description="JWT secret key")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Database
    mongodb_url: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    database_name: str = Field(default="InterACTS", env="DATABASE_NAME")

    # Face Recognition
    face_match_threshold: float = Field(default=0.5, env="FACE_MATCH_THRESHOLD")
    process_every_n_frames: int = Field(default=5, env="PROCESS_EVERY_N_FRAMES")
    jpeg_quality: int = Field(default=70, env="JPEG_QUALITY")
    frame_scale: float = Field(default=0.5, env="FRAME_SCALE")

    # Paths
    project_root: str = Field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    backend_root: str = Field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    face_data_dir: str = Field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "StudentFaceData"))
    encodings_dir: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "encodings"))
    logs_dir: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.encodings_dir, exist_ok=True)
os.makedirs(settings.logs_dir, exist_ok=True)
os.makedirs(settings.face_data_dir, exist_ok=True)
