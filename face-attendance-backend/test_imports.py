#!/usr/bin/env python3
"""
Test script to verify all required imports work correctly.
"""

def test_imports():
    try:
        # Flask imports
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        print("✅ Flask imports OK")

        # Standard libraries
        import os
        import cv2
        import numpy as np
        import pickle
        from datetime import datetime
        from werkzeug.utils import secure_filename
        print("✅ Standard libraries OK")

        # MongoDB imports
        from pymongo import MongoClient
        from motor.motor_asyncio import AsyncIOMotorClient
        from bson import ObjectId
        print("✅ MongoDB imports OK")

        # Authentication imports
        from passlib.context import CryptContext
        import jwt
        print("✅ Authentication imports OK")

        # Pydantic imports
        from pydantic import BaseModel, Field
        print("✅ Pydantic imports OK")

        # FastAPI imports
        from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
        from fastapi.responses import JSONResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
        print("✅ FastAPI imports OK")

        # Face recognition import
        try:
            import face_recognition
            HAVE_FACE_RECOG = True
            print("✅ Face recognition import OK")
        except Exception as e:
            print(f"⚠️  Face recognition import failed: {e}")
            HAVE_FACE_RECOG = False

        print("\n🎉 All imports successful!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True

if __name__ == "__main__":
    test_imports()
