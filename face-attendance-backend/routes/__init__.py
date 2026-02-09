"""
Routes package initialization.
"""
from .auth import router as auth_router
from .recognition import router as recognition_router
from .students import router as student_router
from .teachers import router as teacher_router
from .classes import router as class_router
from .events import router as events_router
from .database_views import router as database_views_router
from attendance import router as attendance_router
