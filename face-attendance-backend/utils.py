import os
import sys

import face_recognition
try:
    from pkg_resources import resource_filename
except ImportError:
    # Fallback for systems where pkg_resources is not available
    import importlib.resources
    def resource_filename(package, resource_name):
        return importlib.resources.files(package).joinpath(resource_name).read_text(encoding='latin1')
import logging

logger = logging.getLogger(__name__)

# Global flag to prevent multiple setups
_face_model_setup_done = False

def setup_face_recognition_model():
    """
    Explicitly locate and set the face recognition model path.
    Call this BEFORE importing face_recognition.
    
    This fixes path resolution issues on Windows/other systems where the model
    can't be found automatically.
    """
    global _face_model_setup_done
    
    if _face_model_setup_done:
        logger.debug("Face recognition model already setup - skipping")
        return True
    
    try:
        # Import required packages for model location
        import face_recognition_models
        
        # Get the exact path to the model file  
        model_path = resource_filename('face_recognition_models', 
                                     'dlib_face_recognition_resnet_model_v1.dat')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at: {model_path}")
            logger.info(f"Available files in package: {os.listdir(os.path.dirname(model_path))}")
            return False
        
        logger.info(f"✅ Face recognition model found: {model_path}")
        
        # Set the model location BEFORE importing face_recognition
        # This makes it available globally for all face_recognition functions
        face_recognition.face_recognition_model_location = model_path
        
        _face_model_setup_done = True
        logger.info("✅ Face recognition model setup completed successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to setup face recognition model: {e}")
        return False

def is_face_model_ready():
    """Check if model setup was successful."""
    global _face_model_setup_done
    return _face_model_setup_done

# DEFER auto-setup to avoid import-time face_recognition dependency
# Call setup_face_recognition_model() explicitly before using face_recognition
pass

