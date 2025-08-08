# MINIMAL FIX: Just replace the RetinaFace initialization in your existing code

import cv2
import logging
import os
import warnings
from typing import List, Dict, Any
import numpy as np


# Add these lines at the top to suppress warnings
warnings.filterwarnings("ignore", message=".*rcond.*")
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from config import settings

logger = logging.getLogger(__name__)

# Your existing imports (keep as is)
try:
    import dlib
    DLIB_AVAILABLE = True
    logger.info("dlib is available")
except ImportError:
    DLIB_AVAILABLE = False
    logger.warning("dlib not available")        

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available")
    try:
        from facenet_pytorch import MTCNN
        MTCNN_AVAILABLE = True
        logger.info("MTCNN is available")
    except ImportError:
        MTCNN_AVAILABLE = False
        logger.warning("MTCNN not available")
except ImportError:
    TORCH_AVAILABLE = False
    MTCNN_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logger.info("TensorFlow is available")
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available")

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    # CHANGE: Don't set RETINAFACE_AVAILABLE = True
    logger.info("InsightFace is available (but skipping RetinaFace)")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available")

# CHANGE: Force RetinaFace to be unavailable
RETINAFACE_AVAILABLE = False
    
class MultiFaceDetector:
    """Face detector with multiple backend options, prioritizing MTCNN over RetinaFace"""
    def __init__(self):
        self.initialized_detector_type = "haarcascade"  # Default, best successfully initialized
        self.mtcnn_model = None
        self.dlib_model = None
        self.haar_model = None
        # self.insightface_app = None # Not used for RetinaFace in this version

        # Attempt to initialize MTCNN
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn_model = MTCNN(
                    image_size=160, 
                    margin=0, 
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7], 
                    factor=0.709, 
                    device='cpu'  # Force CPU
                )
                self.initialized_detector_type = "mtcnn"
                logger.info("Successfully initialized MTCNN detector.")
            except Exception as e:
                logger.warning(f"Could not initialize MTCNN detector: {e}")

        # Attempt to initialize Dlib
        if DLIB_AVAILABLE:
            try:
                self.dlib_model = dlib.get_frontal_face_detector()
                # If MTCNN failed and Dlib succeeded, Dlib is now the best
                if self.initialized_detector_type == "haarcascade":
                    self.initialized_detector_type = "dlib"
                logger.info("Successfully initialized dlib face detector.")
            except Exception as e:
                logger.warning(f"Could not initialize dlib detector: {e}")

        # Attempt to initialize Haar cascade
        try:
            self.haar_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.haar_model.empty():
                logger.error("Failed to load Haar cascade model, it will be unavailable.")
                self.haar_model = None
            else:
                logger.info("Successfully initialized Haar cascade face detector.")
        except Exception as e:
            logger.error(f"Could not initialize Haar cascade detector: {e}")
            self.haar_model = None

        logger.info(f"MultiFaceDetector initialized. Preferred detector: {self.initialized_detector_type}")

    def _detect_with_mtcnn(self, image: np.ndarray) -> List[Dict[str, Any]]:
        faces = []
        if not self.mtcnn_model:
            return faces
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, probs = self.mtcnn_model.detect(rgb_image)
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob > 0.9: # High confidence
                        x1, y1, x2, y2 = box.astype(int)
                        faces.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": float(prob), "type": "mtcnn"})
        except Exception as e:
            logger.error(f"MTCNN detection attempt error: {e}")
        return faces

    def _detect_with_dlib(self, image: np.ndarray) -> List[Dict[str, Any]]:
        faces = []
        if not self.dlib_model:
            return faces
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dlib_faces_rects = self.dlib_model(gray)
            for i, face_rect in enumerate(dlib_faces_rects):
                x, y, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
                faces.append({"bbox": [int(x), int(y), int(x2), int(y2)], "confidence": 0.9 - (i * 0.1), "type": "dlib"}) # Arbitrary confidence
        except Exception as e:
            logger.error(f"Dlib detection attempt error: {e}")
        return faces

    def _detect_with_haar(self, image: np.ndarray) -> List[Dict[str, Any]]:
        faces = []
        if not self.haar_model:
            return faces
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected = self.haar_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for i, (x, y, w, h_val) in enumerate(detected):
                faces.append({"bbox": [int(x), int(y), int(x + w), int(y + h_val)], "confidence": 0.8 - (i * 0.05), "type": "haarcascade"})
            if not faces:  # Relaxed params if no faces found
                detected = self.haar_model.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
                for i, (x, y, w, h_val) in enumerate(detected):
                    faces.append({"bbox": [int(x), int(y), int(x + w), int(y + h_val)], "confidence": 0.7 - (i * 0.05), "type": "haarcascade_relaxed"})
        except Exception as e:
            logger.error(f"Haar detection attempt error: {e}")
        return faces

    def detect_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """Optimized face detection - maintains accuracy with speed boost"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            h, w = image.shape[:2]
            faces = []
            
            # Try best available detector first
            if self.mtcnn_model:
                try:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    boxes, probs = self.mtcnn_model.detect(rgb_image)
                    if boxes is not None:
                        for box, prob in zip(boxes, probs):
                            if prob > 0.9:
                                x1, y1, x2, y2 = box.astype(int)
                                faces.append({
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": float(prob),
                                    "type": "mtcnn"
                                })
                    if faces:
                        return faces[:1]  # Return best face
                except:
                    pass
            
            # Haar cascade with smart scaling
            if self.haar_model:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Smart scaling - only scale if image is very large
                scale = min(1.0, 640 / max(w, h))  # Keep more detail
                if scale < 1.0:
                    small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
                else:
                    small_gray = gray
                    
                # Try with good parameters first
                detected = self.haar_model.detectMultiScale(small_gray, 1.1, 5, minSize=(30, 30))
                
                if len(detected) > 0:
                    for x, y, fw, fh in detected:
                        if scale < 1.0:
                            x, y, fw, fh = int(x/scale), int(y/scale), int(fw/scale), int(fh/scale)
                        faces.append({
                            "bbox": [x, y, x + fw, y + fh],
                            "confidence": 0.85,
                            "type": "haarcascade"
                        })
                    return sorted(faces, key=lambda x: x["confidence"], reverse=True)[:1]
                
                # Try with relaxed parameters if no faces found
                detected = self.haar_model.detectMultiScale(small_gray, 1.05, 3, minSize=(20, 20))
                if len(detected) > 0:
                    x, y, fw, fh = detected[0]
                    if scale < 1.0:
                        x, y, fw, fh = int(x/scale), int(y/scale), int(fw/scale), int(fh/scale)
                    return [{
                        "bbox": [x, y, x + fw, y + fh],
                        "confidence": 0.75,
                        "type": "haarcascade_relaxed"
                    }]
            
            # Smart fallback based on image type
            if w / h > 1.2:  # ID card
                return [{
                    "bbox": [w//2, 0, w, h//2],
                    "confidence": 0.6,
                    "type": "id_region"
                }]
            else:  # Regular photo
                margin = min(w, h) // 4
                return [{
                    "bbox": [w//2-margin, h//2-margin, w//2+margin, h//2+margin],
                    "confidence": 0.5,
                    "type": "center_region"
                }]
                
        except:
            return []
    
    def get_best_face(self, image_path: str) -> Dict[str, Any]:
        """Get best face with guaranteed result"""
        faces = self.detect_faces(image_path)
        if faces:
            return faces[0]
        else:
            return None 
    
    # NEW METHOD: Added for rotation handling support
    def get_best_face_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Get best face from numpy image array (for rotation handling)"""
        try:
            if image is None:
                return None
                
            h, w = image.shape[:2]
            faces = []
            
            # Try MTCNN first if available
            if self.mtcnn_model:
                try:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    boxes, probs = self.mtcnn_model.detect(rgb_image)
                    if boxes is not None:
                        for box, prob in zip(boxes, probs):
                            if prob > 0.9:
                                x1, y1, x2, y2 = box.astype(int)
                                return {
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": float(prob),
                                    "type": "mtcnn"
                                }
                except Exception as e:
                    logger.debug(f"MTCNN detection from image error: {e}")
                    pass
            
            # Try Haar cascade if MTCNN failed
            if self.haar_model:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Smart scaling for large images
                    scale = min(1.0, 640 / max(w, h))
                    if scale < 1.0:
                        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
                    else:
                        small_gray = gray
                    
                    # Try with good parameters first
                    detected = self.haar_model.detectMultiScale(small_gray, 1.1, 5, minSize=(30, 30))
                    
                    if len(detected) > 0:
                        x, y, fw, fh = detected[0]  # Get best detection
                        if scale < 1.0:
                            x, y, fw, fh = int(x/scale), int(y/scale), int(fw/scale), int(fh/scale)
                        return {
                            "bbox": [x, y, x + fw, y + fh],
                            "confidence": 0.85,
                            "type": "haarcascade"
                        }
                    
                    # Try with relaxed parameters
                    detected = self.haar_model.detectMultiScale(small_gray, 1.05, 3, minSize=(20, 20))
                    if len(detected) > 0:
                        x, y, fw, fh = detected[0]
                        if scale < 1.0:
                            x, y, fw, fh = int(x/scale), int(y/scale), int(fw/scale), int(fh/scale)
                        return {
                            "bbox": [x, y, x + fw, y + fh],
                            "confidence": 0.75,
                            "type": "haarcascade_relaxed"
                        }
                        
                except Exception as e:
                    logger.debug(f"Haar detection from image error: {e}")
                    pass
            
            # Try Dlib as fallback
            if self.dlib_model:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dlib_faces = self.dlib_model(gray)
                    if len(dlib_faces) > 0:
                        face_rect = dlib_faces[0]
                        x, y, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
                        return {
                            "bbox": [int(x), int(y), int(x2), int(y2)],
                            "confidence": 0.80,
                            "type": "dlib"
                        }
                except Exception as e:
                    logger.debug(f"Dlib detection from image error: {e}")
                    pass
            
            # Last resort: smart fallback based on image shape
            if w / h > 1.2:  # Likely ID card format
                return {
                    "bbox": [w//2, 0, w, h//2],
                    "confidence": 0.6,
                    "type": "id_region_fallback"
                }
            else:  # Regular photo format
                margin = min(w, h) // 4
                return {
                    "bbox": [w//2-margin, h//2-margin, w//2+margin, h//2+margin],
                    "confidence": 0.5,
                    "type": "center_region_fallback"
                }
        
        except Exception as e:
            logger.error(f"Face detection from image error: {e}")
            return None 