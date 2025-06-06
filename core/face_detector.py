# MINIMAL FIX: Just replace the RetinaFace initialization in your existing code

import cv2
import logging
import os
import warnings
from typing import List, Dict, Any

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
        self.detector_type = "haarcascade"  # Default
        self.detector = None
        self.mtcnn = None
        self.insightface_app = None
        
        # CHANGE: Skip RetinaFace completely, start with MTCNN
        logger.info("Skipping RetinaFace to avoid downloads")
        
        # Try to initialize MTCNN detector FIRST
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN(
                    image_size=160, 
                    margin=0, 
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7], 
                    factor=0.709, 
                    device='cpu'  # Force CPU
                )
                self.detector_type = "mtcnn"
                logger.info("Using MTCNN detector")
                return
            except Exception as e:
                logger.warning(f"Could not initialize MTCNN detector: {e}")
        
        # CHANGE: Skip InsightFace RetinaFace initialization
        # (Remove the old RetinaFace initialization code)
        
        # Try to initialize dlib detector if MTCNN not available
        if DLIB_AVAILABLE:
            try:
                import dlib
                self.detector = dlib.get_frontal_face_detector()
                self.detector_type = "dlib"
                logger.info("Using dlib face detector")
                return
            except Exception as e:
                logger.warning(f"Could not initialize dlib detector: {e}")
        
        # Fallback to Haar cascade
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.detector_type = "haarcascade"
        logger.info("Using Haar cascade face detector")
    
    # Keep all your existing detect_faces and get_best_face methods exactly as they are
    # Just change the detector priority in __init__
    
    def detect_faces(self, image_path: str) -> List[Dict[str, Any]]:
        print(self.detector_type)
        """Detect faces with the best available method"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []
            
            faces = []
            
            # CHANGE: Start with MTCNN instead of RetinaFace
            if self.detector_type == "mtcnn":
                # Use MTCNN detector
                try:
                    # Convert BGR to RGB for MTCNN
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Get detections
                    boxes, probs = self.mtcnn.detect(rgb_image)
                    
                    if boxes is not None:
                        for i, (box, prob) in enumerate(zip(boxes, probs)):
                            if prob > 0.9:  # Only use high confidence detections
                                x1, y1, x2, y2 = box.astype(int)
                                
                                faces.append({
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": float(prob),
                                    "type": "mtcnn"
                                })
                except Exception as e:
                    logger.error(f"MTCNN detection error: {e}")
                    # Fall back to another method
                    self.detector_type = "dlib" if DLIB_AVAILABLE else "haarcascade"
                    return self.detect_faces(image_path)
            
            # REMOVE: The entire retinaface detection block
            # elif self.detector_type == "retinaface":
            #     # (Remove this entire section)
            
            elif self.detector_type == "dlib":
                # Keep your existing dlib code
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dlib_faces = self.detector(gray)
                    
                    for i, face in enumerate(dlib_faces):
                        x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                        faces.append({
                            "bbox": [int(x), int(y), int(x2), int(y2)],
                            "confidence": 0.9 - (i * 0.1),  # Arbitrary confidence score
                            "type": "dlib"
                        })
                except Exception as e:
                    logger.error(f"Dlib detection error: {e}")
                    # Fall back to another method
                    self.detector_type = "haarcascade"
                    return self.detect_faces(image_path)
            
            else:
                # Keep your existing Haar cascade code exactly as is
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # First try with standard parameters
                detected = self.detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                for i, (x, y, w, h) in enumerate(detected):
                    faces.append({
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                        "confidence": 0.8 - (i * 0.05),  # Arbitrary confidence score
                        "type": "haarcascade"
                    })
                
                # If no faces found, try with relaxed parameters
                if not faces:
                    detected = self.detector.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
                    )
                    
                    for i, (x, y, w, h) in enumerate(detected):
                        faces.append({
                            "bbox": [int(x), int(y), int(x + w), int(y + h)],
                            "confidence": 0.7 - (i * 0.05),  # Lower confidence for relaxed params
                            "type": "haarcascade_relaxed"
                        })
            
            # Keep all your existing fallback logic exactly as is
            if not faces:
                h, w = image.shape[:2]
                is_id_card = w / h > 1.4 or w / h < 0.7
                
                if is_id_card:
                    try:
                        from core.image_enhancer import AdvancedImageEnhancer
                        enhancer = AdvancedImageEnhancer()
                        face_region = enhancer._locate_face_in_id(image)
                        
                        if face_region:
                            x, y, rw, rh = face_region
                            faces.append({
                                "bbox": [x, y, x + rw, y + rh],
                                "confidence": 0.6,
                                "type": "id_document_face"
                            })
                        else:
                            # Common regions for ID cards if face detection failed
                            regions = [
                                {"x": w//2, "y": 0, "w": w//2, "h": h//2, "confidence": 0.5, "type": "id_top_right"},
                                {"x": 0, "y": 0, "w": w//2, "h": h//2, "confidence": 0.45, "type": "id_top_left"},
                            ]
                            
                            for region in regions:
                                x, y, r_w, r_h = region["x"], region["y"], region["w"], region["h"]
                                x2 = min(w, x + r_w)
                                y2 = min(h, y + r_h)
                                
                                faces.append({
                                    "bbox": [x, y, x2, y2],
                                    "confidence": region["confidence"],
                                    "type": region["type"]
                                })
                    except:
                        # Fallback regions if enhancer import fails
                        regions = [
                            {"x": w//2, "y": 0, "w": w//2, "h": h//2, "confidence": 0.5, "type": "id_top_right"},
                            {"x": 0, "y": 0, "w": w//2, "h": h//2, "confidence": 0.45, "type": "id_top_left"},
                        ]
                        
                        for region in regions:
                            x, y, r_w, r_h = region["x"], region["y"], region["w"], region["h"]
                            x2 = min(w, x + r_w)
                            y2 = min(h, y + r_h)
                            
                            faces.append({
                                "bbox": [x, y, x2, y2],
                                "confidence": region["confidence"],
                                "type": region["type"]
                            })
                else:
                    # For regular photos that aren't IDs, assume face is in center
                    center_margin = min(w, h) // 4
                    x1 = max(0, w//2 - center_margin)
                    y1 = max(0, h//2 - center_margin)
                    x2 = min(w, w//2 + center_margin)
                    y2 = min(h, h//2 + center_margin)
                    
                    faces.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": 0.4,
                        "type": "center_region"
                    })
            
            # Sort by confidence
            faces.sort(key=lambda x: x["confidence"], reverse=True)
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return []
    
    def get_best_face(self, image_path: str) -> Dict[str, Any]:
        """Get best face with guaranteed result"""
        faces = self.detect_faces(image_path)
        if faces:
            return faces[0]
        else:
            return None 