import cv2
import numpy as np
import logging
#from config import settings

logger = logging.getLogger(__name__)
try:
    import torch
    TORCH_AVAILABLE = True
    try:
        from facenet_pytorch import InceptionResnetV1
        FACENET_AVAILABLE = True
        logger.info("FaceNet is available")
    except ImportError:
        FACENET_AVAILABLE = False
        logger.warning("FaceNet not available")
except ImportError:
    TORCH_AVAILABLE = False
    FACENET_AVAILABLE = False
    logger.warning("PyTorch not available")
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
    ARCFACE_AVAILABLE = True
    logger.info("InsightFace is available (includes (ArcFace)")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    ARCFACE_AVAILABLE = False
    logger.warning("InsightFace not available")
logger = logging.getLogger(__name__)

class FaceRecognizer:
    """Face recognition with multiple backend options, prioritizing ArcFace"""
    
    def __init__(self):
        self.recognizer_type = "none"
        self.model = None
        self.device = "cpu"
        
        # Try to use GPU if available with PyTorch
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        # Try to initialize ArcFace from InsightFace
        if INSIGHTFACE_AVAILABLE and ARCFACE_AVAILABLE:
            try:
                # InsightFace's FaceAnalysis includes both RetinaFace and ArcFace
                # We already initialized it in MultiFaceDetector, so just set the type
                self.recognizer_type = "arcface"
                logger.info("Using ArcFace recognition model from InsightFace")
                return
            except Exception as e:
                logger.warning(f"Could not initialize ArcFace recognizer: {e}")
        
        # Try to initialize FaceNet if InsightFace not available
        if FACENET_AVAILABLE:
            try:
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.recognizer_type = "facenet"
                logger.info("Using FaceNet recognition model")
                return
            except Exception as e:
                logger.warning(f"Could not initialize FaceNet recognizer: {e}")
        
        # Fallback to HOG + SVM approach or other OpenCV methods
        logger.info("Using HOG features for face recognition")
        self.recognizer_type = "hog"
    
    def extract_features(self, face_img: np.ndarray) -> np.ndarray:
        """Extract facial features using the best available model"""
        try:
            if face_img is None or face_img.size == 0:
                return None
                
            # Resize to expected input size
            if self.recognizer_type in ["arcface", "facenet"]:
                face_img = cv2.resize(face_img, (160, 160))
            else:
                face_img = cv2.resize(face_img, (128, 128))
            
            if self.recognizer_type == "arcface":
                # We'll handle ArcFace feature extraction at the verification level
                # since InsightFace's FaceAnalysis combines detection and recognition
                return face_img
                
            elif self.recognizer_type == "facenet":
                # Convert BGR to RGB for PyTorch models
                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Convert to PyTorch tensor
                img_tensor = torch.from_numpy(rgb_img.transpose((2, 0, 1))).float()
                img_tensor = img_tensor.unsqueeze(0).to(self.device) / 255.0  # Add batch dimension
                
                # Extract features
                with torch.no_grad():
                    embeddings = self.model(img_tensor)
                
                return embeddings.cpu().numpy()[0]
                
            else:
                # Fallback to HOG features
                if len(face_img.shape) > 2:
                    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = face_img
                
                # Apply histogram equalization for better feature extraction
                gray = cv2.equalizeHist(gray)
                
                # Extract HOG features
                hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
                features = hog.compute(gray)
                
                # Normalize features
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
                
                return features
                
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return None
