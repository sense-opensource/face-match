import logging

logger = logging.getLogger(__name__)

# Initialize availability flags
DLIB_AVAILABLE = False
RETINAFACE_AVAILABLE = False
MTCNN_AVAILABLE = False
FACENET_AVAILABLE = False
ARCFACE_AVAILABLE = False
INSIGHTFACE_AVAILABLE = False
TF_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
    logger.info("dlib is available")
except ImportError:
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
        logger.warning("MTCNN not available")
        
    try:
        from facenet_pytorch import InceptionResnetV1
        FACENET_AVAILABLE = True
        logger.info("FaceNet is available")
    except ImportError:
        logger.warning("FaceNet not available")
        
except ImportError:
    logger.warning("PyTorch not available")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logger.info("TensorFlow is available")
except ImportError:
    logger.warning("TensorFlow not available")

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    RETINAFACE_AVAILABLE = True
    ARCFACE_AVAILABLE = True
    logger.info("InsightFace is available (includes RetinaFace and ArcFace)")
except ImportError:
    logger.warning("InsightFace not available")