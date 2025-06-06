from fastapi import UploadFile, HTTPException
import cv2
import numpy as np
from PIL import Image
import io
from typing import Any
import logging
from core.temp_file_manager import TempFileManager
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
logger = logging.getLogger(__name__)

def save_uploaded_file(file: UploadFile, temp_manager: TempFileManager) -> str:
    """Save uploaded file to temporary location with basic validation"""
    content = file.file.read()
    
    # Basic validation - verify it's an image
    try:
        Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    _, ext = os.path.splitext(file.filename or '.jpg')
    if not ext:
        ext = '.jpg'
    temp_path = temp_manager.create_temp_file(suffix=ext)
    
    with open(temp_path, "wb") as temp_file:
        temp_file.write(content)
    
    return temp_path

def compute_histogram_similarity(img1_path: str, img2_path: str) -> float:
    """Calculate histogram similarity between two face images"""
    try:
        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return 1.0  # Maximum distance
        
        # Convert to same size
        h, w = 128, 128
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))
        
        # Convert to HSV color space for better color comparison
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms with more bins for better precision
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [36, 36], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [36, 36], [0, 180, 0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Calculate similarity (correlation)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Convert to distance
        distance = 1.0 - similarity
        
        return max(0.0, min(1.0, distance))
    except Exception as e:
        logger.error(f"Error in histogram similarity calculation: {str(e)}")
        return 1.0

def compute_structural_similarity(img1_path: str, img2_path: str) -> float:
    """Calculate structural similarity between two face images"""
    try:
        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return 1.0  # Maximum distance
        
        # Convert to same size
        h, w = 128, 128
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Try to use scikit-image's SSIM implementation if available
        try:
            from skimage.metrics import structural_similarity as ssim
            similarity = ssim(gray1, gray2, data_range=255)
            distance = 1.0 - similarity
            return max(0.0, min(1.0, distance))
        except ImportError:
            # Fallback to simulated SSIM using OpenCV
            # Apply Gaussian blur to simulate local statistics
            blur1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
            blur2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
            
            # Compute MSE
            diff = cv2.absdiff(blur1, blur2)
            diff_sq = cv2.multiply(diff, diff)
            s = cv2.mean(diff_sq)[0]
            
            # Convert to similarity
            max_val = 255.0 * 255.0
            similarity = 1.0 - (s / max_val)
            
            # Convert to distance
            distance = 1.0 - similarity
            
            return max(0.0, min(1.0, distance))
    except Exception as e:
        logger.error(f"Error in structural similarity calculation: {str(e)}")
        return 1.0

def cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate cosine distance between two vectors with validation"""
    try:
        # Validate inputs
        if vector1 is None or vector2 is None:
            return 1.0
            
        # Ensure vectors are 1D
        vector1 = vector1.flatten()
        vector2 = vector2.flatten()
        
        # Check if sizes match
        min_size = min(vector1.size, vector2.size)
        if min_size == 0:
            return 1.0
            
        # Truncate to same size if necessary
        vector1 = vector1[:min_size]
        vector2 = vector2[:min_size]
        
        # Calculate dot product and norms
        dot = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        # Check for zero norms
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 1.0
            
        # Calculate similarity and convert to distance
        similarity = dot / (norm1 * norm2)
        
        # Constrain similarity to -1 to 1 range
        similarity = max(-1.0, min(1.0, similarity))
        
        # Convert to distance
        distance = 1.0 - similarity
        
        # Ensure distance is in valid range [0,1]
        distance = max(0.0, min(1.0, distance))
        
        return float(distance)
    except Exception as e:
        logger.error(f"Error in cosine distance calculation: {str(e)}")
        return 1.0
    
def convert_to_python_types(obj: Any) -> Any:
    """
    Convert numpy/torch types to Python native types for JSON serialization
    with special handling for InsightFace results
    """
    # Direct return for None
    if obj is None:
        return None
    
    # Handle numpy scalar types explicitly
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    
    # Handle numpy arrays and torch tensors
    if isinstance(obj, np.ndarray):
        return [convert_to_python_types(x) for x in obj.tolist()]
    elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return convert_to_python_types(obj.cpu().detach().numpy())
    
    # Handle collections
    if isinstance(obj, dict):
        return {str(k): convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(x) for x in obj]
    
    # Handle InsightFace-specific types (Face objects)
    if hasattr(obj, 'embedding') and hasattr(obj, 'bbox') and hasattr(obj, 'det_score'):
        result = {
            'bbox': convert_to_python_types(obj.bbox) if hasattr(obj, 'bbox') else None,
            'det_score': float(obj.det_score) if hasattr(obj, 'det_score') else 0.0,
            'landmark': convert_to_python_types(obj.landmark) if hasattr(obj, 'landmark') else None,
            'gender': int(obj.gender) if hasattr(obj, 'gender') else None,
            'age': float(obj.age) if hasattr(obj, 'age') else None,
        }
        return result
    
    # Try to convert objects with __dict__ attribute
    if hasattr(obj, '__dict__'):
        try:
            return convert_to_python_types(vars(obj))
        except:
            return str(obj)
    
    # Final fallback - convert to string
    try:
        return str(obj)
    except:
        return "Unconvertible object"