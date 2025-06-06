from .image_enhancer import AdvancedImageEnhancer
from .face_detector import MultiFaceDetector
from .face_recognizer import FaceRecognizer
from config import settings
from typing import Dict, Any
from core.temp_file_manager import temp_file_context
from core.utils import compute_histogram_similarity, compute_structural_similarity, cosine_distance
import time
import logging
import cv2
import uuid
import shutil
import numpy as np
logger = logging.getLogger(__name__)
UPLOADS_DIR = settings.UPLOADS_DIR
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
class SenseFaceVerifier:
    """Advanced face verification using multiple techniques"""
    
    def __init__(self):
        self.face_detector = MultiFaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.image_enhancer = AdvancedImageEnhancer(settings.MODELS_DIR)
        
        # InsightFace app for combined detection and recognition if available
        self.insightface_app = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self.insightface_app = FaceAnalysis(name='buffalo_l')
                self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("Using InsightFace for combined detection and recognition")
            except Exception as e:
                logger.warning(f"Could not initialize InsightFace app: {e}")

    def verify_with_insightface(self, img1_path: str, img2_path: str, threshold: float = 0.4) -> Dict[str, Any]:
        """Verify using InsightFace's end-to-end pipeline"""
        try:
            # Read images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                return {"verified": False, "error": "Could not read images"}
            
            # Get face embeddings
            faces1 = self.insightface_app.get(img1)
            faces2 = self.insightface_app.get(img2)
            
            if len(faces1) == 0 or len(faces2) == 0:
                return {
                    "verified": False, 
                    "error": "Face not detected", 
                    "face_detection": {
                        "id_detected": len(faces1) > 0,
                        "photo_detected": len(faces2) > 0
                    }
                }
            
            # Get best faces
            face1 = faces1[0]
            face2 = faces2[0]
            
            # Get embeddings
            embedding1 = face1.embedding if hasattr(face1, 'embedding') else None
            embedding2 = face2.embedding if hasattr(face2, 'embedding') else None
            
            if embedding1 is None or embedding2 is None:
                return {"verified": False, "error": "Could not extract face embeddings"}
            
            # Compute cosine similarity
            sim = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            # Convert to distance (0 to 1 range)
            distance = float(1.0 - (sim + 1) / 2)
            
            # Determine if verified - convert to primitive bool
            verified = bool(distance < threshold)
            
            # Calculate confidence
            confidence = float(max(0, min(100, (1 - distance / threshold) * 100)))
            
            # Extract bounding boxes
            bbox1 = face1.bbox.astype(int) if hasattr(face1, 'bbox') and face1.bbox is not None else [0, 0, 0, 0]
            bbox2 = face2.bbox.astype(int) if hasattr(face2, 'bbox') and face2.bbox is not None else [0, 0, 0, 0]
            
            # Create result using only primitive types
            result = {
                "verified": verified,  # primitive bool
                "distance": float(distance),  # primitive float
                "confidence": float(confidence),  # primitive float
                "face_detection": {
                    "id_detected": True,  # primitive bool
                    "id_bbox": bbox1.tolist() if hasattr(bbox1, 'tolist') else list(bbox1),  # primitive list
                    "id_confidence": float(face1.det_score) if hasattr(face1, 'det_score') else 0.0,  # primitive float
                    "photo_detected": True,  # primitive bool
                    "photo_bbox": bbox2.tolist() if hasattr(bbox2, 'tolist') else list(bbox2),  # primitive list
                    "photo_confidence": float(face2.det_score) if hasattr(face2, 'det_score') else 0.0  # primitive float
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"InsightFace verification error: {str(e)}")
            return {"verified": False, "error": f"InsightFace verification error: {str(e)}"}
            
    def verify(self, img1_path: str, img2_path: str, threshold: float = 0.4, doc_type: str = "unknown") -> Dict[str, Any]:
        """Verify if two images contain the same face using best available method"""
        try:
            start_time = time.time()
            
            with temp_file_context() as temp_manager:
                # Try InsightFace's end-to-end pipeline first if available
                if INSIGHTFACE_AVAILABLE and self.insightface_app is not None:
                    logger.info("Using InsightFace for verification")
                    
                    # Step 1: Image enhancement
                    logger.info(f"Enhancing ID document image: {img1_path}")
                    img1_enhanced = self.image_enhancer.enhance_for_face_verification(
                        img1_path, is_id_document=True
                    )
                    
                    logger.info(f"Enhancing selfie/photo image: {img2_path}")
                    img2_enhanced = self.image_enhancer.enhance_for_face_verification(
                        img2_path, is_id_document=False
                    )
                    
                    # Step 2: Verification with InsightFace
                    insightface_result = self.verify_with_insightface(
                        img1_enhanced, img2_enhanced, threshold
                    )
                    
                    # If successful, use InsightFace result
                    if "error" not in insightface_result:
                        logger.info("InsightFace verification successful")
                        
                        # Save the enhanced images for display
                        img1_id = str(uuid.uuid4())
                        img2_id = str(uuid.uuid4())
                        
                        img1_upload_path = UPLOADS_DIR / f"{img1_id}.jpg"
                        img2_upload_path = UPLOADS_DIR / f"{img2_id}.jpg"
                        logger.info(img1_upload_path)
                        # Save the enhanced images
                        shutil.copy(img1_enhanced, img1_upload_path)
                        shutil.copy(img2_enhanced, img2_upload_path)
                        
                        id_image = f"/uploads/{img1_id}.jpg"
                        photo_image = f"/uploads/{img2_id}.jpg"
                        
                        # Add missing fields to InsightFace result using only primitive types
                        insightface_result["id_image"] = id_image
                        insightface_result["photo_image"] = photo_image
                        insightface_result["threshold"] = float(threshold)
                        insightface_result["time"] = float(time.time() - start_time)
                        insightface_result["doc_type"] = doc_type
                        # Manually ensure verified is a Python bool, not numpy.bool_
                        if "verified" in insightface_result:
                            insightface_result["verified"] = bool(insightface_result["verified"])
                        
                        return insightface_result
                    
                    logger.warning(f"InsightFace verification failed: {insightface_result.get('error', 'Unknown error')}")
                    # Fall back to manual pipeline
                
                # Manual pipeline if InsightFace failed or is not available
                logger.info("Using manual verification pipeline")
                
                # Step 1: Advanced image enhancement
                logger.info(f"Enhancing ID document image: {img1_path}")
                img1_enhanced = self.image_enhancer.enhance_for_face_verification(
                    img1_path, is_id_document=True
                )
                
                logger.info(f"Enhancing selfie/photo image: {img2_path}")
                img2_enhanced = self.image_enhancer.enhance_for_face_verification(
                    img2_path, is_id_document=False
                )
                
                # Step 2: Face detection on enhanced images
                logger.info("Detecting face in ID document")
                img1_face = self.face_detector.get_best_face(img1_enhanced)
                
                logger.info("Detecting face in selfie/photo")
                img2_face = self.face_detector.get_best_face(img2_enhanced)
                
                # Check if face detection failed on either image
                face1_detected = img1_face is not None
                face2_detected = img2_face is not None
                
                # If face detection failed on either image, verification fails
                if not face1_detected or not face2_detected:
                    logger.warning(f"Face detection failed. ID: {face1_detected}, Photo: {face2_detected}")
                    
                    # Save the enhanced images for display
                    img1_id = str(uuid.uuid4())
                    img2_id = str(uuid.uuid4())
                    
                    img1_upload_path = UPLOADS_DIR / f"{img1_id}.jpg"
                    img2_upload_path = UPLOADS_DIR / f"{img2_id}.jpg"
                    
                    # Save the enhanced images
                    shutil.copy(img1_enhanced, img1_upload_path)
                    shutil.copy(img2_enhanced, img2_upload_path)
                    
                    id_image = f"/uploads/{img1_id}.jpg"
                    photo_image = f"/uploads/{img2_id}.jpg"
                    
                    return {
                        "verified": False,
                        "distance": 1.0,
                        "threshold": threshold,
                        "confidence": 0.0,
                        "id_image": id_image,
                        "photo_image": photo_image,
                        "time": float(time.time() - start_time),
                        "error": "Face detection failed",
                        "face_detection": {
                            "id_detected": face1_detected,
                            "photo_detected": face2_detected
                        }
                    }
                
                # Step 3: Crop faces
                img1_face_bbox = img1_face["bbox"]
                img2_face_bbox = img2_face["bbox"]
                
                # Read enhanced images
                img1 = cv2.imread(img1_enhanced)
                img2 = cv2.imread(img2_enhanced)
                
                # Crop with padding
                x1, y1, x2, y2 = img1_face_bbox
                padding_x = int((x2 - x1) * 0.15)
                padding_y = int((y2 - y1) * 0.15)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(img1.shape[1], x2 + padding_x)
                y2 = min(img1.shape[0], y2 + padding_y)
                
                face1 = img1[y1:y2, x1:x2]
                
                x1, y1, x2, y2 = img2_face_bbox
                padding_x = int((x2 - x1) * 0.15)
                padding_y = int((y2 - y1) * 0.15)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(img2.shape[1], x2 + padding_x)
                y2 = min(img2.shape[0], y2 + padding_y)
                
                face2 = img2[y1:y2, x1:x2]
                
                # Save cropped faces
                face1_path = temp_manager.create_temp_file(suffix='_face1.jpg')
                face2_path = temp_manager.create_temp_file(suffix='_face2.jpg')
                
                cv2.imwrite(face1_path, face1)
                cv2.imwrite(face2_path, face2)
                
                # Save the cropped faces for display
                img1_id = str(uuid.uuid4())
                img2_id = str(uuid.uuid4())
                
                img1_upload_path = UPLOADS_DIR / f"{img1_id}.jpg"
                img2_upload_path = UPLOADS_DIR / f"{img2_id}.jpg"
                
                shutil.copy(face1_path, img1_upload_path)
                shutil.copy(face2_path, img2_upload_path)
                
                id_image = f"/uploads/{img1_id}.jpg"
                photo_image = f"/uploads/{img2_id}.jpg"
                
                # Step 4: Feature extraction
                features1 = self.face_recognizer.extract_features(face1)
                features2 = self.face_recognizer.extract_features(face2)
                
                # Check if feature extraction failed
                if features1 is None or features2 is None:
                    logger.warning("Feature extraction failed")
                    return {
                        "verified": False,
                        "distance": 1.0,
                        "threshold": threshold,
                        "confidence": 0.0,
                        "id_image": id_image,
                        "photo_image": photo_image,
                        "time": float(time.time() - start_time),
                        "error": "Feature extraction failed",
                        "doc_type": doc_type
                    }
                
                # Step 5: Additional comparison methods for robustness
                
                # 1. Feature-based comparison
                if self.face_recognizer.recognizer_type in ["arcface", "facenet"]:
                    # Deep learning features are reliable on their own
                    feature_distance = cosine_distance(features1, features2)
                    combined_distance = feature_distance
                else:
                    # 1. Histogram comparison
                    hist_distance = compute_histogram_similarity(face1_path, face2_path)
                    
                    # 2. Structural similarity
                    ssim_distance = compute_structural_similarity(face1_path, face2_path)
                    
                    # 3. HOG features comparison
                    feature_distance = cosine_distance(features1, features2)
                    
                    # 4. LBP (Local Binary Patterns) for texture
                    lbp_distance = 0.5  # Default if calculation fails
                    try:
                        # Simple LBP implementation
                        def calculate_lbp(image, radius=1, neighbors=8):
                            if len(image.shape) > 2:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                
                            h, w = image.shape
                            lbp = np.zeros((h, w), dtype=np.uint8)
                            
                            for y in range(radius, h-radius):
                                for x in range(radius, w-radius):
                                    center = image[y, x]
                                    binary_code = 0
                                    
                                    # Compare with neighbors
                                    for n in range(neighbors):
                                        # Calculate neighbor coordinates
                                        theta = 2 * np.pi * n / neighbors
                                        x_n = x + int(radius * np.cos(theta))
                                        y_n = y + int(radius * np.sin(theta))
                                        
                                        # Compare with center
                                        if image[y_n, x_n] >= center:
                                            binary_code += 1 << n
                                    
                                    lbp[y, x] = binary_code
                                    
                            return lbp
                        
                        # Calculate LBP for both images
                        lbp1 = calculate_lbp(face1)
                        lbp2 = calculate_lbp(face2)
                        
                        # Calculate histograms
                        hist1, _ = np.histogram(lbp1.flatten(), bins=256, range=(0, 256))
                        hist2, _ = np.histogram(lbp2.flatten(), bins=256, range=(0, 256))
                        
                        # Normalize
                        hist1 = hist1.astype(float) / np.sum(hist1)
                        hist2 = hist2.astype(float) / np.sum(hist2)
                        
                        # Calculate distance (Bhattacharyya distance)
                        lbp_distance = 1.0 - np.sum(np.sqrt(hist1 * hist2))
                    except Exception as e:
                        logger.error(f"Error in LBP calculation: {str(e)}")
                    
                    # Combine distances with weights - give more weight to HOG features
                    combined_distance = (
                        hist_distance * 0.15 + 
                        ssim_distance * 0.15 + 
                        feature_distance * 0.5 +
                        lbp_distance * 0.2
                    )
                
                # Verify against threshold - ensure we get a Python bool, not numpy.bool_
                verified = bool(combined_distance < threshold)
                
                # Calculate confidence percentage (inversely related to distance)
                confidence = float(max(0, min(100, (1 - combined_distance / threshold) * 100)))
                
                processing_time = float(time.time() - start_time)
                
                # Create result with detailed information
                result = {
                    "verified": verified,
                    "distance": float(combined_distance),
                    "threshold": float(threshold),
                    "confidence": float(confidence),
                    "id_image": id_image,
                    "photo_image": photo_image,
                    "time": float(processing_time),
                    "doc_type": doc_type,
                    "face_detection": {
                        "id_type": img1_face.get("type", "unknown"),
                        "id_confidence": float(img1_face.get("confidence", 0.0)),
                        "photo_type": img2_face.get("type", "unknown"),
                        "photo_confidence": float(img2_face.get("confidence", 0.0))
                    },
                    "method": self.face_recognizer.recognizer_type,
                    "error":"none"
                }
                
                # Add comparison details if not using deep learning features
                if self.face_recognizer.recognizer_type not in ["arcface", "facenet"]:
                    result["comparisons"] = {
                        "histogram": float(hist_distance),
                        "structural": float(ssim_distance),
                        "feature": float(feature_distance),
                        "texture": float(lbp_distance)
                    }
                
                # Return converted result to ensure no numpy types
                return result
                
        except Exception as e:
            logger.error(f"Face verification error: {str(e)}")
            # Return error with minimal information
            return {
                "verified": False,
                "error": f"Verification error: {str(e)}",
                "distance": 1.0,
                "threshold": float(threshold),
                "confidence": 0.0,
                "time": float(time.time() - start_time),
                "doc_type": doc_type
            }