from pydantic import BaseModel
from typing import Optional, Dict, List

class FaceDetection(BaseModel):
    id_detected: bool
    id_bbox: Optional[List[int]]
    id_confidence: float
    photo_detected: bool
    photo_bbox: Optional[List[int]]
    photo_confidence: float

class FaceMatchResponse(BaseModel):
    verified: bool
    distance: float
    threshold: float
    confidence: float
    id_image: Optional[str]
    photo_image: Optional[str]
    time: float
    doc_type: str
    error: Optional[str]
    face_detection: Optional[FaceDetection]
    api_version: str
    models: dict
    error: Optional[str] = None
    #models: Dict[str, str]
    #comparisons: Optional[Dict[str, float]]  # Added for non-deep learning methods