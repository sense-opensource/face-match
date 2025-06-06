from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from datetime import datetime
from .models import FaceMatchResponse
from core.face_verifier import SenseFaceVerifier
from core.utils import save_uploaded_file, convert_to_python_types
from core.temp_file_manager import temp_file_context
from config import settings
import time
import shutil
import uuid
import os
import logging

logger = logging.getLogger(__name__)

def register_endpoints(app: FastAPI):
    @app.post("/face-match")
    async def face_match(
        background_tasks: BackgroundTasks,
        id_card: UploadFile = File(...),
        photo: UploadFile = File(...),
        doc_type: str = Form("unknown")
    ):
        try:
            with temp_file_context() as temp_manager:
                # Save uploaded files
                id_card_path = save_uploaded_file(id_card, temp_manager)
                photo_path = save_uploaded_file(photo, temp_manager)
                
                # Get threshold for document type
                threshold = settings.DOCUMENT_THRESHOLDS.get(doc_type, 0.35)
                
                # Verify faces
                verifier = SenseFaceVerifier()
                raw_result = verifier.verify(id_card_path, photo_path, threshold, doc_type)
                # Add version info and available models
                
                # raw_result["models"] = {
                #     "face_detection": verifier.face_detector.detector_type,
                #     "face_recognition": verifier.face_recognizer.recognizer_type,
                #     "insightface": verifier.insightface_app is not None,
                #     "error": ""  
                # }
                # Convert to Python native types
                result = convert_to_python_types(raw_result)
                # Fallback if conversion fails
                if not isinstance(result, dict):
                    logger.warning("Conversion failed, using fallback")
                    result = {
                        "verified": bool(raw_result.get("verified", False)),
                        "confidence": float(raw_result.get("confidence", 0.0)),
                        "distance": float(raw_result.get("distance", 1.0)),
                        "threshold": float(threshold),
                        "id_image": raw_result.get("id_image"),
                        "photo_image": raw_result.get("photo_image"),
                        "api_version": settings.API_VERSION,
                        "error": "Data conversion issue, simplified result provided"
                    }
               
                # Background cleanup task
                async def delayed_cleanup():
                    try:
                        current_time = time.time()
                        for file_path in settings.UPLOADS_DIR.glob("*"):
                            if current_time - file_path.stat().st_mtime > 86400:
                                os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Cleanup error: {str(e)}")
                
                background_tasks.add_task(delayed_cleanup)
                return result
            
        except Exception as e:
            import traceback
            logger.error(f"API error: {str(e)}")
            print("🔥 Internal Server Error:", traceback.format_exc())
            return JSONResponse(status_code=550, content={"error": str(e)})
            return {
                "verified": False,
                "error": f"API error: {str(e)}",
                "doc_type": doc_type,
                "api_version": settings.API_VERSION
            }

    @app.get("/uploads/{filename}")
    async def serve_uploaded_file(filename: str):
        file_path = settings.UPLOADS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": settings.API_VERSION,
            "mode": "high-level",
            "available_models": {
                "dlib": False,  # Update based on actual availability
                "insightface": False,
                "retinaface": False,
                "mtcnn": False,
                "arcface": False,
                "facenet": False,
                "tensorflow": False,
                "pytorch": False
            },
            "detector_type": "haarcascade",  # Update dynamically if needed
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/")
    async def serve_frontend():
        index_path = settings.STATIC_DIR / "index.html"
        if not index_path.exists():
            with open(index_path, "w") as f:
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>high-Level Face Verification API</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        h1 { color: #3f51b5; }
                        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    </style>
                </head>
                <body>
                    <h1>Sense Face Verification API</h1>
                    <div class="card">
                        <h2>API is running</h2>
                        <p>The Face Verification API is running with:</p>
                        <ul>
                            <li>RetinaFace and MTCNN for state-of-the-art face detection</li>
                            <li>ArcFace and FaceNet for accurate face recognition</li>
                            <li>Advanced image enhancement for low-light and poor quality images</li>
                            <li>Adaptive quality-based verification</li>
                            <li>Commercial-grade accuracy similar to high</li>
                        </ul>
                        <p>Use the <code>/face-match</code> endpoint to verify faces.</p>
                    </div>
                </body>
                </html>
                """)
        return FileResponse(index_path)