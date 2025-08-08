from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional, Dict, Any
import time
import uuid
import os
import logging
import aiohttp
from PIL import Image
import io
import traceback
import asyncio
from asyncio import Semaphore
from collections import defaultdict
import psutil
import threading
import shutil
import tempfile

from core.face_verifier import SenseFaceVerifier
from core.utils import convert_to_python_types
from core.temp_file_manager import temp_file_context
from config import settings

logger = logging.getLogger(__name__)


class SenseConcurrencyManager:
    """Manages concurrent face verification requests."""
    
    def __init__(self, max_concurrent: int = 5, max_queue: int = 50):
        self.semaphore = Semaphore(max_concurrent)
        self.max_queue = max_queue
        self.active_requests = {}
        self.verifier_pool = []
        self.pool_lock = threading.Lock()
        self._stats = defaultdict(int)
    
    async def sense_acquire_slot(self, reference_number: str) -> bool:
        """Acquire a processing slot for concurrent request."""
        if len(self.active_requests) >= self.max_queue:
            return False
        
        await self.semaphore.acquire()
        self.active_requests[reference_number] = {
            "start_time": time.time(),
            "status": "processing"
        }
        self._stats["total_requests"] += 1
        self._stats["active_requests"] = len(self.active_requests)
        return True
    
    def sense_release_slot(self, reference_number: str):
        """Release a processing slot."""
        if reference_number in self.active_requests:
            del self.active_requests[reference_number]
            self._stats["active_requests"] = len(self.active_requests)
            self._stats["completed_requests"] += 1
        
        self.semaphore.release()
    
    def sense_get_verifier(self) -> SenseFaceVerifier:
        """Get a verifier instance from pool or create new one."""
        with self.pool_lock:
            if self.verifier_pool:
                return self.verifier_pool.pop()
            else:
                return SenseFaceVerifier()
    
    def sense_return_verifier(self, verifier: SenseFaceVerifier):
        """Return verifier to pool for reuse."""
        with self.pool_lock:
            if len(self.verifier_pool) < 3:
                self.verifier_pool.append(verifier)
    
    def sense_get_stats(self) -> Dict[str, Any]:
        """Get concurrency statistics."""
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        return {
            "active_requests": len(self.active_requests),
            "total_requests": self._stats["total_requests"],
            "completed_requests": self._stats["completed_requests"],
            "queue_capacity": self.max_queue,
            "memory_usage_mb": round(memory_usage, 2),
            "verifier_pool_size": len(self.verifier_pool)
        }


# Global instance
sense_concurrency = SenseConcurrencyManager(
    max_concurrent=getattr(settings, 'MAX_CONCURRENT_REQUESTS', 5),
    max_queue=getattr(settings, 'MAX_QUEUE_SIZE', 50)
)


async def sense_save_uploaded_file_permanent(file: UploadFile) -> str:
    """Save uploaded file permanently without auto-deletion."""
    try:
        file_ext = "jpg"
        if file.filename and "." in file.filename:
            file_ext = file.filename.split(".")[-1].lower()
        upload_dir = os.path.join(os.getcwd(), "uploads")  # or "temp_uploads"

        # Create permanent temp file
        # fd, temp_path = tempfile.mkstemp(suffix=f".{file_ext}")
        fd, temp_path = tempfile.mkstemp(suffix=f".{file_ext}", dir=upload_dir)

        content = await file.read()
        
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(content)
        
        return str(temp_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {str(e)}")


async def sense_save_base64_image(base64_data: str) -> str:
    """Save base64 image to temporary file."""
    try:
        import base64
        import imghdr

        image_data = base64.b64decode(base64_data)
        file_ext = imghdr.what(None, h=image_data) or "unknown"

        # Create temp file
        upload_dir = os.path.join(os.getcwd(), "uploads") 
        fd, temp_path = tempfile.mkstemp(suffix=f".{file_ext}", dir=upload_dir)
        print(f"Temporary file created at: {temp_path}")
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(image_data)
        
        return str(temp_path)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


async def sense_get_image_from_url_or_base64(data: str, document_info="S3") -> str:
    """Handle both URL and base64 image data."""
    if(document_info == "S3"):
        # It's a URL
        return await sense_download_image_from_url(data)
    elif document_info == "BASE64":  # Likely base64
        # It's base64
        return await sense_save_base64_image(data)
    else:
        raise HTTPException(status_code=400, detail="Invalid image data - must be URL or base64")


async def sense_download_image_from_url(url: str, max_retries: int = 3) -> str:
    """Download image from URL with retry logic."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=getattr(settings, 'DOWNLOAD_TIMEOUT', 60))
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/jpeg,image/png,image/*,*/*;q=0.8"
            }
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    print(response)
                    
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Failed to download image from URL: {url}"
                        )
                    
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        raise HTTPException(
                            status_code=400, 
                            detail=f"URL does not point to an image"
                        )
                    
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > getattr(settings, 'MAX_IMAGE_SIZE', 20 * 1024 * 1024):
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Image too large"
                        )
                    
                    content = await response.read()
                    
                    # Validate image
                    img_buffer = io.BytesIO(content)
                    try:
                        img = Image.open(img_buffer)
                        img.verify()
                        img_buffer.seek(0)
                        img = Image.open(img_buffer)
                    except Exception as e:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Invalid image format"
                        )
                    
                    # Determine file extension
                    file_ext = 'jpg'
                    if img.format:
                        original_format = img.format.lower()
                        if original_format == 'jpeg':
                            file_ext = 'jpg'
                        elif original_format in ['png', 'gif', 'bmp', 'webp']:
                            file_ext = original_format
                    
                    # Save to permanent temp file
                    upload_dir = os.path.join(os.getcwd(), "uploads")  # or "temp_uploads"
                    fd, temp_path = tempfile.mkstemp(suffix=f".{file_ext}", dir=upload_dir)
                    # fd, temp_path = tempfile.mkstemp(suffix=f".{file_ext}")
                    
                    if file_ext == 'jpg' and img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    with os.fdopen(fd, "wb") as temp_file:
                        img.save(temp_file, format=img.format or 'JPEG')
                    
                    return str(temp_path)
                    
        except HTTPException:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = e
        except Exception as e:
            last_error = e

        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"Retrying in {wait_time}s due to error: {last_error}")
            await asyncio.sleep(wait_time)

    raise HTTPException(
        status_code=500, 
        detail=f"Failed to download image after {max_retries} attempts"
    )


async def sense_get_image_path_sync(file: Optional[UploadFile], url: Optional[str], field_name: str) -> str:
    """Get image path for sync endpoint (permanent storage)."""
    if file and url:
        raise HTTPException(
            status_code=400, 
            detail=f"Provide either {field_name} file or {field_name}_url, not both"
        )
    
    if not file and not url:
        raise HTTPException(
            status_code=400, 
            detail=f"Provide either {field_name} file or {field_name}_url"
        )
    
    if file:
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} must be an image file"
            )
        return await sense_save_uploaded_file_permanent(file)
    else:
        return await sense_download_image_from_url(url)


async def sense_get_image_path_async(file: Optional[UploadFile], url: Optional[str], temp_manager, field_name: str) -> str:
    """Get image path for async endpoint (temp storage)."""
    if file and url:
        raise HTTPException(
            status_code=400, 
            detail=f"Provide either {field_name} file or {field_name}_url, not both"
        )
    
    if not file and not url:
        raise HTTPException(
            status_code=400, 
            detail=f"Provide either {field_name} file or {field_name}_url"
        )
    
    if file:
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} must be an image file"
            )
        # Use temp manager for async
        file_ext = "jpg"
        if file.filename and "." in file.filename:
            file_ext = file.filename.split(".")[-1].lower()
        
        temp_path = temp_manager.create_temp_file(suffix=f".{file_ext}")
        content = await file.read()
        
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        
        return str(temp_path)
    else:
        # For URL, create temp file using temp manager
        temp_path = temp_manager.create_temp_file(suffix=".jpg")
        
        timeout = aiohttp.ClientTimeout(total=getattr(settings, 'DOWNLOAD_TIMEOUT', 60))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to download image")
                
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="URL does not point to an image")
                
                content = await response.read()
                img_buffer = io.BytesIO(content)
                img = Image.open(img_buffer)
                img.save(temp_path)
        
        return str(temp_path)


def sense_clean_response(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Clean response with match_score and confidence_level."""
    # Get confidence and ensure it's valid
    confidence = raw_result.get("confidence", 0.0)
    if confidence is None or str(confidence).lower() == 'nan':
        confidence = 0.0
    
    # Determine confidence level based on score
    if confidence >= 80:
        confidence_level = "HIGH"
    elif confidence >= 50:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"
    
    return {
        "verified": raw_result.get("verified", False),  # Keep original verified value
        "match_score": round(confidence, 2),
        "confidence_level": confidence_level,
        "distance": raw_result.get("distance", 1.0),
        "processing_time": raw_result.get("processing_time", 0.0),
        "api_version": raw_result.get("api_version", "1.0.0"),
        "doc_type": raw_result.get("doc_type", "unknown")
    }


async def send_callback(callback_url: str, reference_number: str, result: Dict[str, Any]):
    """Send the face-matching result to the callback URL."""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "reference_number": reference_number,
                "status_code": 200 if result.get("verified") else 400,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            logger.info(f"Sending callback to {callback_url} for reference {reference_number}")
            
            async with session.post(
                callback_url, 
                json=payload, 
                headers=headers, 
                timeout=getattr(settings, 'CALLBACK_TIMEOUT', 30)
            ) as response:
                if response.status == 200:
                    logger.info(f"Successfully sent callback to {callback_url} for reference {reference_number}")
                else:
                    response_text = await response.text()
                    logger.error(f"Callback to {callback_url} failed with status {response.status}: {response_text}")
    except aiohttp.ClientTimeout:
        logger.error(f"Callback to {callback_url} timed out after {getattr(settings, 'CALLBACK_TIMEOUT', 30)} seconds for reference {reference_number}")
    except Exception as e:
        logger.error(f"Failed to send callback to {callback_url} for reference {reference_number}: {str(e)}")
        logger.error(traceback.format_exc())


def sense_copy_file_for_background(source_path: str, reference_number: str) -> str:
    """Copy temp file for background processing."""
    try:
        file_ext = source_path.split('.')[-1] if '.' in source_path else 'jpg'
        dest_path = f"/tmp/face_match_{reference_number}_{uuid.uuid4().hex[:8]}.{file_ext}"
        
        shutil.copy2(source_path, dest_path)
        return dest_path
    except Exception as e:
        logger.error(f"Failed to copy file {source_path}: {e}")
        raise


async def sense_process_face_match_background(
    id_card_path: str,
    photo_path: str,
    doc_type: str,
    threshold: float,
    reference_number: str
):
    """Process face-matching in background."""
    start_time = time.time()
    verifier = None
    
    try:
        if not await sense_concurrency.sense_acquire_slot(reference_number):
            error_result = {
                "verified": False, 
                "error": "Server busy", 
                "processing_time": time.time() - start_time,
                "api_version": settings.API_VERSION
            }
            cleaned_error = sense_clean_response(error_result)
            await send_callback(settings.CALLBACK_URL, reference_number, cleaned_error)
            return
        
        verifier = sense_concurrency.sense_get_verifier()
        raw_result = verifier.verify(id_card_path, photo_path, threshold, doc_type)
        
        raw_result["api_version"] = settings.API_VERSION
        raw_result["processing_time"] = time.time() - start_time
        
        result = convert_to_python_types(raw_result)
        cleaned_result = sense_clean_response(result)
        
        await send_callback(settings.CALLBACK_URL, reference_number, cleaned_result)
        
    except Exception as e:
        logger.error(f"Background processing error: {str(e)}")
        error_result = {
            "verified": False, 
            "error": f"Processing error: {str(e)}", 
            "processing_time": time.time() - start_time, 
            "api_version": settings.API_VERSION
        }
        cleaned_error = sense_clean_response(error_result)
        await send_callback(settings.CALLBACK_URL, reference_number, cleaned_error)
    
    finally:
        # Clean up copied files
        try:
            if os.path.exists(id_card_path):
                os.unlink(id_card_path)
            if os.path.exists(photo_path):
                os.unlink(photo_path)
        except Exception as e:
            logger.error(f"Failed to cleanup files: {e}")
        
        if verifier:
            sense_concurrency.sense_return_verifier(verifier)
        sense_concurrency.sense_release_slot(reference_number)


def register_endpoints(app: FastAPI):
    """Register all endpoints to the FastAPI app."""
    
    @app.post("/face-match")
    async def face_match_sync(
        id_card: Optional[UploadFile] = File(None, description="Document image file"),
        photo: Optional[UploadFile] = File(None, description="Selfie image file"),
        doc_type: str = Form("unknown")
    ):
        """Synchronous face matching endpoint (form-data upload)."""
        try:
            # Use permanent file storage (no auto-deletion)
            id_card_path = await sense_get_image_path_sync(id_card, None, "document")
            photo_path = await sense_get_image_path_sync(photo, None, "photo")
            
            threshold = settings.DOCUMENT_THRESHOLDS.get(doc_type, 0.4)
            
            if len(sense_concurrency.active_requests) >= sense_concurrency.max_queue:
                raise HTTPException(
                    status_code=503,
                    detail="Server busy, please try again later"
                )
            
            start_time = time.time()
            verifier = sense_concurrency.sense_get_verifier()
            
            try:
                raw_result = verifier.verify(id_card_path, photo_path, threshold, doc_type)
                raw_result["api_version"] = settings.API_VERSION
                raw_result["processing_time"] = time.time() - start_time
                
                result = convert_to_python_types(raw_result)
                cleaned_result = sense_clean_response(result)
                
                return cleaned_result
            
            finally:
                sense_concurrency.sense_return_verifier(verifier)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "verified": False,
                    "error": f"API error: {str(e)}",
                    "api_version": settings.API_VERSION
                }
            )

    @app.post("/api/face-match")
    async def face_match_with_async_support(
        request: Request, 
        background_tasks: BackgroundTasks
    ):
        try:
            post_data = await request.json()
            
            id_card_url = post_data.get("id_card_url", "")
            photo_url = post_data.get("photo_url", "")
            doc_type = post_data.get("doc_type", "unknown")
            document_info = post_data.get("document_info", "S3")
            # NEW: Async processing parameters
            source = post_data.get("source", None)
            reference_number = post_data.get("reference_number", None)
            doc_type1 = 'unknown'
            if not id_card_url or not photo_url:
                raise HTTPException(
                    status_code=400,
                    detail="Both id_card_url and photo_url are required"
                )
            
            # Generate reference number if not provided
            if not reference_number:
                reference_number = f"api_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"API Face match request - doc_type: {doc_type}, async: {bool(source and source.upper().strip() == 'API' and reference_number)}")
            
            # Handle both URL and base64 images
            id_card_path = await sense_get_image_from_url_or_base64(id_card_url, document_info)
            photo_path = await sense_get_image_from_url_or_base64(photo_url)
            
            threshold = settings.DOCUMENT_THRESHOLDS.get(doc_type1, 0.35)
            
            # 🚀 CHECK FOR ASYNC PROCESSING
            if source and source.upper().strip() == 'API' and reference_number:
                # ASYNC REQUEST: Return immediate "in_progress" response
                logger.info(f"Processing ASYNC API request with reference: {reference_number}")
                                
                background_tasks.add_task(
                    sense_process_face_match_background,
                    id_card_path,
                    photo_path,
                    doc_type,
                    threshold,
                    reference_number
                )
                
                return JSONResponse(
                    status_code=202,  
                    content={
                        "status": "in_progress",
                        "reference_number": reference_number,
                        "message": "Face matching is being processed. Results will be sent to the callback URL.",
                        "api_version": settings.API_VERSION,
                        "queue_position": len(sense_concurrency.active_requests)
                    }
                )
            else:
                # SYNC PROCESSING: Return immediate result
                start_time = time.time()
                verifier = sense_concurrency.sense_get_verifier()
                
                try:
                    # Process immediately and return result
                    raw_result = verifier.verify(id_card_path, photo_path, threshold, doc_type)
                    raw_result["api_version"] = settings.API_VERSION
                    raw_result["processing_time"] = time.time() - start_time
                    
                    result = convert_to_python_types(raw_result)
                    cleaned_result = sense_clean_response(result)
                    
                    return cleaned_result
                
                finally:
                    sense_concurrency.sense_return_verifier(verifier)
                    # Clean up downloaded files
                    try:
                        if os.path.exists(id_card_path):
                            os.unlink(id_card_path)
                        if os.path.exists(photo_path):
                            os.unlink(photo_path)
                    except:
                        pass
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "verified": False,
                    "error": f"API error: {str(e)}",
                    "api_version": settings.API_VERSION
                }
            )

    @app.delete("/api/cleanup/temp")
    async def cleanup_temp_files():
        """Clean up temporary files."""
        try:
            temp_count = 0
            total_size = 0
            
            if hasattr(settings, 'TEMP_DIR') and settings.TEMP_DIR.exists():
                files = list(settings.TEMP_DIR.glob("*"))
                for file_path in files:
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_path.unlink()
                        temp_count += 1
            
            # Clean up background task files
            import glob
            bg_files = glob.glob("/tmp/face_match_*")
            for file_path in bg_files:
                try:
                    total_size += os.path.getsize(file_path)
                    os.unlink(file_path)
                    temp_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
            
            size_mb = total_size / (1024 * 1024)
            return {
                "status": "success",
                "message": f"Cleaned up {temp_count} temporary files",
                "deleted_files": temp_count,
                "space_freed_mb": round(size_mb, 2)
            }
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

    @app.get("/api/stats")
    async def sense_get_stats():
        """Get current concurrency and system statistics."""
        stats = sense_concurrency.sense_get_stats()
        
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            stats.update({
                "system": {
                    "memory_percent": round(memory.percent, 2),
                    "cpu_percent": round(cpu_percent, 2),
                    "available_memory_mb": round(memory.available / 1024 / 1024, 2)
                }
            })
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
        
        return stats

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        try:
            verifier = SenseFaceVerifier()
            verification_status = "operational"
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            verification_status = "error"
        
        try:
            temp_count = len(list(settings.TEMP_DIR.glob("*"))) if hasattr(settings, 'TEMP_DIR') and settings.TEMP_DIR.exists() else 0
        except:
            temp_count = 0
        
        concurrency_stats = sense_concurrency.sense_get_stats()
        
        return {
            "status": "healthy",
            "version": settings.API_VERSION,
            "verification_engine": verification_status,
            "timestamp": datetime.now().isoformat(),
            "supported_doc_types": list(settings.DOCUMENT_THRESHOLDS.keys()),
            "system": {
                "temp_files": temp_count,
                "max_upload_size": f"{(settings.MAX_UPLOAD_SIZE or 0) / 1024 / 1024:.1f}MB"
            },
            "concurrency": concurrency_stats
        }

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Face Verification API",
            "version": settings.API_VERSION,
            "docs": "/docs",
            "health": "/api/health",
            "stats": "/api/stats"
        }
    
    logger.info("Face Verification API endpoints registered successfully")