from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import shutil
import cv2 # Still useful for potential future preprocessing, but not for cropping here
import numpy as np
import os
import tempfile
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider restricting this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- (Removed detect_and_crop_face function) ---

@app.post("/verify/")
async def verify(img1: UploadFile = File(..., description="Selfie image file"),
                 img2: UploadFile = File(..., description="Document image file")):
    """
    Verifies if the face in the selfie image matches the face in the document image
    using DeepFace with high-accuracy settings (ArcFace + RetinaFace).
    """
    img1_path = None
    img2_path = None
    temp_files_to_clean = []

    try:
        # --- Save uploaded files temporarily ---
        # Use tempfile to avoid issues with original filenames
        _, img1_ext = os.path.splitext(img1.filename)
        _, img2_ext = os.path.splitext(img2.filename)
        img1_ext = img1_ext if img1_ext else '.jpg' # Default to jpg if no ext
        img2_ext = img2_ext if img2_ext else '.jpg'

        with tempfile.NamedTemporaryFile(delete=False, suffix=img1_ext) as temp1:
            shutil.copyfileobj(img1.file, temp1)
            img1_path = temp1.name
            temp_files_to_clean.append(img1_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=img2_ext) as temp2:
            shutil.copyfileobj(img2.file, temp2)
            img2_path = temp2.name
            temp_files_to_clean.append(img2_path)

        logger.info(f"Saved uploaded files to temporary paths: {img1_path}, {img2_path}")

        # --- Verify using DeepFace with High-Accuracy Settings ---
        logger.info("Attempting verification with ArcFace model and retinaface detector...")
        try:
            # Use ArcFace model and retinaface detector for potentially higher accuracy
            # enforce_detection=True ensures faces are found by retinaface before comparison
            result = DeepFace.verify(
                img1_path=img1_path,      # Pass the full image path
                img2_path=img2_path,      # Pass the full image path
                model_name="ArcFace",
                detector_backend="retinaface", # Use a strong detector
                enforce_detection=True,   # Crucial: Fail if detector doesn't find a face
                align=True                # Ensure faces are aligned (usually default True)
            )
            # Log the detailed result including distance and threshold
            logger.info(f"DeepFace verification completed. Result details: {result}")

            # Optional: Check distance for finer control (Example)
            # distance = result.get('distance', float('inf'))
            # threshold = result.get('threshold', 0.68) # Default ArcFace threshold
            # custom_threshold = 0.60 # Example: A stricter threshold you might determine
            # is_verified_custom = distance <= custom_threshold
            # logger.info(f"Distance: {distance:.4f}, Threshold: {threshold}, Custom Verified (thr={custom_threshold}): {is_verified_custom}")
            # You could return is_verified_custom instead of result['verified'] if needed

        except ValueError as ve:
             # This is often triggered by enforce_detection=True if a face isn't found
             logger.error(f"DeepFace.verify raised ValueError (likely no face detected by retinaface): {ve}", exc_info=True)
             if "Face could not be detected" in str(ve) or "cannot be aligned" in str(ve):
                 # Be more specific about which image failed if possible (check exception string)
                 detail_msg = "Face could not be detected or aligned properly in one or both images using retinaface detector. Ensure faces are clear and unobstructed."
                 if "img1" in str(ve):
                     detail_msg = f"Face could not be detected or aligned properly in selfie image ('{img1.filename}') using retinaface detector."
                 elif "img2" in str(ve):
                      detail_msg = f"Face could not be detected or aligned properly in document image ('{img2.filename}') using retinaface detector."
                 raise HTTPException(status_code=400, detail=detail_msg)
             else:
                 # Other ValueErrors
                 raise HTTPException(status_code=400, detail=f"Image processing error: {ve}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during DeepFace.verify: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="An error occurred during the face comparison process.")

        # Return the full verification result dictionary
        # The 'verified' key indicates the result based on the model's default threshold
        return {"result": result}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors (e.g., file saving)
        logger.error("An unexpected error occurred in the /verify endpoint.", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")

    finally:
        # --- Cleanup temporary files ---
        logger.info("Cleaning up temporary files...")
        for path in temp_files_to_clean:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Removed temporary file: {path}")
                except Exception as e:
                    logger.error(f"Error removing temporary file {path}: {e}", exc_info=True)


if __name__ == "__main__":
    import uvicorn
    # Recommended: uvicorn server:app --host 127.0.0.1 --port 8000 --reload
    uvicorn.run(app, host="127.0.0.1", port=8000)

