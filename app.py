from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import shutil
import cv2
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
    allow_origins=["*"],  # Consider restricting in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

# Preprocessing function to handle blurry images
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)

        # Skip if image not loaded
        if image is None:
            raise ValueError("Failed to load image for preprocessing.")

        # Sharpening kernel
        kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)

        # Convert to grayscale and apply histogram equalization
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        # Convert back to 3-channel image
        final = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

        # Save preprocessed image
        preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg')
        cv2.imwrite(preprocessed_path, final)
        logger.info(f"Preprocessed image saved to: {preprocessed_path}")
        return preprocessed_path
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        return image_path  # fallback
    
    

@app.post("/verify/")
async def verify(img1: UploadFile = File(..., description="Selfie image file"),
                 img2: UploadFile = File(..., description="Document image file")):
    """
    Verifies if the face in the selfie image matches the face in the document image
    using DeepFace with ArcFace model and retinaface detector.
    """
    img1_path = None
    img2_path = None
    temp_files_to_clean = []

    try:
        # Save uploaded files to temp paths
        _, img1_ext = os.path.splitext(img1.filename)
        _, img2_ext = os.path.splitext(img2.filename)
        img1_ext = img1_ext or '.jpg'
        img2_ext = img2_ext or '.jpg'

        with tempfile.NamedTemporaryFile(delete=False, suffix=img1_ext) as temp1:
            shutil.copyfileobj(img1.file, temp1)
            img1_path = temp1.name
            temp_files_to_clean.append(img1_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=img2_ext) as temp2:
            shutil.copyfileobj(img2.file, temp2)
            img2_path = temp2.name
            temp_files_to_clean.append(img2_path)

        logger.info(f"Saved images: {img1_path}, {img2_path}")

        # Preprocess both images
        img1_preprocessed = preprocess_image(img1_path)
        img2_preprocessed = preprocess_image(img2_path)
        temp_files_to_clean.extend([img1_preprocessed, img2_preprocessed])

        logger.info("Starting DeepFace verification...")

        try:
            result = DeepFace.verify(
                img1_path=img1_preprocessed,
                img2_path=img2_preprocessed,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=True,
                align=True
            )
            logger.info(f"Verification result: {result}")

        except ValueError as ve:
            logger.error(f"Face detection error: {ve}", exc_info=True)
            detail_msg = "Face could not be detected or aligned. Please upload clear images."
            if "img1" in str(ve):
                detail_msg = f"Face not detected in selfie image ('{img1.filename}')."
            elif "img2" in str(ve):
                detail_msg = f"Face not detected in document image ('{img2.filename}')."
            raise HTTPException(status_code=400, detail=detail_msg)
        except Exception as e:
            logger.error(f"Unexpected error during verification: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="An error occurred during face comparison.")

        return {"result": result}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error("Unexpected server error.", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
    finally:
        logger.info("Cleaning up temporary files...")
        for path in temp_files_to_clean:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Deleted: {path}")
            except Exception as e:
                logger.warning(f"Could not delete temp file: {path}. Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)