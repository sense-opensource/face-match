from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ROOT_DIR: Path = Path(__file__).parent
    MODELS_DIR: Path = ROOT_DIR / "models"
    TEMP_DIR: Path = ROOT_DIR / "temp_files"
    UPLOADS_DIR: Path = ROOT_DIR / "uploads"
    STATIC_DIR: Path = ROOT_DIR / "static"
    LOG_DIR: Path = ROOT_DIR / "logs"
    API_VERSION: str = "1.0.0"
    DOCUMENT_THRESHOLDS: dict = {
        "passport": 0.25,
        "national_id": 0.30,
        "drivers_license": 0.35,
        "voter_id": 0.40,
        "pan_card": 0.40,
        "aadhaar": 0.35,
        "unknown": 0.35
    }
    ENHANCEMENT_PARAMS: dict = {
        "passport": {
            "brightness_boost": 1.3,
            "contrast_boost": 1.4,
            "denoise_strength": 5,
            "sharpen_strength": 1.2,
            "gamma": 1.2
        },
        # ... other enhancement parameters from original script
        "default": {
            "brightness_boost": 1.3,
            "contrast_boost": 1.3,
            "denoise_strength": 5,
            "sharpen_strength": 1.1,
            "gamma": 1.2
        }
    }
    

settings = Settings()