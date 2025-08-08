from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Static paths
    ROOT_DIR: Path = Path(__file__).parent
    MODELS_DIR: Path = ROOT_DIR / "models"
    TEMP_DIR: Path = ROOT_DIR / "temp_files"
    UPLOADS_DIR: Path = ROOT_DIR / "uploads"
    STATIC_DIR: Path = ROOT_DIR / "static"
    LOG_DIR: Path = ROOT_DIR / "logs"

    # From .env with defaults
    CALLBACK_URL: str = os.getenv("CALLBACK_URL", "http://localhost:3026/api/sense/face-match-webhook")
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")
    CALLBACK_TIMEOUT: int = os.getenv("CALLBACK_TIMEOUT", 30)
    MAX_UPLOAD_SIZE: int = os.getenv("MAX_UPLOAD_SIZE", 10485760)

    # BANKING-GRADE: Stricter thresholds for financial security
    DOCUMENT_THRESHOLDS: dict = {
        "passport": 0.25,        # Banking grade - stricter
        "national_id": 0.25,     # Banking grade - stricter  
        "drivers_license": 0.30,  # Banking grade - stricter
        "voter_id": 0.35,        # Banking grade - stricter
        "pan_card": 0.35,        # Banking grade - stricter
        "aadhaar": 0.30,         # Banking grade - stricter
        "unknown": 0.4          # Banking grade - stricter
    }
    
    # BANKING: Security limits
    MAX_CONCURRENT_REQUESTS: int = 3  # Lower for banking security
    MAX_QUEUE_SIZE: int = 10          # Lower for banking security
    DOWNLOAD_TIMEOUT: int = 30        # Shorter timeout
    MAX_IMAGE_SIZE: int = 5242880     # 5MB max for banking

    ENHANCEMENT_PARAMS: dict = {
        "passport": {
            "brightness_boost": 1.3,
            "contrast_boost": 1.4,
            "denoise_strength": 5,
            "sharpen_strength": 1.2,
            "gamma": 1.2
        },
        "default": {
            "brightness_boost": 1.3,
            "contrast_boost": 1.3,
            "denoise_strength": 5,
            "sharpen_strength": 1.1,
            "gamma": 1.2
        }
    }

# Initialize settings and create required directories
settings = Settings()

# SECURITY: Ensure all directories exist with proper permissions
for directory in [settings.MODELS_DIR, settings.TEMP_DIR, settings.UPLOADS_DIR, 
                 settings.STATIC_DIR, settings.LOG_DIR]:
    directory.mkdir(exist_ok=True, mode=0o755)
