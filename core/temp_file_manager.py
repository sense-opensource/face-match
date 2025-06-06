import tempfile
import os
from contextlib import contextmanager
from config import settings
import logging

logger = logging.getLogger(__name__)

class TempFileManager:
    def __init__(self):
        self.temp_files = []
    
    def create_temp_file(self, suffix: str = '.jpg') -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.TEMP_DIR)
        temp_path = temp_file.name
        temp_file.close()
        self.temp_files.append(temp_path)
        return temp_path
    
    def cleanup(self):
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")
        self.temp_files.clear()

@contextmanager
def temp_file_context():
    manager = TempFileManager()
    try:
        yield manager
    finally:
        manager.cleanup()