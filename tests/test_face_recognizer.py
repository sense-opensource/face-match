import pytest
from core.face_recognizer import FaceRecognizer

def test_face_recognizer_initialization():
    recognizer = FaceRecognizer()
    assert recognizer.recognizer_type in ["arcface", "facenet", "hog", "none"]