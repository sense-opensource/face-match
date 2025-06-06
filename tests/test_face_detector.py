import pytest
from core.face_detector import MultiFaceDetector

def test_face_detector_initialization():
    detector = MultiFaceDetector()
    assert detector.detector_type in ["retinaface", "mtcnn", "dlib", "dnn", "haarcascade"]