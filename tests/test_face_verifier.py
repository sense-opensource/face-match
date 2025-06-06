import pytest
from core.face_verifier import SenseFaceVerifier

def test_face_verifier_initialization():
    verifier = SenseFaceVerifier()
    assert verifier.face_detector is not None
    assert verifier.face_recognizer is not None
    assert verifier.image_enhancer is not None