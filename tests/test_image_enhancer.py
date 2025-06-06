import pytest
from core.image_enhancer import AdvancedImageEnhancer

def test_image_enhancer_initialization():
    enhancer = AdvancedImageEnhancer()
    assert enhancer.models_dir is not None
    assert isinstance(enhancer.enhancement_params, dict)