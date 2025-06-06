import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union, Tuple
from config import settings

logger = logging.getLogger(__name__)

class AdvancedImageEnhancer:
    """
    Advanced image enhancement class for face verification
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize the image enhancer
        
        Args:
            models_dir: Directory containing models for enhancement
        """
        self.models_dir = models_dir
        
        # Enhancement parameters by document type
        self.enhancement_params = {
            "passport": {
                "brightness_boost": 1.3,
                "contrast_boost": 1.4,
                "denoise_strength": 5,
                "sharpen_strength": 1.2,
                "gamma": 1.2
            },
            "drivers_license": {
                "brightness_boost": 1.4, 
                "contrast_boost": 1.5,
                "denoise_strength": 7,
                "sharpen_strength": 1.3,
                "gamma": 1.3
            },
            "id_card": {
                "brightness_boost": 1.3,
                "contrast_boost": 1.4,
                "denoise_strength": 6,
                "sharpen_strength": 1.2,
                "gamma": 1.2
            },
            "selfie": {
                "brightness_boost": 1.5,
                "contrast_boost": 1.4,
                "denoise_strength": 8,
                "sharpen_strength": 1.0,
                "gamma": 1.4
            },
            "low_light": {
                "brightness_boost": 2.0,
                "contrast_boost": 1.8,
                "denoise_strength": 10,
                "sharpen_strength": 1.1,
                "gamma": 1.8
            },
            "default": {
                "brightness_boost": 1.3,
                "contrast_boost": 1.3,
                "denoise_strength": 5,
                "sharpen_strength": 1.1,
                "gamma": 1.2
            }
        }
    
    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image quality and determine enhancement needs
        
        Args:
            image: Input image
            
        Returns:
            Quality metrics and enhancement type needed
        """
        if image is None or image.size == 0:
            return {"error": "Invalid image", "type": "default"}
        
        # Convert to grayscale for analysis
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate quality metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Compute noise estimate
        noise_level = self._estimate_noise(gray)
        
        # Determine if it's a document image
        is_document = self._is_document_image(image)
        
        # Determine enhancement type
        enhancement_type = "default"
        
        if is_document:
            if "license" in is_document or "driving" in is_document:
                enhancement_type = "drivers_license"
            elif "passport" in is_document:
                enhancement_type = "passport"
            else:
                enhancement_type = "id_card"
        else:
            # For non-document (likely selfie)
            enhancement_type = "selfie"
            
            # Check if it's a low light image
            if brightness < 50:
                enhancement_type = "low_light"
        
        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "blur_score": float(blur_score),
            "noise_level": float(noise_level),
            "is_document": is_document,
            "enhancement_type": enhancement_type
        }
    
    def _estimate_noise(self, gray_img: np.ndarray) -> float:
        """
        Estimate noise level in an image
        
        Args:
            gray_img: Grayscale image
            
        Returns:
            Estimated noise level
        """
        # Apply median filter (noise removal)
        median_filtered = cv2.medianBlur(gray_img, 3)
        
        # Compute difference between original and filtered
        diff = cv2.absdiff(gray_img, median_filtered)
        
        # Noise estimation
        noise_level = np.mean(diff)
        
        return float(noise_level)
    
    def _is_document_image(self, image: np.ndarray) -> Union[str, bool]:
        """
        Check if image is likely a document
        
        Args:
            image: Input image
            
        Returns:
            Document type if detected, False otherwise
        """
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Most ID documents have specific aspect ratios
        if 1.4 < aspect_ratio < 1.6:
            # Common ratio for ID cards/driver's licenses
            return "id_card"
        elif 1.3 < aspect_ratio < 1.5:
            # Driver's license often has this ratio
            return "drivers_license"
        elif 0.68 < aspect_ratio < 0.72:
            # Passport usually has this ratio
            return "passport"
            
        # Do basic text detection to confirm it's a document
        # Simple edge-based text detection
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply edge detection and check for text-like patterns
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Documents typically have many small rectangular contours (text)
        small_rect_count = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h > 0 else 0
            if 10 < w < 100 and 5 < h < 50 and 0.5 < aspect < 5:
                small_rect_count += 1
        
        if small_rect_count > 20:
            return "document"
            
        return False
    
    def enhance_image(self, image_path: str, enhancement_type: str = None) -> np.ndarray:
        """
        Enhance image using multiple algorithms based on the image type
        
        Args:
            image_path: Path to input image
            enhancement_type: Type of enhancement to apply (auto-detected if None)
            
        Returns:
            Enhanced image
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            # Analyze image if enhancement type not specified
            if enhancement_type is None:
                analysis = self.analyze_image_quality(image)
                enhancement_type = analysis["enhancement_type"]
                
            # Get enhancement parameters
            params = self.enhancement_params.get(enhancement_type, self.enhancement_params["default"])
            
            # Apply adaptive enhancement based on parameters
            enhanced = self._apply_enhancement_pipeline(image, params)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Enhancement error: {str(e)}")
            return None
    
    def _apply_enhancement_pipeline(self, image: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Apply a comprehensive enhancement pipeline
        
        Args:
            image: Input image
            params: Enhancement parameters
            
        Returns:
            Enhanced image
        """
        if image is None or image.size == 0:
            return None
            
        # 1. Denoise
        denoised = self._apply_advanced_denoising(image, params["denoise_strength"])
        
        # 2. Apply CLAHE for better contrast
        enhanced = self._apply_adaptive_clahe(denoised)
        
        # 3. Adjust brightness and contrast
        enhanced = self._adjust_brightness_contrast(
            enhanced, 
            params["brightness_boost"], 
            params["contrast_boost"]
        )
        
        # 4. Apply gamma correction
        enhanced = self._apply_gamma_correction(enhanced, params["gamma"])
        
        # 5. Sharpen
        enhanced = self._apply_adaptive_sharpening(enhanced, params["sharpen_strength"])
        
        # 6. Color correction (for color images)
        if len(enhanced.shape) > 2:
            enhanced = self._apply_color_correction(enhanced)
        
        return enhanced
    
    def _apply_advanced_denoising(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply advanced denoising based on image characteristics
        
        Args:
            image: Input image
            strength: Denoising strength
            
        Returns:
            Denoised image
        """
        try:
            # Analyze the image noise
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            noise_level = self._estimate_noise(gray)
            
            # Adjust denoising strength based on noise level
            adjusted_strength = min(10, max(3, int(noise_level * strength / 2)))
            
            # Apply appropriate denoising method
            if len(image.shape) > 2:
                # Color image: Non-Local Means denoising
                h_luminance = adjusted_strength
                h_color = adjusted_strength
                template_window_size = 7
                search_window_size = 21
                
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, 
                    None, 
                    h_luminance, 
                    h_color, 
                    template_window_size, 
                    search_window_size
                )
            else:
                # Grayscale image: Non-Local Means denoising for grayscale
                h = adjusted_strength
                template_window_size = 7
                search_window_size = 21
                
                denoised = cv2.fastNlMeansDenoising(
                    image, 
                    None, 
                    h, 
                    template_window_size, 
                    search_window_size
                )
                
            return denoised
            
        except Exception as e:
            logger.error(f"Denoising error: {str(e)}")
            return image
    
    def _apply_adaptive_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space for CLAHE
            if len(image.shape) > 2:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                
                # Merge channels
                enhanced_lab = cv2.merge((cl, a, b))
                
                # Convert back to BGR
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
                
            return enhanced
            
        except Exception as e:
            logger.error(f"CLAHE error: {str(e)}")
            return image
    
    def _adjust_brightness_contrast(self, image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
        """
        Adjust brightness and contrast adaptively
        
        Args:
            image: Input image
            brightness: Brightness boost factor
            contrast: Contrast boost factor
            
        Returns:
            Adjusted image
        """
        try:
            # Convert to float32 for processing
            adjusted = image.astype(np.float32) / 255.0
            
            # Apply brightness adjustment
            adjusted = adjusted * brightness
            
            # Apply contrast adjustment (around mean)
            mean = np.mean(adjusted)
            adjusted = (adjusted - mean) * contrast + mean
            
            # Clip values to [0, 1] range
            adjusted = np.clip(adjusted, 0, 1)
            
            # Convert back to uint8
            adjusted = (adjusted * 255).astype(np.uint8)
            
            return adjusted
                
        except Exception as e:
            logger.error(f"Brightness/contrast adjustment error: {str(e)}")
            return image
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction
        
        Args:
            image: Input image
            gamma: Gamma value
            
        Returns:
            Gamma-corrected image
        """
        try:
            # Build lookup table for gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
            ]).astype(np.uint8)
            
            # Apply gamma correction
            if len(image.shape) > 2:
                # Apply to each channel
                b, g, r = cv2.split(image)
                b = cv2.LUT(b, table)
                g = cv2.LUT(g, table)
                r = cv2.LUT(r, table)
                corrected = cv2.merge([b, g, r])
            else:
                # Apply to grayscale image
                corrected = cv2.LUT(image, table)
                
            return corrected
            
        except Exception as e:
            logger.error(f"Gamma correction error: {str(e)}")
            return image
    
    def _apply_adaptive_sharpening(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply adaptive sharpening based on image characteristics
        
        Args:
            image: Input image
            strength: Sharpening strength
            
        Returns:
            Sharpened image
        """
        try:
            # Check if image is blurry to adapt sharpening strength
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Adjust sharpening strength based on blur detection
            # More aggressive for blurry images
            if blur_score < 100:
                adjusted_strength = min(2.0, strength * 1.5)
            else:
                adjusted_strength = strength
            
            # Method 1: Unsharp Masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 3)
            sharpened = cv2.addWeighted(
                image, 1 + adjusted_strength, 
                gaussian, -adjusted_strength, 
                0
            )
            
            # Method 2: For stronger sharpening, use kernel-based approach
            if adjusted_strength > 1.3:
                kernel = np.array([
                    [-1, -1, -1],
                    [-1, 9 + adjusted_strength, -1],
                    [-1, -1, -1]
                ]) / (9 + adjusted_strength - 8)
                
                kernel_sharpened = cv2.filter2D(image, -1, kernel)
                
                # Blend the two methods
                alpha = min(1.0, max(0.0, (adjusted_strength - 1.0) / 0.5))
                sharpened = cv2.addWeighted(
                    sharpened, 1 - alpha,
                    kernel_sharpened, alpha,
                    0
                )
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Sharpening error: {str(e)}")
            return image
    
    def _apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color correction to improve color balance
        
        Args:
            image: Input color image
            
        Returns:
            Color-corrected image
        """
        try:
            # Only process color images
            if len(image.shape) <= 2:
                return image
                
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Compute histogram of a and b channels
            hist_a = cv2.calcHist([a], [0], None, [256], [0, 256])
            hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
            
            # Find peaks
            a_peak = np.argmax(hist_a)
            b_peak = np.argmax(hist_b)
            
            # Calculate shifts to center the histograms (neutral color)
            a_shift = 128 - a_peak
            b_shift = 128 - b_peak
            
            # Apply shifts moderately (50% of the calculated shift for subtle correction)
            a = np.clip(a.astype(np.int32) + int(a_shift * 0.5), 0, 255).astype(np.uint8)
            b = np.clip(b.astype(np.int32) + int(b_shift * 0.5), 0, 255).astype(np.uint8)
            
            # Merge channels
            corrected_lab = cv2.merge((l, a, b))
            
            # Convert back to BGR
            corrected = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
            
            return corrected
            
        except Exception as e:
            logger.error(f"Color correction error: {str(e)}")
            return image
    
    def enhance_for_face_verification(self, image_path: str, is_id_document: bool = False) -> str:
        """
        Enhanced specifically for face verification tasks
        
        Args:
            image_path: Path to input image
            is_id_document: Whether the image is an ID document
            
        Returns:
            Path to enhanced image (temporary file)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return image_path
            
            # Analyze image quality
            analysis = self.analyze_image_quality(image)
            
            # Override enhancement type if specified
            if is_id_document and "document" not in analysis["enhancement_type"]:
                enhancement_type = "id_card"
            else:
                enhancement_type = analysis["enhancement_type"]
            
            # Special case for extremely dark images (like the example provided)
            if analysis["brightness"] < 50:
                logger.info(f"Low light image detected, applying specialized enhancement")
                enhancement_type = "low_light"
                
                # For very dark images, apply extreme enhancement
                if analysis["brightness"] < 30:
                    # Create custom params for extremely dark images
                    custom_params = self.enhancement_params["low_light"].copy()
                    custom_params["brightness_boost"] = 2.5
                    custom_params["contrast_boost"] = 2.0
                    custom_params["gamma"] = 2.2
                    
                    # Apply special enhancement pipeline
                    enhanced = self._apply_enhancement_pipeline(image, custom_params)
                    
                    # If still too dark, apply histogram equalization
                    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                    if np.mean(enhanced_gray) < 70:
                        # Apply additional processing for extremely dark images
                        enhanced = self._enhance_very_dark_image(enhanced)
                else:
                    # Regular low light enhancement
                    enhanced = self._apply_enhancement_pipeline(
                        image, 
                        self.enhancement_params["low_light"]
                    )
            else:
                # Normal enhancement
                enhanced = self._apply_enhancement_pipeline(
                    image, 
                    self.enhancement_params[enhancement_type]
                )
            
            # Save enhanced image to temporary file
            output_path = f"{image_path}_enhanced.jpg"
            cv2.imwrite(output_path, enhanced)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Face verification enhancement error: {str(e)}")
            return image_path
    
    def _enhance_very_dark_image(self, image: np.ndarray) -> np.ndarray:
        """
        Special enhancement for very dark images
        
        Args:
            image: Input dark image
            
        Returns:
            Heavily enhanced image
        """
        try:
            # Convert to HSV for better brightness adjustment
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Apply strong histogram equalization to V channel
            v_eq = cv2.equalizeHist(v)
            
            # Apply CLAHE again for better local contrast
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            v_eq = clahe.apply(v_eq)
            
            # Merge channels
            hsv_eq = cv2.merge([h, s, v_eq])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
            
            # Apply additional gamma correction
            enhanced = self._apply_gamma_correction(enhanced, 2.2)
            
            # Denoise again as these operations can introduce noise
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Very dark image enhancement error: {str(e)}")
            return image
    
    def _locate_face_in_id(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate face region in ID document
        
        Args:
            image: ID document image
            
        Returns:
            Face region as (x, y, w, h) or None if not found
        """
        try:
            # Try to use a face detector first
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Return the largest face
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                return tuple(faces[0])
            
            # If face detection fails, try to use heuristics for ID cards
            h, w = image.shape[:2]
            
            # Common locations for face photos in IDs
            regions = [
                # Top right quadrant (common in many IDs)
                (w // 2, 0, w // 2, h // 2),
                # Top left quadrant
                (0, 0, w // 2, h // 2),
                # Right third (vertical IDs)
                (2 * w // 3, 0, w // 3, h // 2)
            ]
            
            # Check skin tone presence in these regions to identify the most likely face region
            max_skin_ratio = 0
            best_region = None
            
            for region in regions:
                x, y, rw, rh = region
                roi = image[y:y+rh, x:x+rw]
                
                # Calculate skin tone ratio
                skin_ratio = self._calculate_skin_ratio(roi)
                
                if skin_ratio > max_skin_ratio:
                    max_skin_ratio = skin_ratio
                    best_region = region
            
            # If a likely region is found
            if max_skin_ratio > 0.15:  # At least 15% skin tones
                return best_region
            
            return None
            
        except Exception as e:
            logger.error(f"Face location error: {str(e)}")
            return None
    
    def _calculate_skin_ratio(self, image: np.ndarray) -> float:
        """
        Calculate ratio of skin tone pixels in image
        
        Args:
            image: Input image
            
        Returns:
            Ratio of skin tone pixels
        """
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin tone range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            
            # Create skin mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate ratio
            skin_pixels = cv2.countNonZero(mask)
            total_pixels = image.shape[0] * image.shape[1]
            
            return skin_pixels / total_pixels
            
        except Exception as e:
            logger.error(f"Skin ratio calculation error: {str(e)}")
            return 0.0
