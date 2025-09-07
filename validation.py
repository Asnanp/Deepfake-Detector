"""
Comprehensive validation and fallback system for Deepfake Detection API
Provides robust validation, error handling, and graceful degradation
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import hashlib
import tempfile
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors with detailed context"""
    def __init__(self, message: str, code: str = "", details: Dict[str, Any] = {}):
        super().__init__(message)
        self.message = message
        self.code = code or 'VALIDATION_ERROR'
        self.details = details or {}

class MediaValidator:
    """Comprehensive media file validation with fallback strategies"""
    
    # Configuration
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_IMAGE_SIZE = (224, 224)
    MAX_IMAGE_SIZE = (4096, 4096)
    SUPPORTED_FORMATS = {
        'image': {'extensions': ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'],
                 'mimes': ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/bmp']},
        'video': {'extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'],
                 'mimes': ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 
                          'video/webm', 'video/x-flv']}
    }
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.validation_cache = {}
    
    def validate_file(self, file, filename: str) -> Dict[str, Any]:
        """Comprehensive file validation with detailed error reporting"""
        try:
            validation_result = {
                'is_valid': False,
                'file_type': None,
                'file_size': 0,
                'dimensions': None,
                'duration': None,
                'hash': None,
                'warnings': [],
                'errors': []
            }
            
            # Basic file existence check
            if not file or not filename:
                raise ValidationError(
                    "No file provided",
                    code="NO_FILE",
                    details={"filename": filename}
                )
            
            # File size validation
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            validation_result['file_size'] = file_size
            
            if file_size > self.MAX_FILE_SIZE:
                raise ValidationError(
                    f"File size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed ({self.MAX_FILE_SIZE / (1024*1024)}MB)",
                    code="FILE_TOO_LARGE",
                    details={"size": file_size, "max_size": self.MAX_FILE_SIZE}
                )
            
            if file_size == 0:
                raise ValidationError(
                    "File appears to be empty",
                    code="EMPTY_FILE",
                    details={"size": file_size}
                )
            
            # File type detection
            file_type = self._detect_file_type(file, filename)
            validation_result['file_type'] = file_type
            
            # Format-specific validation
            if file_type == 'image':
                self._validate_image(file, validation_result)
            elif file_type == 'video':
                self._validate_video(file, validation_result)
            else:
                raise ValidationError(
                    f"Unsupported file format: {os.path.splitext(filename)[1]}",
                    code="UNSUPPORTED_FORMAT",
                    details={"extension": os.path.splitext(filename)[1]}
                )
            
            # Generate file hash for caching
            file.seek(0)
            file_hash = hashlib.md5(file.read()).hexdigest()
            validation_result['hash'] = file_hash
            file.seek(0)
            
            validation_result['is_valid'] = True
            logger.info(f"File validation successful: {filename} ({file_type}, {file_size} bytes)")
            
            return validation_result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected validation error: {str(e)}")
            raise ValidationError(
                f"Validation failed: {str(e)}",
                code="VALIDATION_FAILED",
                details={"error": str(e)}
            )
    
    def _detect_file_type(self, file, filename: str) -> str:
        """Detect file type using extension and MIME type"""
        ext = os.path.splitext(filename)[1].lower()
        
        # Check by extension
        for file_type, config in self.SUPPORTED_FORMATS.items():
            if ext in config['extensions']:
                return file_type
        
        # Fallback: try to read file header
        file.seek(0)
        header = file.read(32)
        file.seek(0)
        
        # Image signatures
        if header.startswith(b'\xFF\xD8\xFF') or header.startswith(b'\x89PNG') or \
           header.startswith(b'RIFF') or header.startswith(b'GIF'):
            return 'image'
        
        # Video signatures
        if header.startswith(b'ftyp') or header.startswith(b'RIFF') or \
           header.startswith(b'\x00\x00\x00\x20ftyp'):
            return 'video'
        
        return 'unknown'
    
    def _validate_image(self, file, validation_result: Dict[str, Any]):
        """Validate image file with comprehensive checks"""
        try:
            # Try PIL first
            file.seek(0)
            img = Image.open(file)
            
            # Check dimensions
            width, height = img.size
            validation_result['dimensions'] = {'width': width, 'height': height}
            
            if width < self.MIN_IMAGE_SIZE[0] or height < self.MIN_IMAGE_SIZE[1]:
                validation_result['warnings'].append(
                    f"Image resolution ({width}x{height}) is below recommended minimum {self.MIN_IMAGE_SIZE}"
                )
            
            if width > self.MAX_IMAGE_SIZE[0] or height > self.MAX_IMAGE_SIZE[1]:
                validation_result['warnings'].append(
                    f"Image resolution ({width}x{height}) is very high and may impact processing speed"
                )
            
            # Check mode and format
            if img.mode not in ['RGB', 'RGBA', 'L']:
                validation_result['warnings'].append(
                    f"Image mode {img.mode} may be converted to RGB for processing"
                )
            
            # Validate image data integrity
            try:
                img.verify()
            except Exception as e:
                raise ValidationError(
                    f"Image file appears to be corrupted: {str(e)}",
                    code="CORRUPTED_IMAGE",
                    details={"error": str(e)}
                )
            
        except Exception as e:
            # Fallback to OpenCV
            file.seek(0)
            img_array = np.frombuffer(file.read(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValidationError(
                    "Unable to decode image file",
                    code="INVALID_IMAGE",
                    details={"error": "OpenCV decode failed"}
                )
            
            height, width = img.shape[:2]
            validation_result['dimensions'] = {'width': width, 'height': height}
    
    def _validate_video(self, file, validation_result: Dict[str, Any]):
        """Validate video file with comprehensive checks"""
        try:
            # Create temporary file for video processing
            temp_path = os.path.join(self.temp_dir, secure_filename('temp_video'))
            file.seek(0)
            
            with open(temp_path, 'wb') as f:
                f.write(file.read())
            
            # Use OpenCV to validate video
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                raise ValidationError(
                    "Unable to open video file",
                    code="INVALID_VIDEO",
                    details={"error": "OpenCV cannot open file"}
                )
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            validation_result['dimensions'] = {'width': width, 'height': height}
            validation_result['duration'] = duration
            
            # Validation checks
            if duration > 300:  # 5 minutes
                validation_result['warnings'].append(
                    f"Video duration ({duration:.1f}s) is long and may take significant processing time"
                )
            
            if fps < 10 or fps > 60:
                validation_result['warnings'].append(
                    f"Video framerate ({fps} fps) is outside typical range (10-60 fps)"
                )
            
            if width < self.MIN_IMAGE_SIZE[0] or height < self.MIN_IMAGE_SIZE[1]:
                validation_result['warnings'].append(
                    f"Video resolution ({width}x{height}) is below recommended minimum"
                )
            
            cap.release()
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Video validation error: {str(e)}")
            raise ValidationError(
                f"Video validation failed: {str(e)}",
                code="VIDEO_VALIDATION_FAILED",
                details={"error": str(e)}
            )

class FallbackHandler:
    """Handles graceful degradation and fallback strategies"""
    
    @staticmethod
    def create_fallback_response(error: ValidationError, file_info: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Create a safe fallback response when analysis fails"""
        return {
        'prediction': 'unknown',
        'authenticity_score': 0,
        'real_score': 0,
        'fake_score': 0,
        'confidence': 0,
        'model_type': 'unknown',
        'model_name': 'Analysis failed',
        'processing_time': 0,
        'is_real': False,  # ✅ ADD THIS LINE
        'error': {
            'type': error.code,
            'message': str(error),
            'details': error.details
        },
        'file_info': file_info or {},
        'timestamp': datetime.now().isoformat()
    }
    
    @staticmethod
    def validate_and_correct_dimensions(file_path: str, target_size: tuple = (224, 224)) -> str:
        """Auto-correct image dimensions if needed"""
        try:
            img = Image.open(file_path)
            
            # Check if resizing is needed
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save corrected image
                corrected_path = file_path.replace('.', '_corrected.')
                img.save(corrected_path)
                
                return corrected_path
            
            return file_path
            
        except Exception as e:
            logger.warning(f"Could not auto-correct dimensions: {str(e)}")
            return file_path
    
    @staticmethod
    def compress_image_if_needed(file_path: str, max_size: int = 5 * 1024 * 1024) -> str:
        """Compress image if file size is too large"""
        try:
            file_size = os.path.getsize(file_path)
            
            if file_size <= max_size:
                return file_path
            
            img = Image.open(file_path)
            
            # Calculate compression ratio
            compression_ratio = max_size / file_size
            quality = max(10, int(95 * compression_ratio))
            
            # Save compressed image
            compressed_path = file_path.replace('.', '_compressed.')
            img.save(compressed_path, optimize=True, quality=quality)
            
            return compressed_path
            
        except Exception as e:
            logger.warning(f"Could not compress image: {str(e)}")
            return file_path

class SecurityValidator:
    """Security-focused validation for uploaded files"""
    
    @staticmethod
    def scan_for_malicious_content(file_path: str) -> bool:
        """Basic security scan for suspicious patterns"""
        try:
            # Check file extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.exe', '.bat', '.cmd', '.com', '.scr', '.vbs', '.js']:
                return False
            
            # Check file magic numbers
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
            # Common malicious signatures
            malicious_signatures = [
                b'MZ',  # DOS executable
                b'\x7fELF',  # Linux executable
                b'\xfe\xed\xfa\xce',  # Mach-O executable
            ]
            
            for signature in malicious_signatures:
                if header.startswith(signature):
                    return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename for security"""
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = ['<', '>', ':', '"', '|', '?', '*']
        for pattern in suspicious_patterns:
            if pattern in filename:
                return False
        
        return True

class CacheManager:
    """Caching system for validation results"""
    
    def __init__(self, cache_dir: str = ""):
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        self.cache_file = os.path.join(self.cache_dir, 'validation_cache.json')
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load validation cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
    
    def _save_cache(self):
        """Save validation cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache: {str(e)}")
    
    def get_cached_result(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached validation result"""
        return self.cache.get(file_hash)
    
    def cache_result(self, file_hash: str, result: Dict[str, Any]):
        """Cache validation result"""
        self.cache[file_hash] = {
            **result,
            'cached_at': datetime.now().isoformat()
        }
        self._save_cache()
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries"""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            self.cache = {
                k: v for k, v in self.cache.items()
                if datetime.fromisoformat(v['cached_at']).timestamp() > cutoff_time
            }
            self._save_cache()
        except Exception as e:
            logger.warning(f"Could not cleanup cache: {str(e)}")

# Global validator instances
media_validator = MediaValidator()
fallback_handler = FallbackHandler()
security_validator = SecurityValidator()
cache_manager = CacheManager()

# Convenience functions for API usage
def validate_and_prepare_file(file, filename: str) -> Tuple[Dict[str, Any], bool]:
    """High-level function to validate and prepare a file for processing"""
    try:
        validation_result = media_validator.validate_file(file, filename)
        
        # Check cache
        cached_result = cache_manager.get_cached_result(validation_result['hash'])
        if cached_result:
            logger.info(f"Using cached validation result for {filename}")
            return cached_result, True
        
        # Cache the result
        cache_manager.cache_result(validation_result['hash'], validation_result)
        
        return validation_result, False
        
    except ValidationError as e:
        logger.error(f"Validation failed for {filename}: {str(e)}")
        return fallback_handler.create_fallback_response(e), False
    except Exception as e:
        logger.error(f"Unexpected error validating {filename}: {str(e)}")
        error = ValidationError(str(e), code="UNEXPECTED_ERROR")
        return fallback_handler.create_fallback_response(error), False