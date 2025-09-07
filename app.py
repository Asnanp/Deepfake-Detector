import os
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model import DeepfakeDetector
from validation import validate_and_prepare_file, ValidationError, fallback_handler, security_validator

# Utilities: JSON sanitization to ensure Flask can serialize responses
def _make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable types.
    - Handles numpy types (bool_, integer, floating, ndarray)
    - Handles datetime objects
    - Converts sets/tuples to lists
    - Falls back to str(obj) for unknown types
    """
    try:
        import numpy as np  # Optional; only if available
    except Exception:  # pragma: no cover
        np = None

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(_make_json_serializable(k)): _make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [ _make_json_serializable(v) for v in obj ]

    # Numpy handling
    if np is not None:
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

    # Fallback: stringify unknown objects
    return str(obj)

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request tracking
request_stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'average_processing_time': 0
}

# Initialize the deepfake detector with comprehensive error handling
class DetectorManager:
    def __init__(self):
        self.detector = None
        self.model_status = 'uninitialized'
        self.init_detector()
    
    def init_detector(self):
        """Initialize detector with fallback strategies"""
        try:
            logger.info("Initializing deepfake detector...")
            self.detector = DeepfakeDetector()
            
            if self.detector.model_type == 'huggingface':
                self.model_status = 'huggingface_loaded'
                logger.info("Deepfake detector initialized successfully with Hugging Face model")
            elif self.detector.model_type == 'tensorflow':
                self.model_status = 'tensorflow_loaded'
                logger.info("Deepfake detector initialized with TensorFlow model")
            else:
                self.model_status = 'demo_mode'
                logger.warning("Deepfake detector running in demo mode")
                
        except Exception as e:
            self.model_status = 'error'
            logger.error(f"Error initializing deepfake detector: {str(e)}")
            # Continue with demo mode as fallback
            self.detector = DeepfakeDetector()
            self.model_status = 'demo_mode'
            logger.info("Falling back to demo mode")
    
    def get_status(self):
        return {
            'status': self.model_status,
            'is_ready': self.detector is not None
        }
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze file with error handling and fallback"""
        if not self.detector:
            raise Exception("Detector not initialized")
        
        try:
            return self.detector.analyze_media(file_path)
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            # Return fallback response
            return fallback_handler.create_fallback_response(
                ValidationError(str(e), code="ANALYSIS_FAILED"),
                file_info={'file_path': file_path}
            )

# Initialize detector manager
detector_manager = DetectorManager()

# Status endpoint
@app.route('/api/status')
def status():
    """Status endpoint for monitoring"""
    status = detector_manager.get_status()
    return jsonify({
        'status': 'healthy' if status['is_ready'] else 'unhealthy',
        'model_status': status['status'],
        'uptime': request_stats['total_requests'],
        'stats': request_stats
    })

# Health check endpoint (alias)
@app.route('/api/health')
def health_check():
    return status()

# Routes
@app.route('/')
def index():
    return send_from_directory(app.static_folder or '', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder or '', path)

@app.route('/api/detect', methods=['POST', 'OPTIONS'])
def detect_deepfake():
    """Enhanced deepfake detection with comprehensive validation and error handling"""
    if request.method == 'OPTIONS':
        response = jsonify({
            'status': 'ok',
            'methods': ['POST'],
            'allowed_types': list(ALLOWED_EXTENSIONS)
        })
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    start_time = time.time()
    request_stats['total_requests'] += 1
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            request_stats['failed_requests'] += 1
            return jsonify({
                'error': 'No file provided',
                'error_code': 'NO_FILE',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            request_stats['failed_requests'] += 1
            return jsonify({
                'error': 'No file selected',
                'error_code': 'EMPTY_FILENAME',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Validate filename security
        if not security_validator.validate_filename(file.filename or ''):
            request_stats['failed_requests'] += 1
            return jsonify({
                'error': 'Invalid filename',
                'error_code': 'INVALID_FILENAME',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Comprehensive file validation
        validation_result, is_cached = validate_and_prepare_file(file, file.filename or '')
        
        if not validation_result.get('is_valid', False):
            request_stats['failed_requests'] += 1
            return jsonify({
                'error': 'File validation failed',
                'error_code': 'VALIDATION_FAILED',
                'details': validation_result.get('errors', []),
                'warnings': validation_result.get('warnings', []),
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Save validated file
        filename = secure_filename(file.filename or '')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Security scan
        if not security_validator.scan_for_malicious_content(file_path):
            os.remove(file_path)
            request_stats['failed_requests'] += 1
            return jsonify({
                'error': 'Security scan failed',
                'error_code': 'SECURITY_FAILED',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Log file info
        logger.info(f"Processing file: {filename} ({validation_result['file_type']}, "
                   f"{validation_result['file_size']} bytes)")
        
        # Analyze file
        result = detector_manager.analyze(file_path)
        
        # Ensure all values are JSON-serializable (handles numpy types, etc.)
        result = _make_json_serializable(result)
        # Guarantee we have a dict to attach metadata to
        if not isinstance(result, dict):
            result = {'result': result}
        
        # Add processing metadata
        processing_time = time.time() - start_time
        result.update({
            'processing_time': round(processing_time, 2),
            'file_info': {
                'filename': filename,
                'file_type': validation_result['file_type'],
                'file_size': validation_result['file_size'],
                'dimensions': validation_result.get('dimensions'),
                'duration': validation_result.get('duration')
            },
            'warnings': validation_result.get('warnings', []),
            'timestamp': datetime.now().isoformat()
        })
        
        # Update stats
        request_stats['successful_requests'] += 1
        request_stats['average_processing_time'] = int(
            (request_stats['average_processing_time'] * (request_stats['successful_requests'] - 1) + processing_time) 
            / request_stats['successful_requests']
        )
        
        # Cleanup
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(_make_json_serializable(result)), 200
        
    except ValidationError as e:
        request_stats['failed_requests'] += 1
        logger.error(f"Validation error: {str(e)}")
        return jsonify(_make_json_serializable(fallback_handler.create_fallback_response(e))), 400
        
    except Exception as e:
        request_stats['failed_requests'] += 1
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'error_code': 'INTERNAL_ERROR',
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)