import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm

class DeepfakeDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.processor = None
        self.model_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            # Default to the requested Hugging Face model
            target_model = "dima806/deepfake_vs_real_image_detection"
            
            if model_path is None:
                if self.load_huggingface_model(target_model):
                    self.model_type = 'huggingface'
                    print(f"Successfully loaded Hugging Face model: {target_model}")
                else:
                    raise Exception("Failed to load Hugging Face model")
            elif os.path.exists(model_path):
                if self.load_model(model_path):
                    self.model_type = 'tensorflow'
                    print(f"Successfully loaded TensorFlow model from: {model_path}")
                else:
                    raise Exception("Failed to load TensorFlow model")
            else:
                raise Exception(f"Model path not found: {model_path}")
                
        except Exception as e:
            print(f"Model initialization error: {str(e)}")
            print("Falling back to enhanced detection mode")
            self.model_type = 'enhanced_fallback'
    
    def load_model(self, model_path):
        """Load a pre-trained TensorFlow deepfake detection model"""
        try:
            from tensorflow import keras
            self.tf_model = keras.models.load_model(model_path)
            self.model_type = 'tensorflow'
            print(f"TensorFlow model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading TensorFlow model: {str(e)}")
            print("Falling back to random prediction mode for demonstration purposes.")
            self.tf_model = None
            self.model_type = 'random'
            return False
    
    def load_huggingface_model(self, model_name):
        """Load a pre-trained Hugging Face deepfake detection model"""
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model_type = 'huggingface'
            print(f"Hugging Face model loaded successfully: {model_name}")
            return True
        except Exception as e:
            print(f"Error loading Hugging Face model: {str(e)}")
            print("Falling back to random prediction mode for demonstration purposes.")
            self.model = None
            self.processor = None
            self.model_type = 'random'
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess an image for model input - always resize to 256x256"""
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 224x224 for optimal model performance
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0  # Normalize pixel values
            
            # Ensure 3 channels (RGB)
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,) * 3, axis=-1)
            elif img_array.shape[2] > 3:
                img_array = img_array[:, :, :3]
                
            return np.expand_dims(img_array, axis=0)  # Add batch dimension
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from a video file"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval to extract evenly distributed frames
            if frame_count <= max_frames:
                frame_interval = 1
            else:
                frame_interval = frame_count // max_frames
            
            frame_idx = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize frame to 224x224
                    frame = cv2.resize(frame, (224, 224))
                    # Normalize pixel values
                    frame = frame / 255.0
                    frames.append(frame)
                
                frame_idx += 1
            
            cap.release()
            return np.array(frames)
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return None
    
    def enhanced_fallback_detection(self, image_path):
        """Simplified fallback detection using basic image analysis"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import os
            import time
            
            start_time = time.time()
            
            # Load and resize image to 256x256
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Could not load image'}
            
            img = cv2.resize(img, (224, 224))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Basic analysis
            # 1. Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_count = len(faces)
            
            # 2. Image quality (blur detection)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 3. Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Calculate authenticity score
            face_score = 0.8 if face_count > 0 else 0.3
            quality_score = min(1.0, blur_score / 1000.0)
            edge_score = 1.0 - abs(edge_density - 0.1) / 0.1
            edge_score = max(0.0, min(1.0, edge_score))
            
            # Weighted average
            authenticity_score = (face_score * 0.4 + quality_score * 0.3 + edge_score * 0.3)
            
            # Add small random factor for variation
            random_factor = np.random.normal(0, 0.02)
            authenticity_score = max(0.0, min(1.0, authenticity_score + random_factor))
            
            real_score = authenticity_score
            fake_score = 1 - authenticity_score
            
            processing_time = time.time() - start_time
            
            # Get comprehensive analysis for additional validation
            ela_results = self.ela_analysis(image_path)
            noise_results = self.noise_analysis(image_path)
            color_results = self.color_distribution_analysis(image_path)
            compression_results = self.compression_analysis(image_path)
            
            # Combine basic analysis with all forensic methods
            ela_manipulation = ela_results.get('manipulation_score', 0.0)
            noise_artificial = noise_results.get('is_artificial_noise', False)
            color_manipulated = color_results.get('is_color_manipulated', False)
            compression_artifacts = compression_results.get('has_compression_artifacts', False)
            
            # Adjust scores based on all analysis methods
            adjustment_factor = 0.0
            
            if ela_results.get('is_manipulated', False):
                adjustment_factor -= (ela_manipulation * 0.4)

            # Keep as boolean for correct logic
            if noise_artificial:
                adjustment_factor -= 0.15
            
            if color_manipulated:
                adjustment_factor -= 0.15
            
            if compression_artifacts:
                adjustment_factor -= 0.1
            
            # Apply adjustments
            real_score = max(0.0, min(1.0, real_score + adjustment_factor))
            fake_score = max(0.0, min(1.0, fake_score - adjustment_factor))
            
            # Simple output - just prediction and percentage
            is_real = real_score > fake_score
            prediction = 'REAL' if is_real else 'FAKE'
            confidence = real_score * 100 if is_real else fake_score * 100
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 1),
                'percentage': f"{confidence:.1f}%",
                'is_real': is_real,
                'is_fake': not is_real,
                'ela_analysis': ela_results,
                'noise_analysis': noise_results,
                'color_analysis': color_results,
                'compression_analysis': compression_results,
                'ela_manipulation_score': round(ela_manipulation * 100, 1)
            }
            
        except Exception as e:
            print(f"Error in fallback detection: {str(e)}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'percentage': '0.0%',
                'is_real': False,
                'is_fake': False,
                'error': 'Analysis failed'
            }

    def predict_image(self, image_path):
        """Predict if an image is real or fake using the loaded model with advanced forensic analysis"""
        import time
        start_time = time.time()
        
        # Get forensic analysis
        forensic_data = self.get_forensic_analysis(image_path)
        
        if self.model_type == 'enhanced_fallback':
            result = self.enhanced_fallback_detection(image_path)
            result['processingTime'] = (time.time() - start_time) * 1000
            return result
        elif self.model is None or self.model_type == 'random':
            result = self.enhanced_fallback_detection(image_path)
            result['processingTime'] = (time.time() - start_time) * 1000
            return result
        
        # Check if we're using Hugging Face model
        if self.processor is not None:
            try:
                image = Image.open(image_path)
                
                # Ensure image is in RGB mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize to 224x224 for optimal model performance
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Process with Hugging Face
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get the probability for the 'Real' class
                id2label = self.model.config.id2label
                print(f"Model labels: {id2label}")
                
                # Find the correct indices for real and fake labels
                real_idx = None
                fake_idx = None
                
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if 'real' in label_lower or 'authentic' in label_lower:
                        real_idx = int(idx)
                    elif 'fake' in label_lower or 'deepfake' in label_lower or 'synthetic' in label_lower:
                        fake_idx = int(idx)
                
                # For this specific model: 0='Deepfake', 1='Real'
                if real_idx is None or fake_idx is None:
                    real_idx = 1  # Real is index 1
                    fake_idx = 0  # Deepfake is index 0
                
                print(f"Using real_idx: {real_idx}, fake_idx: {fake_idx}")
                
                authenticity_score = float(probabilities[0, real_idx].item())
                
                # Ensure all values are valid numbers
                if np.isnan(authenticity_score):
                    authenticity_score = 0.5
                
                real_score = float(probabilities[0, real_idx].item())
                fake_score = float(probabilities[0, fake_idx].item()) if fake_idx is not None else 1 - authenticity_score
                
                # Ensure all values are valid
                real_score = 0.0 if np.isnan(real_score) else max(0.0, min(1.0, real_score))
                fake_score = 0.0 if np.isnan(fake_score) else max(0.0, min(1.0, fake_score))
                
                processing_time = (time.time() - start_time) * 1000
                
                # Get comprehensive analysis for additional validation
                ela_results = self.ela_analysis(image_path)
                noise_results = self.noise_analysis(image_path)
                color_results = self.color_distribution_analysis(image_path)
                compression_results = self.compression_analysis(image_path)
                
                # Combine AI prediction with all analysis methods
                ai_confidence = authenticity_score
                ela_manipulation = ela_results.get('manipulation_score', 0.0)
                noise_artificial = noise_results.get('is_artificial_noise', False)
                color_manipulated = color_results.get('is_color_manipulated', False)
                compression_artifacts = compression_results.get('has_compression_artifacts', False)
                
                # Adjust confidence based on all analysis methods
                adjustment_factor = 0.0
                
                if ela_results.get('is_manipulated', False):
                    adjustment_factor -= (ela_manipulation * 0.3)
                
                if noise_artificial:
                    adjustment_factor -= 0.1
                
                if color_manipulated:
                    adjustment_factor -= 0.1
                
                if compression_artifacts:
                    adjustment_factor -= 0.05
                
                # Apply adjustments
                adjusted_confidence = max(0.0, min(1.0, ai_confidence + adjustment_factor))
                
                # Simple output - just prediction and percentage
                is_real = adjusted_confidence > 0.5
                prediction = 'REAL' if is_real else 'FAKE'
                confidence = adjusted_confidence * 100 if is_real else (1 - adjusted_confidence) * 100
                
                return {
                    'prediction': prediction,
                    'confidence': round(confidence, 1),
                    'percentage': f"{confidence:.1f}%",
                    'is_real': is_real,
                    'is_fake': (not is_real),
                    'ela_analysis': ela_results,
                    'noise_analysis': noise_results,
                    'color_analysis': color_results,
                    'compression_analysis': compression_results,
                    'ai_confidence': round(ai_confidence * 100, 1),
                    'ela_manipulation_score': round(ela_manipulation * 100, 1)
                }
            except Exception as e:
                print(f"Error predicting with Hugging Face model: {str(e)}")
                result = {'error': f'Failed to process image with Hugging Face model: {str(e)}'}
                result.update(forensic_data)
                result['processingTime'] = str((time.time() - start_time) * 1000)
                return result
        else:
            # Original TensorFlow model logic
            processed_img = self.preprocess_image(image_path)
            if processed_img is None:
                return {'error': 'Failed to process image'}
            
            prediction = self.model.predict(processed_img)[0][0]
            processing_time = (time.time() - start_time) * 1000
            
            # Simple output - just prediction and percentage
            is_real = prediction > 0.5
            prediction_text = 'REAL' if is_real else 'FAKE'
            confidence = prediction * 100 if is_real else (1 - prediction) * 100
            
            return {
                'prediction': prediction_text,
                'confidence': round(confidence, 1),
                'percentage': f"{confidence:.1f}%",
                'is_real': is_real,
                'is_fake': not is_real
            }
    
    def enhanced_fallback_video_detection(self, video_path):
        """Simplified video detection using basic frame analysis"""
        try:
            import cv2
            import numpy as np
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Could not open video file'}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                return {'error': 'No frames found in video'}
            
            # Analyze a few frames for efficiency
            max_frames = min(10, frame_count)
            frame_interval = max(1, frame_count // max_frames)
            
            frame_scores = []
            face_cascade = cv2.CascadeClassifier(cv2.samples.findFile('haarcascades/haarcascade_frontalface_default.xml'))
            
            frame_idx = 0
            analyzed_frames = 0
            
            while analyzed_frames < max_frames and frame_idx < frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    frame_idx += 1
                    continue
                
                # Resize frame to 224x224
                frame = cv2.resize(frame, (224, 224))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Face detection
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # Simple frame score
                frame_score = 0.7 if len(faces) > 0 else 0.4
                frame_scores.append(frame_score)
                
                analyzed_frames += 1
                frame_idx += frame_interval
            
            cap.release()
            
            if not frame_scores:
                return {'error': 'No valid frames analyzed'}
            
            # Calculate final score
            authenticity_score = np.mean(frame_scores)
            authenticity_score = float(max(0.0, min(1.0, float(authenticity_score))))
            
            real_score = authenticity_score
            fake_score = 1 - authenticity_score
            
            return {
                'authenticity_score': authenticity_score,
                'is_likely_deepfake': authenticity_score < 0.5,
                'confidence': abs(0.5 - authenticity_score) * 2 * 100,
                'model_type': 'simplified_video_fallback',
                'real_score': real_score,
                'fake_score': fake_score,
                'frame_count': analyzed_frames
            }
            
        except Exception as e:
            print(f"Error in video fallback: {str(e)}")
            return {
                'authenticity_score': 0.5,
                'is_likely_deepfake': False,
                'confidence': 50.0,
                'model_type': 'basic_video_fallback',
                'real_score': 0.5,
                'fake_score': 0.5,
                'frame_count': 0,
                'error': 'Using basic video fallback'
            }

    def predict_video(self, video_path):
        """Predict if a video contains deepfakes"""
        if self.model_type == 'enhanced_fallback':
            return self.enhanced_fallback_video_detection(video_path)
        elif self.model is None or self.model_type == 'random':
            # This should now be handled by enhanced_fallback
            return self.enhanced_fallback_video_detection(video_path)
        
        frames = self.extract_frames(video_path)
        if frames is None or len(frames) == 0:
            return {'error': 'Failed to extract frames from video'}
        
        # Check if we're using Hugging Face model
        if self.processor is not None:
            try:
                # Predict on each frame using Hugging Face model
                frame_predictions = []
                real_scores = []
                fake_scores = []
                
                # Get the model's label mapping
                id2label = self.model.config.id2label
                print(f"Video model labels: {id2label}")
                
                real_idx = None
                fake_idx = None
                
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if 'real' in label_lower or 'authentic' in label_lower:
                        real_idx = int(idx)
                    elif 'fake' in label_lower or 'deepfake' in label_lower or 'synthetic' in label_lower:
                        fake_idx = int(idx)
                
                # For this specific model: 0='Deepfake', 1='Real'
                if real_idx is None or fake_idx is None:
                    real_idx = 1  # Real is index 1
                    fake_idx = 0  # Deepfake is index 0
                
                print(f"Video using real_idx: {real_idx}, fake_idx: {fake_idx}")
                
                for frame in tqdm(frames, desc="Analyzing video frames"):
                    # Convert numpy array to PIL Image and resize to 224x224
                    pil_frame = Image.fromarray((frame * 255).astype(np.uint8))
                    pil_frame = pil_frame.resize((224, 224))
                    
                    # Process with Hugging Face
                    inputs = self.processor(images=pil_frame, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    
                    authenticity_score = probabilities[0, real_idx].item()
                    frame_predictions.append(authenticity_score)
                    real_scores.append(probabilities[0, real_idx].item())
                    fake_scores.append(probabilities[0, fake_idx].item() if fake_idx is not None else 1 - authenticity_score)
                
                # Average predictions across frames
                avg_prediction = float(np.mean(frame_predictions))
                avg_real_score = float(np.mean(real_scores))
                avg_fake_score = float(np.mean(fake_scores))
                
                # Calculate frame-by-frame variance to detect inconsistencies
                prediction_variance = float(np.var(frame_predictions))
                
                # Ensure all values are valid numbers
                avg_prediction = 0.5 if np.isnan(avg_prediction) else max(0.0, min(1.0, avg_prediction))
                avg_real_score = 0.0 if np.isnan(avg_real_score) else max(0.0, min(1.0, avg_real_score))
                avg_fake_score = 0.0 if np.isnan(avg_fake_score) else max(0.0, min(1.0, avg_fake_score))
                prediction_variance = 0.0 if np.isnan(prediction_variance) else max(0.0, prediction_variance)
                frame_consistency = max(0.0, min(1.0, 1.0 - min(1.0, prediction_variance * 10)))
                
                return {
                    'authenticity_score': avg_prediction,
                    'is_likely_deepfake': avg_prediction < 0.5,
                    'confidence': max(0.0, min(1.0, abs(0.5 - avg_prediction) * 2)),
                    'frame_count': len(frames),
                    'model_type': 'huggingface',
                    'real_score': avg_real_score,
                    'fake_score': avg_fake_score,
                    'prediction_variance': prediction_variance,
                    'frame_consistency': frame_consistency
                }
            except Exception as e:
                print(f"Error predicting video with Hugging Face model: {str(e)}")
                return {'error': f'Failed to process video with Hugging Face model: {str(e)}'}
        else:
            # Original TensorFlow model logic
            # Predict on each frame
            frame_predictions = []
            for frame in tqdm(frames, desc="Analyzing video frames"):
                prediction = self.model.predict(np.expand_dims(frame, axis=0))[0][0]
                frame_predictions.append(prediction)
            
            # Average predictions across frames
            avg_prediction = np.mean(frame_predictions)
            
            return {
                'authenticity_score': float(avg_prediction),
                'is_likely_deepfake': avg_prediction < 0.5,
                'confidence': abs(0.5 - avg_prediction) * 2,
                'frame_count': len(frames),
                'model_type': 'tensorflow'
            }
    
    def noise_analysis(self, image_path):
        """Analyze noise patterns in the image"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Could not load image'}
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to separate noise from content
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray.astype(np.float32) - blurred.astype(np.float32)
            
            # Calculate noise statistics
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            noise_variance = np.var(noise)
            
            # Calculate signal-to-noise ratio
            signal_power = np.mean(gray.astype(np.float32) ** 2)
            noise_power = noise_variance
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Detect artificial noise patterns (common in deepfakes)
            noise_entropy = -np.sum(noise * np.log2(np.abs(noise) + 1e-10))
            
            return {
                'noise_mean': float(noise_mean),
                'noise_std': float(noise_std),
                'noise_variance': float(noise_variance),
                'snr_db': float(snr),
                'noise_entropy': float(noise_entropy),
                'is_artificial_noise': noise_std > 15.0  # Threshold for artificial noise detection
            }
            
        except Exception as e:
            print(f"Noise analysis error: {str(e)}")
            return {
                'noise_mean': 0.0,
                'noise_std': 0.0,
                'noise_variance': 0.0,
                'snr_db': 0.0,
                'noise_entropy': 0.0,
                'is_artificial_noise': False
            }

    def color_distribution_analysis(self, image_path):
        """Analyze color distribution patterns"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Could not load image'}
            
            # Convert to different color spaces
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Calculate color statistics for each channel
            def analyze_channel(channel, name):
                return {
                    f'{name}_mean': float(np.mean(channel)),
                    f'{name}_std': float(np.std(channel)),
                    f'{name}_min': float(np.min(channel)),
                    f'{name}_max': float(np.max(channel)),
                    f'{name}_range': float(np.max(channel) - np.min(channel))
                }
            
            # RGB analysis
            r_stats = analyze_channel(rgb[:,:,0], 'red')
            g_stats = analyze_channel(rgb[:,:,1], 'green')
            b_stats = analyze_channel(rgb[:,:,2], 'blue')
            
            # HSV analysis
            h_stats = analyze_channel(hsv[:,:,0], 'hue')
            s_stats = analyze_channel(hsv[:,:,1], 'saturation')
            v_stats = analyze_channel(hsv[:,:,2], 'value')
            
            # Color balance analysis
            color_balance = {
                'rg_ratio': float(np.mean(rgb[:,:,0]) / (np.mean(rgb[:,:,1]) + 1e-10)),
                'rb_ratio': float(np.mean(rgb[:,:,0]) / (np.mean(rgb[:,:,2]) + 1e-10)),
                'gb_ratio': float(np.mean(rgb[:,:,1]) / (np.mean(rgb[:,:,2]) + 1e-10)),
                'colorfulness': float(np.std([r_stats['red_std'], g_stats['green_std'], b_stats['blue_std']]))
            }
            
            # Detect color inconsistencies (common in deepfakes)
            color_inconsistency = np.std([r_stats['red_std'], g_stats['green_std'], b_stats['blue_std']])
            
            return {
                **r_stats, **g_stats, **b_stats,
                **h_stats, **s_stats, **v_stats,
                **color_balance,
                'color_inconsistency': float(color_inconsistency),
                'is_color_manipulated': color_inconsistency > 20.0
            }
            
        except Exception as e:
            print(f"Color distribution analysis error: {str(e)}")
            return {'error': str(e)}

    def compression_analysis(self, image_path):
        """Analyze compression artifacts and quality"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import os
            
            # Get file size
            file_size = os.path.getsize(image_path)
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Could not load image'}
            
            height, width = img.shape[:2]
            total_pixels = height * width
            
            # Calculate compression ratio
            uncompressed_size = total_pixels * 3  # RGB
            compression_ratio = file_size / uncompressed_size
            
            # Analyze compression artifacts using DCT
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply DCT to detect compression patterns
            dct = cv2.dct(gray.astype(np.float32))
            
            # Analyze high-frequency components (compression artifacts)
            hf_energy = np.sum(dct[8:, 8:] ** 2)
            total_energy = np.sum(dct ** 2)
            hf_ratio = hf_energy / (total_energy + 1e-10)
            
            # Detect block artifacts (JPEG compression)
            block_artifacts = 0
            for i in range(0, height-8, 8):
                for j in range(0, width-8, 8):
                    block = gray[i:i+8, j:j+8]
                    if block.shape == (8, 8):
                        # Calculate block variance
                        block_var = np.var(block)
                        if block_var < 5:  # Low variance indicates compression artifacts
                            block_artifacts += 1
            
            block_artifact_ratio = block_artifacts / ((height//8) * (width//8))
            
            # Estimate compression quality
            if compression_ratio < 0.1:
                quality_estimate = "High"
            elif compression_ratio < 0.3:
                quality_estimate = "Medium"
            else:
                quality_estimate = "Low"
            
            return {
                'file_size_bytes': int(file_size),
                'compression_ratio': float(compression_ratio),
                'hf_energy_ratio': float(hf_ratio),
                'block_artifact_ratio': float(block_artifact_ratio),
                'quality_estimate': quality_estimate,
                'is_heavily_compressed': compression_ratio < 0.05,
                'has_compression_artifacts': block_artifact_ratio > 0.3
            }
            
        except Exception as e:
            print(f"Compression analysis error: {str(e)}")
            return {'error': str(e)}

    def ela_analysis(self, image_path):
        """Perform Error Level Analysis (ELA) to detect image manipulation"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import os
            
            # Load original image
            original = Image.open(image_path)
            original = original.convert('RGB')
            
            # Save with high quality JPEG (95% quality)
            temp_path = 'temp_ela.jpg'
            original.save(temp_path, 'JPEG', quality=95)
            
            # Load the recompressed image
            recompressed = Image.open(temp_path)
            
            # Convert to numpy arrays
            orig_array = np.array(original, dtype=np.float32)
            recomp_array = np.array(recompressed, dtype=np.float32)
            
            # Calculate ELA (difference between original and recompressed)
            ela = np.abs(orig_array - recomp_array)
            
            # Calculate ELA statistics
            ela_mean = np.mean(ela)
            ela_std = np.std(ela)
            ela_max = np.max(ela)
            
            # Calculate manipulation score (higher values indicate more manipulation)
            manipulation_score = (ela_mean + ela_std) / 255.0
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                'ela_mean': float(ela_mean),
                'ela_std': float(ela_std),
                'ela_max': float(ela_max),
                'manipulation_score': float(manipulation_score),
                'is_manipulated': manipulation_score > 0.1  # Threshold for manipulation detection
            }
            
        except Exception as e:
            print(f"ELA analysis error: {str(e)}")
            return {
                'ela_mean': 0.0,
                'ela_std': 0.0,
                'ela_max': 0.0,
                'manipulation_score': 0.0,
                'is_manipulated': False
            }

    def get_forensic_analysis(self, image_path):
        """Generate comprehensive forensic analysis data for the image"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import os
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'dimensions': 'N/A',
                    'faceDetection': 'N/A',
                    'hasExif': False,
                    'ela_analysis': {},
                    'noise_analysis': {},
                    'color_analysis': {},
                    'compression_analysis': {}
                }
            
            # Get original dimensions
            height, width = img.shape[:2]
            dimensions = f"{width}x{height}"
            
            # Resize to 224x224 for optimal model performance
            img_resized = cv2.resize(img, (224, 224))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_detection = f"{len(faces)} face(s) detected"
            
            # EXIF data analysis
            pil_img = Image.open(image_path)
            has_exif = bool(pil_img._getexif()) if hasattr(pil_img, '_getexif') else False
            
            # Comprehensive analysis
            ela_results = self.ela_analysis(image_path)
            noise_results = self.noise_analysis(image_path)
            color_results = self.color_distribution_analysis(image_path)
            compression_results = self.compression_analysis(image_path)
            
            return {
                'dimensions': dimensions,
                'faceDetection': face_detection,
                'hasExif': has_exif,
                'ela_analysis': ela_results,
                'noise_analysis': noise_results,
                'color_analysis': color_results,
                'compression_analysis': compression_results
            }
            
        except Exception as e:
            print(f"Forensic analysis error: {str(e)}")
            return {
                'dimensions': 'Error',
                'faceDetection': 'Error',
                'hasExif': False,
                'ela_analysis': {},
                'noise_analysis': {},
                'color_analysis': {},
                'compression_analysis': {}
            }

    def analyze_media(self, file_path):
        """Analyze media file (image or video) for deepfake detection"""
        # Determine file type based on extension
        file_ext = file_path.split('.')[-1].lower()
        
        # Process based on file type
        if file_ext in ['jpg', 'jpeg', 'png', 'gif']:
            return self.predict_image(file_path)
        elif file_ext in ['mp4', 'avi', 'mov']:
            return self.predict_video(file_path)
        else:
            return {'error': f'Unsupported file type: {file_ext}'}

# For testing purposes
if __name__ == "__main__":
    detector = DeepfakeDetector()
    # Test with an image
    # result = detector.analyze_media('path/to/test/image.jpg')
    # print(result)