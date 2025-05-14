#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import subprocess
from tqdm import tqdm
import warnings
import logging
import time

# Setup logging if not already set up
logger = logging.getLogger('mmsa_audio')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

warnings.filterwarnings('ignore')

# Create a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class AudioSentimentAnalyzer:
    """
    Audio sentiment analyzer using CNN model trained on RAVDESS dataset,
    following the notebook approach for accurate sentiment detection
    """
    
    def __init__(self, model_path=None, norm_params_path=None):
        """
        Initialize the audio sentiment analyzer
        
        Args:
            model_path (str): Path to pre-trained model. If None, the model will need to be trained.
            norm_params_path (str): Path to normalization parameters. If None, will try to infer from model_path.
        """
        # Audio parameters
        self.sampling_rate = 44100
        self.audio_duration = 2.5
        self.n_mfcc = 30
        
        # Emotions in RAVDESS dataset (in order)
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Sentiment mapping with enhanced positive emotion weights
        self.sentiment_weights = {
            'happy': 1.0,       # Maximum positive (was 0.9)
            'surprised': 0.7,   # More positive (was 0.5)
            'calm': 0.5,        # Moderately positive (was 0.3)
            'neutral': 0.0,     # Neutral
            'fearful': -0.6,    # Moderate negative
            'sad': -0.7,        # Strong negative (was -0.8)
            'disgust': -0.7,    # Strong negative (was -0.8)
            'angry': -0.8       # Very strong negative (was -0.9)
        }
        
        # Model and normalization parameters
        self.model = None
        self.mean = None
        self.std = None
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
                
                # If norm_params_path not provided, try to infer from model_path
                if norm_params_path is None and model_path:
                    base_path = os.path.splitext(model_path)[0]
                    potential_norm_params = f"{base_path}_norm_params.json"
                    if os.path.exists(potential_norm_params):
                        norm_params_path = potential_norm_params
                    elif os.path.exists("audio_norm_params.json"):
                        # Try the default location as well
                        norm_params_path = "audio_norm_params.json"
                
                # Load normalization parameters if provided or inferred
                if norm_params_path and os.path.exists(norm_params_path):
                    try:
                        with open(norm_params_path, 'r') as f:
                            params = json.load(f)
                        self.mean = np.array(params['mean'])
                        self.std = np.array(params['std'])
                        logger.info(f"Normalization parameters loaded from {norm_params_path}")
                    except Exception as e:
                        logger.error(f"Error loading normalization parameters: {str(e)}")
                else:
                    logger.warning("No normalization parameters found. For best results, provide normalization parameters.")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
        else:
            if model_path:
                logger.warning(f"Model file not found: {model_path}")
            logger.warning("No model loaded. You'll need to train a model or provide a valid model path.")
        
        logger.info(f"Using audio emotion weights: {self.sentiment_weights}")
    
    def extract_audio_from_video(self, video_path, output_path=None):
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path (str): Path to video file
            output_path (str): Path to save extracted audio. If None, a temporary file is created.
            
        Returns:
            str: Path to extracted audio file
        """
        try:
            # Create output path if not provided
            if output_path is None:
                # Create temp directory if it doesn't exist
                os.makedirs('tmp_audio', exist_ok=True)
                
                # Generate output path with same name as video but .wav extension
                base_name = os.path.basename(video_path).split('.')[0]
                timestamp = int(time.time())
                output_path = os.path.join('tmp_audio', f"temp_{base_name}_{timestamp}.wav")
                logger.debug(f"Created temporary audio path: {output_path}")
            
            # Extract audio using ffmpeg
            logger.debug(f"Extracting audio with ffmpeg: {video_path} -> {output_path}")
            subprocess.run([
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '44100', '-ac', '1', output_path, '-y'
            ], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            if os.path.exists(output_path):
                logger.info(f"Audio extracted successfully: {output_path}")
                return output_path
            else:
                logger.warning(f"Failed to extract audio from {video_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {str(e)}")
            return None
    
    def extract_features(self, file_path):
        """
        Extract MFCC features from an audio file
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            numpy.ndarray: Extracted MFCC features
        """
        try:
            # Load audio with specific duration and offset for consistency
            data, _ = librosa.load(file_path, sr=self.sampling_rate, 
                                   duration=self.audio_duration, offset=0.5)
            
            input_length = self.sampling_rate * self.audio_duration
            
            # Handle audio length differences with padding or cropping
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")
            
            # Extract MFCC features (2D representation)
            mfcc = librosa.feature.mfcc(y=data, sr=self.sampling_rate, n_mfcc=self.n_mfcc)
            mfcc = np.expand_dims(mfcc, axis=-1)
            
            return mfcc
        
        except Exception as e:
            logger.error(f"Error extracting features from {file_path}: {str(e)}")
            return None
    
    def build_model(self):
        """
        Build the 2D CNN model following the notebook implementation
        
        Returns:
            tensorflow.keras.models.Model: CNN model
        """
        # Input shape is (n_mfcc, timesteps, 1)
        inp = Input(shape=(self.n_mfcc, 216, 1), name='input_layer')
        
        # First Conv2D layer
        x = Conv2D(32, (4,10), padding="same")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D()(x)
        x = Dropout(rate=0.2)(x)
        
        # Second Conv2D layer
        x = Conv2D(32, (4,10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D()(x)
        x = Dropout(rate=0.2)(x)
        
        # Third Conv2D layer
        x = Conv2D(32, (4,10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D()(x)
        x = Dropout(rate=0.2)(x)
        
        # Fourth Conv2D layer
        x = Conv2D(32, (4,10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D()(x)
        x = Dropout(rate=0.2)(x)
        
        # Flatten layer
        x = Flatten()(x)
        
        # Dense layers
        x = Dense(64)(x)
        x = Dropout(rate=0.2)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(rate=0.2)(x)
        
        # Output layer for 8 emotions
        out = Dense(len(self.emotions), activation='softmax')(x)
        
        model = Model(inputs=inp, outputs=out)
        
        # Optimizer
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def prepare_data(self, df):
        """
        Prepare data for training, extracting MFCC features from audio files
        
        Args:
            df: DataFrame with 'path' column containing paths to audio files
        
        Returns:
            X: Array of MFCC features
        """
        X = np.empty(shape=(df.shape[0], self.n_mfcc, 216, 1))
        
        logger.info("Extracting MFCC features from audio files...")
        for i, file_path in enumerate(tqdm(df.path)):
            try:
                features = self.extract_features(file_path)
                if features is not None:
                    X[i,] = features
                else:
                    # Fill with zeros if extraction failed
                    X[i,] = np.zeros((self.n_mfcc, 216, 1))
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                X[i,] = np.zeros((self.n_mfcc, 216, 1))
        
        return X
    
    def train(self, data_path, test_size=0.25, epochs=50, batch_size=16, save_path='audio_sentiment_model.h5'):
        """
        Train the model on RAVDESS dataset
        
        Args:
            data_path (str): Path to RAVDESS dataset
            test_size (float): Portion of data to use for testing
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            save_path (str): Path to save the trained model
            
        Returns:
            dict: Training results
        """
        # Check if data path exists
        if not os.path.exists(data_path):
            logger.warning(f"Data path not found: {data_path}")
            return None
        
        # Find all wav files
        file_paths = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            logger.warning(f"No audio files found in {data_path}")
            return None
        
        logger.info(f"Found {len(file_paths)} audio files")
        
        # Create dataframe
        df = pd.DataFrame({'path': file_paths})
        
        # Extract emotion from file names (RAVDESS format)
        emotions = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            parts = file_name.split('-')
            
            try:
                if len(parts) >= 3:
                    emotion_code = int(parts[2])
                    emotion_idx = emotion_code - 1  # 0-based indexing
                    emotions.append(emotion_idx)
                else:
                    emotions.append(0)  # Default to neutral
            except:
                emotions.append(0)  # Default to neutral
        
        df['emotion'] = emotions
        
        # Prepare data - extract MFCC features
        X = self.prepare_data(df)
        
        # Create labels
        y = np.array(df.emotion)
        y_onehot = to_categorical(y, num_classes=len(self.emotions))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=test_size, shuffle=True, random_state=42
        )
        
        # Normalize the data
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)
        
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1e-6, self.std)
        
        X_train = (X_train - self.mean) / self.std
        X_test = (X_test - self.mean) / self.std
        
        # Build model
        self.model = self.build_model()
        self.model.summary()
        
        # Train model
        logger.info(f"Training model for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Save model and normalization parameters
        if save_path:
            # Create directory if needed
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save model
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
            
            # Save normalization parameters
            norm_params = {
                'mean': self.mean.tolist() if isinstance(self.mean, np.ndarray) else self.mean,
                'std': self.std.tolist() if isinstance(self.std, np.ndarray) else self.std
            }
            
            norm_params_path = f"{os.path.splitext(save_path)[0]}_norm_params.json"
            with open(norm_params_path, 'w') as f:
                json.dump(norm_params, f, cls=NumpyEncoder)
            
            logger.info(f"Normalization parameters saved to {norm_params_path}")
        
        # Create training curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('audio_training_curves.png')
        
        return {
            'accuracy': accuracy,
            'history': history.history,
            'model_path': save_path,
            'norm_params_path': norm_params_path if save_path else None
        }
    
    def predict_from_file(self, file_path):
        """
        Predict emotion and sentiment from audio file
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            dict: Prediction results with emotion, confidence, and sentiment score
        """
        if self.model is None:
            logger.warning("No model loaded. Please train or load a model first.")
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': {e: 0.0 for e in self.emotions},
                'sentiment_score': 0.0,
                'error': "No model loaded"
            }
        
        try:
            # Extract features
            features = self.extract_features(file_path)
            if features is None:
                logger.error(f"Failed to extract features from {file_path}")
                return {
                    'emotion': 'neutral',
                    'confidence': 0.5,
                    'all_emotions': {e: 0.0 for e in self.emotions},
                    'sentiment_score': 0.0,
                    'error': "Feature extraction failed"
                }
            
            # Add batch dimension
            features = np.expand_dims(features, axis=0)
            
            # Normalize if mean and std available
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
            
            # Advanced shape compatibility handling
            input_shape = None
            try:
                # Get expected input shape from model
                input_shape = self.model.input_shape
                current_shape = features.shape
                logger.debug(f"Model expects shape: {input_shape}, Current feature shape: {current_shape}")
                
                # Check for Conv1D vs Conv2D mismatch
                if len(input_shape) == 3 and len(current_shape) == 4:
                    # Model expects 3D input (batch, time_steps, features) but we have 4D
                    # This is typically for Conv1D layers
                    logger.info("Adapting 4D features to 3D model input (Conv1D)")
                    features = features.squeeze(axis=-1)
                
                # Crucial fix for Conv1D models that expect a different feature dimension
                if len(input_shape) == 3 and len(features.shape) == 3:
                    if input_shape[1] != features.shape[1]:
                        # Need to reshape to match the expected time_steps dimension
                        logger.warning(f"Dimension mismatch at axis 1: model expects {input_shape[1]}, got {features.shape[1]}")
                        
                        # Handle case where model expects specific dimensions like (batch, 40, 1)
                        if input_shape[2] == 1:
                            logger.info(f"Advanced reshape for precise model compatibility")
                            
                            # For Conv1D, we need to match the exact expected shape (batch, 40, 1)
                            # Determine the target shape
                            target_shape = tuple(dim if dim is not None else features.shape[i] for i, dim in enumerate(input_shape))
                            logger.info(f"Target model shape: {target_shape}")
                            
                            # 1. Reduce to single channel if needed
                            if features.shape[2] > 1:
                                # Take mean across features to get single channel
                                features = np.mean(features, axis=2, keepdims=True)
                                logger.info(f"After feature reduction: {features.shape}")
                            
                            # 2. Resize time dimension to match expected input
                            original_time_steps = features.shape[1]
                            target_time_steps = input_shape[1]
                            
                            if original_time_steps != target_time_steps:
                                # Resize to match expected time steps
                                # Option 1: Truncate or pad if reasonable
                                if original_time_steps > target_time_steps:
                                    # Truncate - take center portion
                                    start = (original_time_steps - target_time_steps) // 2
                                    features = features[:, start:start+target_time_steps, :]
                                    logger.info(f"Truncated time steps from {original_time_steps} to {target_time_steps}")
                                else:
                                    # Pad with zeros
                                    pad_width = ((0, 0), (0, target_time_steps - original_time_steps), (0, 0))
                                    features = np.pad(features, pad_width, mode='constant')
                                    logger.info(f"Padded time steps from {original_time_steps} to {target_time_steps}")
                            
                            logger.info(f"Final features shape: {features.shape}")
                            
                            # One final check to ensure the shape is compatible
                            if features.shape[1:] != input_shape[1:]:
                                logger.warning(f"Shape mismatch persists. Model expects {input_shape[1:]} but got {features.shape[1:]}")
                                
                                # As a last resort, completely reshape to target dimensions
                                # This is not ideal for accuracy but will at least make the model run
                                batch_size = features.shape[0]
                                try:
                                    # Completely reshape to match expected dimensions - preserving batch size
                                    features = np.reshape(features, (batch_size,) + input_shape[1:])
                                    logger.info(f"Force-reshaped features to {features.shape}")
                                except Exception as e:
                                    logger.error(f"Reshape failed: {str(e)}")
                
                elif len(input_shape) == 4 and len(current_shape) == 3:
                    # Model expects 4D input (batch, height, width, channels) but we have 3D
                    # This is typically for Conv2D layers
                    logger.info("Adapting 3D features to 4D model input (Conv2D)")
                    features = np.expand_dims(features, axis=-1)
                
                # Check for channel dimension mismatch in 4D tensors
                elif (len(input_shape) == 4 and len(current_shape) == 4 and 
                      input_shape[-1] != current_shape[-1] and input_shape[-1] is not None):
                    logger.info(f"Reshaping features to match expected channels: {input_shape[-1]}")
                    features = np.reshape(features, (current_shape[0], current_shape[1], 
                                               current_shape[2], input_shape[-1]))
                
                # Check if we need to match exact dimensions for middle axes
                for i in range(1, min(len(input_shape), len(features.shape))-1):
                    if input_shape[i] is not None and features.shape[i] != input_shape[i]:
                        logger.warning(f"Dimension mismatch at axis {i}: model expects {input_shape[i]}, got {features.shape[i]}")
            except Exception as e:
                logger.warning(f"Failed to adapt feature shape: {str(e)}")
            
            # Predict
            logger.debug(f"Running prediction with feature shape: {features.shape}")
            try:
                prediction = self.model.predict(features, verbose=0)[0]
                emotion_idx = np.argmax(prediction)
                emotion = self.emotions[emotion_idx]
                confidence = float(prediction[emotion_idx])
                
                # Map to sentiment score
                sentiment_score = self.sentiment_weights.get(emotion, 0.0)
                
                # Create result
                result = {
                    'emotion': emotion,
                    'confidence': confidence,
                    'all_emotions': {e: float(prediction[i]) for i, e in enumerate(self.emotions)},
                    'sentiment_score': sentiment_score
                }
                
                return result
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                logger.warning("Using fallback prediction method")
                
                # Fallback: Use a simplified heuristic based on audio characteristics
                try:
                    # Calculate basic audio statistics as a simple fallback
                    audio_data, _ = librosa.load(file_path, sr=self.sampling_rate, duration=self.audio_duration)
                    
                    # Calculate energy and other basic features
                    energy = np.mean(librosa.feature.rms(y=audio_data))
                    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data))
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=self.sampling_rate))
                    
                    # Simple heuristic for emotion
                    # High energy + high zero crossing rate often correlates with anger/excitement
                    # Low energy often correlates with sadness/calmness
                    if energy > 0.1 and zero_crossing_rate > 0.05:
                        fallback_emotion = 'angry' if spectral_centroid > 2000 else 'surprised'
                        confidence = min(energy * 5, 0.7)  # Cap at 0.7 confidence
                        sentiment_score = -0.6 if fallback_emotion == 'angry' else 0.3
                    elif energy < 0.05:
                        fallback_emotion = 'sad'
                        confidence = min((1 - energy * 10), 0.6)
                        sentiment_score = -0.7
                    else:
                        fallback_emotion = 'neutral'
                        confidence = 0.5
                        sentiment_score = 0.0
                    
                    logger.info(f"Fallback prediction: {fallback_emotion} (confidence: {confidence:.2f})")
                    
                    return {
                        'emotion': fallback_emotion,
                        'confidence': float(confidence),
                        'all_emotions': {e: (0.7 if e == fallback_emotion else 0.0) for e in self.emotions},
                        'sentiment_score': sentiment_score,
                        'fallback': True,
                        'fallback_reason': str(e)
                    }
                except Exception as fallback_error:
                    logger.error(f"Fallback prediction also failed: {str(fallback_error)}")
                    # Last resort - return neutral with low confidence
                    return {
                        'emotion': 'neutral',
                        'confidence': 0.5,
                        'all_emotions': {e: 0.0 for e in self.emotions},
                        'sentiment_score': 0.0,
                        'error': str(e),
                        'fallback': True
                    }
            
        except Exception as e:
            logger.error(f"Error predicting from audio: {str(e)}", exc_info=True)
            
            # Return a default result on error with error information
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': {e: 0.0 for e in self.emotions},
                'sentiment_score': 0.0,
                'error': str(e)
            }
    
    def predict_from_video(self, video_path):
        """
        Extract audio from video and predict emotion
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Predicted emotion, confidence, and sentiment score
        """
        try:
            # Extract audio from video
            audio_path = self.extract_audio_from_video(video_path)
            
            if not audio_path or not os.path.exists(audio_path):
                logger.warning(f"Failed to extract audio from {video_path}")
                return None
            
            # Predict from extracted audio
            result = self.predict_from_file(audio_path)
            
            # Delete the temporary audio file
            try:
                if os.path.dirname(audio_path).startswith('tmp_'):
                    os.remove(audio_path)
                    logger.debug(f"Deleted temporary audio file: {audio_path}")
            except Exception as e:
                logger.warning(f"Error deleting temporary file {audio_path}: {str(e)}")
            
            # Apply special handling for happy, surprised or calm emotions
            # to boost detection of positive sentiment in audio
            if result:
                emotion = result.get('emotion', '')
                confidence = result.get('confidence', 0)
                
                # Apply post-processing to boost happy/positive emotions in audio
                if emotion in ['happy', 'surprised', 'calm'] and confidence > 0.3:
                    # Apply a boost to the sentiment score for positive emotions
                    original_score = result.get('sentiment_score', 0)
                    
                    # Calculate boost based on emotion type and confidence
                    boost_factor = 0.0
                    if emotion == 'happy':
                        boost_factor = 0.2  # Strong boost for happy
                    elif emotion == 'surprised':
                        boost_factor = 0.15  # Medium boost for surprised
                    elif emotion == 'calm':
                        boost_factor = 0.1   # Slight boost for calm
                    
                    # Scale boost by confidence level
                    boost_amount = boost_factor * confidence
                    new_score = min(1.0, original_score + boost_amount)
                    
                    logger.info(f"Applied audio {emotion} boost: {original_score:.2f} → {new_score:.2f}")
                    result['sentiment_score'] = new_score
                    result['original_sentiment_score'] = original_score
                
                # Check if we have probabilities for all emotions
                if 'probabilities' in result:
                    probs = result['probabilities']
                    
                    # Check if there's a mix of happy/surprised with other emotions
                    happy_prob = probs.get('happy', 0)
                    surprised_prob = probs.get('surprised', 0)
                    calm_prob = probs.get('calm', 0)
                    
                    # If significant positive components exist but not dominant,
                    # still apply a moderate boost
                    positive_component = happy_prob + surprised_prob + calm_prob
                    if positive_component > 0.3 and emotion not in ['happy', 'surprised', 'calm']:
                        original_score = result.get('sentiment_score', 0)
                        boost_amount = positive_component * 0.2
                        new_score = min(0.7, original_score + boost_amount)
                        
                        logger.info(f"Applied mixed positive audio boost: {original_score:.2f} → {new_score:.2f}")
                        result['sentiment_score'] = new_score
                        result['original_sentiment_score'] = original_score
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting from video {video_path}: {str(e)}")
            return None

def main():
    """
    Main function for testing the audio sentiment analyzer
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Sentiment Analysis')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--data', type=str, default='ravdess_data', help='Path to training data')
    parser.add_argument('--model', type=str, default='audio_sentiment_model.h5', help='Path to model file')
    parser.add_argument('--input', type=str, help='Audio or video file to analyze')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Determine norm params path based on model path
    norm_params_path = os.path.splitext(args.model)[0] + '_norm_params.json'
    
    # Create analyzer
    analyzer = AudioSentimentAnalyzer(
        model_path=None if args.train else args.model,
        norm_params_path=None if args.train else norm_params_path
    )
    
    # Train or predict
    if args.train:
        logger.info(f"Training new model with data from {args.data}")
        results = analyzer.train(args.data, epochs=args.epochs, save_path=args.model)
        if results:
            logger.info(f"Training completed with accuracy: {results['accuracy'] * 100:.2f}%")
    
    elif args.input:
        if not analyzer.model:
            logger.warning("No model loaded. Please train a model or provide a valid model path.")
            return
        
        logger.info(f"Analyzing {args.input}...")
        
        # Detect if input is audio or video
        if args.input.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
            result = analyzer.predict_from_file(args.input)
        else:
            result = analyzer.predict_from_video(args.input)
        
        if result:
            logger.info(f"Emotion: {result['emotion']} (Confidence: {result['confidence']:.4f})")
            logger.info(f"Sentiment Score: {result['sentiment_score']:.2f}")
            logger.info("All emotions:")
            for emotion, score in sorted(result['all_emotions'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {emotion}: {score:.4f}")
        else:
            logger.warning("Analysis failed.")
    
    else:
        logger.warning("No action specified. Use --train to train a model or --input to analyze a file.")

if __name__ == "__main__":
    main() 