#!/usr/bin/env python3

"""
Audio sentiment analyzer for MMSA project.
Provides emotion and sentiment analysis from audio files.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
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
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
        self.emotion_weights = {
            'happy': 1.0,       # Maximum positive (was 0.9)
            'surprised': 0.7,   # More positive (was 0.5)
            'calm': 0.5,        # Moderately positive (was 0.3)
            'neutral': 0.0,     # Neutral
            'fearful': -0.6,    # Moderate negative
            'sad': -0.7,        # Strong negative (was -0.8)
            'disgust': -0.7,    # Strong negative (was -0.8)
            'angry': -0.8       # Very strong negative (was -0.9)
        }
        
        # Store for model paths
        self.model_path = model_path
        self.norm_params_path = norm_params_path
        
        # Model and normalization parameters
        self.model = None
        self.mean = None
        self.std = None
        self.last_prediction = None
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            try:
                logger.info(f"Loading audio model from {model_path}")
                
                # Improved model loading with additional verification
                try:
                    # First try with custom object scope for compatibility
                    with tf.keras.utils.custom_object_scope({'CustomLayer': tf.keras.layers.Layer}):
                        self.model = load_model(model_path)
                except:
                    # Try standard loading if custom scope fails
                    self.model = load_model(model_path)
                
                logger.info(f"Model loaded successfully from {model_path}")
                
                # Verify model is functional by checking structure
                if hasattr(self.model, 'layers'):
                    logger.info(f"Model has {len(self.model.layers)} layers")
                    logger.info(f"Model input shape: {self.model.input_shape}")
                    logger.info(f"Model output shape: {self.model.output_shape}")
                else:
                    logger.warning("Model structure appears invalid - missing layers attribute")
                
                # Try to compile the model if it's not already compiled
                try:
                    # Use loss and optimizer attributes as a safer way to check if compiled
                    if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
                        logger.warning("Model may not be compiled. Attempting to compile with default settings...")
                        self.model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        logger.info("Model compiled successfully")
                except Exception as e:
                    logger.warning(f"Error checking/compiling model: {str(e)}")
                
                # If norm_params_path not provided, try to infer from model_path
                if norm_params_path is None and model_path:
                    base_path = os.path.splitext(model_path)[0]
                    potential_norm_params = f"{base_path}_norm_params.json"
                    if os.path.exists(potential_norm_params):
                        norm_params_path = potential_norm_params
                        self.norm_params_path = norm_params_path
                        logger.info(f"Found norm params at inferred path: {norm_params_path}")
                    elif os.path.exists("audio_norm_params.json"):
                        # Try the default location as well
                        norm_params_path = "audio_norm_params.json"
                        self.norm_params_path = norm_params_path
                        logger.info(f"Using default norm params path: {norm_params_path}")
                
                # Load normalization parameters if provided or inferred
                if norm_params_path and os.path.exists(norm_params_path):
                    try:
                        with open(norm_params_path, 'r') as f:
                            params = json.load(f)
                        self.mean = np.array(params['mean'])
                        self.std = np.array(params['std'])
                        logger.info(f"Normalization parameters loaded from {norm_params_path}")
                        logger.info(f"Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
                    except Exception as e:
                        logger.error(f"Error loading normalization parameters: {str(e)}")
                else:
                    logger.warning("No normalization parameters found. For best results, provide normalization parameters.")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}", exc_info=True)
                logger.error(f"Model path: {model_path}")
                logger.error(f"Model path exists: {os.path.exists(model_path)}")
                
                # Try to list files in the directory to help debugging
                try:
                    model_dir = os.path.dirname(model_path)
                    if model_dir and os.path.exists(model_dir):
                        logger.info(f"Files in model directory: {os.listdir(model_dir)}")
                except:
                    pass
        else:
            if model_path:
                logger.warning(f"Model file not found: {model_path}")
                # Try to find H5 files in the directory
                try:
                    model_dir = os.path.dirname(model_path) or "."
                    h5_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
                    if h5_files:
                        logger.info(f"Available H5 files: {h5_files}")
                except:
                    pass
            logger.warning("No model loaded. You'll need to train a model or provide a valid model path.")
        
        logger.info(f"Using audio emotion weights: {self.emotion_weights}")
    
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
            dict: Dictionary with prediction results
        """
        try:
            logger.info(f"Predicting from file: {file_path}")
            
            # Ensure we have a model
            if self.model is None:
                # If we have a model path, try loading again
                if self.model_path and os.path.exists(self.model_path):
                    logger.warning("Model not loaded yet. Attempting to load from saved path...")
                    try:
                        self.model = load_model(self.model_path)
                        logger.info(f"Model loaded successfully from {self.model_path}")
                    except Exception as e:
                        logger.error(f"Failed to load model: {e}", exc_info=True)
                        raise ValueError("No model available for prediction")
                else:
                    raise ValueError("No model available for prediction")
            
            # Ensure we have normalization parameters
            if self.mean is None or self.std is None:
                if self.norm_params_path and os.path.exists(self.norm_params_path):
                    logger.warning("Missing normalization parameters. Attempting to load from saved path...")
                    try:
                        with open(self.norm_params_path, 'r') as f:
                            params = json.load(f)
                        self.mean = np.array(params['mean'])
                        self.std = np.array(params['std'])
                        logger.info(f"Normalization parameters loaded from {self.norm_params_path}")
                    except Exception as e:
                        logger.error(f"Failed to load normalization parameters: {e}", exc_info=True)
                        raise ValueError("Missing normalization parameters required for prediction")
                else:
                    raise ValueError("Missing normalization parameters required for prediction")
            
            # Load audio file
            try:
                audio_data, _ = librosa.load(file_path, sr=self.sampling_rate, duration=self.audio_duration)
                
                # Check if audio data is valid
                if len(audio_data) == 0:
                    raise ValueError("Empty audio data")
                
                # Print some metadata for debugging
                logger.info(f"Audio data shape: {audio_data.shape}")
                logger.info(f"Audio data min/max: {np.min(audio_data)}/{np.max(audio_data)}")
                logger.info(f"Audio data mean/std: {np.mean(audio_data)}/{np.std(audio_data)}")
                
                # Extract features
                mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sampling_rate, n_mfcc=self.n_mfcc)
                
                # Log shapes for debugging
                logger.info(f"MFCC shape: {mfccs.shape}")
                logger.info(f"Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
                
                # Handle different shapes and dimensions for normalization
                if len(self.mean.shape) == 3 and self.mean.shape[-1] == 1:  # If mean has shape (30, 216, 1)
                    # Reshape MFCCs to match normalization parameters
                    mfccs_reshaped = np.expand_dims(mfccs, axis=-1)  # Shape becomes (30, t, 1)
                    logger.info(f"Reshaped MFCCs to: {mfccs_reshaped.shape}")
                    
                    # Adjust mean and std shapes for broadcasting if needed
                    if self.mean.shape[1] == 1:
                        # If mean has shape (30, 1, 1), we need to broadcast it
                        mean_broadcast = np.broadcast_to(self.mean, (self.mean.shape[0], mfccs_reshaped.shape[1], 1))
                        std_broadcast = np.broadcast_to(self.std, (self.std.shape[0], mfccs_reshaped.shape[1], 1))
                        logger.info(f"Broadcasting normalization params to match MFCC shape")
                        # Normalize
                        mfccs_norm = (mfccs_reshaped - mean_broadcast) / std_broadcast
                    elif self.mean.shape[1] == 216 and mfccs_reshaped.shape[1] != 216:
                        # If MFCCs have different time dimension, resize
                        logger.info(f"Resizing MFCCs time dimension from {mfccs_reshaped.shape[1]} to 216")
                        # Resize time dimension to match norm params
                        from scipy.ndimage import zoom
                        zoom_factor = 216 / mfccs_reshaped.shape[1]
                        mfccs_resized = zoom(mfccs_reshaped, (1, zoom_factor, 1), order=1)
                        logger.info(f"Resized MFCCs to: {mfccs_resized.shape}")
                        mfccs_norm = (mfccs_resized - self.mean) / self.std
                    else:
                        # Standard normalization with broadcasting
                        mfccs_norm = (mfccs_reshaped - self.mean) / self.std
                elif len(self.mean.shape) == 4 and self.mean.shape[1] == 1:  # If mean has shape (30, 1, 216, 1)
                    # Handle the case where mean and std have an extra dimension
                    mfccs_reshaped = np.expand_dims(mfccs, axis=-1)  # Shape becomes (30, t, 1)
                    logger.info(f"Reshaped MFCCs to: {mfccs_reshaped.shape}")
                    
                    # Remove the extra dimension from mean and std
                    mean_reshaped = np.squeeze(self.mean, axis=1)  # Shape becomes (30, 216, 1)
                    std_reshaped = np.squeeze(self.std, axis=1)    # Shape becomes (30, 216, 1)
                    logger.info(f"Reshaped mean to: {mean_reshaped.shape}, std to: {std_reshaped.shape}")
                    
                    if mfccs_reshaped.shape[1] != 216:
                        # If MFCCs have different time dimension, resize
                        logger.info(f"Resizing MFCCs time dimension from {mfccs_reshaped.shape[1]} to 216")
                        from scipy.ndimage import zoom
                        zoom_factor = 216 / mfccs_reshaped.shape[1]
                        mfccs_resized = zoom(mfccs_reshaped, (1, zoom_factor, 1), order=1)
                        logger.info(f"Resized MFCCs to: {mfccs_resized.shape}")
                        mfccs_norm = (mfccs_resized - mean_reshaped) / std_reshaped
                    else:
                        # Standard normalization
                        mfccs_norm = (mfccs_reshaped - mean_reshaped) / std_reshaped
                else:
                    # Fallback case for simpler shapes
                    logger.info("Using simple normalization approach with reshape")
                    # Reshape mean and std to make them compatible with broadcasting
                    mean_flat = np.reshape(self.mean, (self.mean.shape[0], -1))[:, 0]
                    std_flat = np.reshape(self.std, (self.std.shape[0], -1))[:, 0]
                    
                    # Apply normalization with broadcasting
                    mfccs_norm = ((mfccs.T - mean_flat) / std_flat).T
                    # Add channel dimension
                    mfccs_norm = np.expand_dims(mfccs_norm, axis=-1)
                
                logger.info(f"Normalized MFCCs shape: {mfccs_norm.shape}")
                
                # Reshape for model input (add batch dimension)
                mfccs_shaped = np.expand_dims(mfccs_norm, axis=0)
                logger.info(f"Final input shape to model: {mfccs_shaped.shape}")
                
                # Get model prediction
                logger.info("Making prediction with model...")
                prediction = self.model.predict(mfccs_shaped)[0]
                
                # Get predicted emotion
                emotion_idx = np.argmax(prediction)
                emotion = self.emotions[emotion_idx]
                confidence = prediction[emotion_idx]
                
                # Calculate sentiment score based on emotion and weights
                sentiment_score = self.emotion_weights.get(emotion, 0) * confidence
                
                # Create result dictionary
                result = {
                    'emotion': emotion,
                    'confidence': float(confidence),
                    'all_emotions': {e: float(prediction[i]) for i, e in enumerate(self.emotions)},
                    'sentiment_score': float(sentiment_score)
                }
                
                # Save the prediction for reference
                self.last_prediction = result
                
                logger.info(f"Prediction result: {result}")
                return result
                
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"Error in audio prediction: {str(e)}", exc_info=True)
            # Re-raise the exception to let the caller handle it
            raise
    
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
            
        """
        Alias for predict_from_video to maintain compatibility with the Gradio interface
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Predicted emotion, confidence, and sentiment score
        """
        return self.predict_from_video(video_path)

        """Alias for predict_from_video to maintain compatibility with the Gradio interface"""
        return self.predict_from_video(video_path)

def analyze_video(self, video_path):
        """Alias for predict_from_video to maintain compatibility with the Gradio interface"""
        return self.predict_from_video(video_path)

def main():
    """Run a simple test of the audio sentiment analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the audio sentiment analyzer")
    parser.add_argument("--model", type=str, default="./audio_sentiment_model_notebook.h5",
                        help="Path to the audio sentiment model")
    parser.add_argument("--norm-params", type=str, default="./audio_norm_params.json",
                        help="Path to the normalization parameters")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to an audio file to analyze (optional)")
    args = parser.parse_args()
    
    # Set up logging to console
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Initialize the analyzer
    logger.info("Initializing audio sentiment analyzer...")
    analyzer = AudioSentimentAnalyzer(
        model_path=args.model,
        norm_params_path=args.norm_params
    )
    
    # Check if model loaded
    if analyzer.model is None:
        logger.error("Model failed to load. Exiting.")
        return 1
    
    # Process audio file if provided
    if args.audio and os.path.exists(args.audio):
        logger.info(f"Analyzing audio file: {args.audio}")
        result = analyzer.predict_from_file(args.audio)
        logger.info(f"Analysis result: {result}")
    else:
        # Try to find some sample audio files
        logger.info("Looking for sample audio files...")
        import glob
        
        # Look in common directories
        potential_dirs = [
            "./examples", 
            "../examples", 
            "./data",
            "../data",
            "./assets",
            "../assets",
            "./samples",
            "../samples"
        ]
        
        # Look for audio files with common extensions
        audio_files = []
        for dir_path in potential_dirs:
            if os.path.exists(dir_path):
                for ext in ["*.wav", "*.mp3", "*.m4a", "*.ogg"]:
                    audio_files.extend(glob.glob(os.path.join(dir_path, ext)))
        
        if audio_files:
            # Use the first audio file found
            sample_audio = audio_files[0]
            logger.info(f"Found sample audio file: {sample_audio}")
            result = analyzer.predict_from_file(sample_audio)
            logger.info(f"Analysis result: {result}")
        else:
            logger.warning("No sample audio files found. Creating a test audio file...")
            
            # Create a simple sine wave audio as a test
            import numpy as np
            from scipy.io import wavfile
            
            # Create a temporary directory
            os.makedirs("./tmp_audio", exist_ok=True)
            
            # Generate simple sine waves for different emotions
            sample_rate = 44100
            duration = 3  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # Happy: higher frequency, amplitude variations
            happy_signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
            happy_signal *= (1 + 0.2 * np.sin(2 * np.pi * 2 * t))  # amplitude modulation
            happy_path = "./tmp_audio/test_happy.wav"
            wavfile.write(happy_path, sample_rate, (happy_signal * 32767).astype(np.int16))
            
            # Sad: lower frequency, slower amplitude
            sad_signal = 0.7 * np.sin(2 * np.pi * 220 * t)
            sad_signal *= (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t))  # slow modulation
            sad_path = "./tmp_audio/test_sad.wav"
            wavfile.write(sad_path, sample_rate, (sad_signal * 32767).astype(np.int16))
            
            # Neutral: mid-range frequency, little variation
            neutral_signal = 0.5 * np.sin(2 * np.pi * 330 * t)
            neutral_path = "./tmp_audio/test_neutral.wav"
            wavfile.write(neutral_path, sample_rate, (neutral_signal * 32767).astype(np.int16))
            
            # Test each generated audio
            for audio_path, emotion in [(happy_path, "happy"), (sad_path, "sad"), (neutral_path, "neutral")]:
                logger.info(f"Testing with generated {emotion} audio: {audio_path}")
                result = analyzer.predict_from_file(audio_path)
                logger.info(f"Analysis result: {result}")
                
                # Check if the score is 0.0 (indicating potential issues)
                if abs(result.get('sentiment_score', 0.0)) < 0.01:
                    logger.warning(f"Zero sentiment score detected for {emotion} audio - model may not be working correctly")
    
    logger.info("Audio sentiment test complete")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
