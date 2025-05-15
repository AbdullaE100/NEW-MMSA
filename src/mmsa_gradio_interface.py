#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
import tempfile
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil
import glob
import logging
import time
import subprocess
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mmsa_gradio.log'))
    ]
)
logger = logging.getLogger('mmsa_gradio')

# Import our custom modules
from mmsa_audio_sentiment import AudioSentimentAnalyzer
from deepface_emotion_detector import DeepFaceEmotionDetector

# Check if transformers is available for text analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not available. Text sentiment analysis will be limited.")
    TRANSFORMERS_AVAILABLE = False

# Check if speech recognition is available for transcription
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    logger.warning("Speech recognition not available. Install with 'pip install SpeechRecognition'")
    SPEECH_RECOGNITION_AVAILABLE = False

# Check if whisper is available for better transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("Whisper not available. Install with 'pip install -U openai-whisper'")
    WHISPER_AVAILABLE = False

class MultimodalSentimentGradio:
    """
    Gradio interface for the Multimodal Sentiment Analysis tool
    """
    
    def __init__(self, 
                audio_model_path=None, 
                audio_norm_params_path=None,
                output_dir=None,
                num_frames=15,
                detector_backend='retinaface',
                cleanup_temp=True,
                allow_fallback=False):
        """
        Initialize the multimodal sentiment analyzer interface
        
        Args:
            audio_model_path (str): Path to audio sentiment model. If None, will use default paths.
            audio_norm_params_path (str): Path to audio normalization parameters.
            output_dir (str): Directory for output files. If None, will use default directory.
            num_frames (int): Number of frames to analyze in videos
            detector_backend (str): Face detector backend for DeepFace
            cleanup_temp (bool): Whether to clean up temporary files automatically
            allow_fallback (bool): Whether to allow audio feature fallback if model fails
        """
        self.cleanup_temp = cleanup_temp
        self.temp_files = []
        self.allow_fallback = allow_fallback
        self.apply_happy_heuristic = True  # Enable happy heuristic by default
        
        # Initialize speech transcription ability
        self.speech_transcriber = True if WHISPER_AVAILABLE or SPEECH_RECOGNITION_AVAILABLE else False
        
        # Initialize score storage attributes
        self.visual_score = 0.0
        self.visual_confidence = 0.0
        self.audio_score = 0.0
        self.audio_confidence = 0.0
        self.text_score = 0.0
        self.text_confidence = 0.0
        self.overall_score = 0.0
        self.overall_confidence = 0.0
        self.last_transcript = ""
        self.last_video_path = None
        
        # Find model files if not provided
        if audio_model_path is None:
            # Search in multiple locations
            search_paths = [
                # Current directory
                ".",
                # Common model directories
                "models",
                "trained_models",
                os.path.expanduser("~/.mmsa/models"),
                # Parent directory
                ".."
            ]
            
            # Try common model file names
            model_names = [
                "audio_sentiment_model_notebook.h5", 
                "audio_sentiment_model.h5",
                "audio_sentiment_model_mmsa.h5"
            ]
            
            # Search for models
            for path in search_paths:
                if not os.path.exists(path):
                    continue
                    
                for name in model_names:
                    potential_path = os.path.join(path, name)
                    if os.path.exists(potential_path):
                        audio_model_path = potential_path
                        logger.info(f"Found audio model: {audio_model_path}")
                        break
                        
                if audio_model_path:
                    break
        
        if audio_model_path and not os.path.exists(audio_model_path):
            logger.warning(f"Specified audio model not found: {audio_model_path}")
            audio_model_path = None
            
            # Try to find any .h5 file that might be an audio model
            h5_files = glob.glob("*.h5")
            if h5_files:
                for h5_file in h5_files:
                    if "audio" in h5_file.lower():
                        audio_model_path = h5_file
                        logger.info(f"Using fallback audio model: {audio_model_path}")
                        break
        
        if audio_norm_params_path is None and audio_model_path:
            # Try to infer norm params path from model path
            base_path = os.path.splitext(audio_model_path)[0]
            potential_norm_params = f"{base_path}_norm_params.json"
            if os.path.exists(potential_norm_params):
                audio_norm_params_path = potential_norm_params
                logger.info(f"Using norm params: {audio_norm_params_path}")
            elif os.path.exists("audio_norm_params.json"):
                audio_norm_params_path = "audio_norm_params.json"
                logger.info(f"Using norm params: {audio_norm_params_path}")
        
        # Set output directory
        self.output_dir = output_dir or "./gradio_results"
        
        # Initialize the audio analyzer
        self.audio_analyzer = AudioSentimentAnalyzer(
            model_path=audio_model_path,
            norm_params_path=audio_norm_params_path
        )
        
        # Initialize the visual analyzer
        self.visual_analyzer = DeepFaceEmotionDetector(
            output_dir=os.path.join(self.output_dir, "visual"),
            num_frames=num_frames
        )
        
        # Set detector backend
        self.visual_analyzer.detector_backend = detector_backend
        logger.info(f"Using face detector backend: {detector_backend}")
        
        # Initialize text sentiment analyzer
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use our custom RoBERTa text sentiment analyzer
                from mmsa_text_sentiment import TextSentimentAnalyzer
                self.text_analyzer = TextSentimentAnalyzer(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest")
                logger.info(f"Text sentiment analysis enabled with model: {self.text_analyzer.model_name}")
            except Exception as e:
                logger.error(f"Error initializing RoBERTa text analyzer: {str(e)}")
                self.text_analyzer = None
        else:
            self.text_analyzer = None
            logger.warning("Transformers not available, text analysis disabled")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "temp"), exist_ok=True)
        
        logger.info("MMSA Gradio interface initialized successfully")
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self.cleanup_temp:
            self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        if not self.temp_files:
            return
            
        logger.info(f"Cleaning up {len(self.temp_files)} temporary files")
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {temp_file}: {str(e)}")
        
        self.temp_files = []
    
    def _apply_happy_heuristic(self, result):
        """
        Apply a heuristic to better detect happy videos
        
        Args:
            result: Analysis result from DeepFaceEmotionDetector
            
        Returns:
            float: Adjusted sentiment score
        """
        if not result or 'emotion_distribution' not in result:
            return result.get('sentiment_score', 0)
        
        emotions = result['emotion_distribution']
        
        # Get original sentiment score
        sentiment_score = result.get('sentiment_score', 0)
        
        # Get happy and surprise scores
        happy_score = emotions.get('happy', 0)
        surprise_score = emotions.get('surprise', 0)
        neutral_score = emotions.get('neutral', 0)
        
        # Get dominant emotion
        dominant_emotion = result.get('dominant_emotion', 'neutral')
        
        # If neutral is dominant but there's significant happy component
        # boost the sentiment score
        if dominant_emotion == 'neutral' and happy_score > 0.1:
            # Boost proportional to happy score
            happy_boost = happy_score * 2.0  # Apply 2.0x multiplier to happy score
            
            # Add the happy boost directly to sentiment score
            adjusted_score = sentiment_score + happy_boost * 0.5  # Scale the boost
            
            logger.info(f"Applied happy boost: +{happy_boost * 0.5:.2f} (happy={happy_score:.2f})")
            
            return min(adjusted_score, 0.9)  # Cap at 0.9
        
        # If happy is dominant, make it strongly positive
        if dominant_emotion == 'happy':
            adjusted_score = max(sentiment_score, 0.5)  # Ensure at least 0.5
            
            logger.info(f"Applied strong happy boost: score {sentiment_score:.2f} -> {adjusted_score:.2f}")
            
            return adjusted_score
        
        # If happy or surprise components are significant, ensure positive sentiment
        if happy_score + surprise_score > 0.3 and sentiment_score < 0.2:
            adjusted_score = max(0.2, sentiment_score + 0.15)
            
            logger.info(f"Applied positive sentiment boost: score {sentiment_score:.2f} -> {adjusted_score:.2f}")
            
            return adjusted_score
        
        return sentiment_score
    
    def analyze_audio(self, audio_input):
        """
        Analyze sentiment from audio input
        
        Args:
            audio_input: Audio file from Gradio
            
        Returns:
            dict: Analysis results
        """
        try:
            # Create a temporary directory for this operation
            temp_dir = os.path.join(self.output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_path = os.path.join(temp_dir, f"temp_audio_{os.getpid()}_{int(time.time())}.wav")
            self.temp_files.append(temp_path)
            
            # If audio_input is a tuple (from Gradio's audio component)
            if isinstance(audio_input, tuple):
                sample_rate, audio_data = audio_input
                import soundfile as sf
                sf.write(temp_path, audio_data, sample_rate)
            else:
                # Assume it's a path
                temp_path = audio_input
            
            # Analyze the audio
            result = self.audio_analyzer.predict_from_file(temp_path)
            
            # Clean up temp file if it was created by us
            if isinstance(audio_input, tuple) and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    self.temp_files.remove(temp_path)
                except:
                    pass
            
            if result:
                return result
            else:
                return {"error": "Failed to analyze audio"}
        
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}", exc_info=True)
            return {"error": f"Error analyzing audio: {str(e)}"}
    
    def transcribe_video(self, video_path):
        """
        Transcribe speech from video to text
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Transcribed text from the video
        """
        try:
            # Extract audio from video
            temp_dir = os.path.join(self.output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            audio_path = os.path.join(temp_dir, f"temp_transcribe_{int(time.time())}.wav")
            self.temp_files.append(audio_path)
            
            # Use ffmpeg to extract audio from video
            logger.info(f"Extracting audio from video for transcription: {video_path}")
            subprocess.run([
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', audio_path, '-y'
            ], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            # Check if audio extraction was successful
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
                logger.warning(f"Audio extraction failed or file too small: {audio_path}")
                return ""
            
            # Try different transcription methods based on available libraries
            transcript = ""
            
            # Method 1: Try using Whisper if available (best quality)
            if WHISPER_AVAILABLE:
                try:
                    logger.info("Transcribing with Whisper model...")
                    model = whisper.load_model("base")
                    result = model.transcribe(audio_path)
                    transcript = result["text"]
                    logger.info(f"Whisper transcription successful: {transcript[:50]}...")
                    return transcript
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {str(e)}")
            
            # Method 2: Try using SpeechRecognition with Google (requires internet)
            if SPEECH_RECOGNITION_AVAILABLE:
                try:
                    logger.info("Transcribing with Google Speech Recognition...")
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(audio_path) as source:
                        audio_data = recognizer.record(source)
                        transcript = recognizer.recognize_google(audio_data)
                        logger.info(f"Google transcription successful: {transcript[:50]}...")
                        return transcript
                except Exception as e:
                    logger.warning(f"Google transcription failed: {str(e)}")
            
            # If no transcription was successful, return empty string
            logger.warning("All transcription methods failed")
            return ""
            
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            return ""
    
    def analyze_video(self, video_input):
        """
        Analyze sentiment from video input
        
        Args:
            video_input: Video file from Gradio
            
        Returns:
            dict: Analysis results
        """
        try:
            # Process video for visual sentiment using multiple backends if needed
            backends = [self.visual_analyzer.detector_backend, 'retinaface', 'mtcnn', 'opencv']
            visual_results = None
            
            for backend in backends:
                try:
                    logger.info(f"Attempting visual analysis with detector: {backend}")
                    visual_results = self.visual_analyzer.process_video(
                        video_input, 
                        detector_backend=backend
                    )
                    
                    # Check for valid results
                    if visual_results and visual_results.get('dominant_emotion_confidence', 0) > 0:
                        logger.info(f"Successful visual analysis with {backend}")
                        
                        # Save the visual scores and emotion information for later use
                        self.visual_score = visual_results.get('sentiment_score', 0)
                        self.visual_confidence = visual_results.get('dominant_emotion_confidence', 0)
                        logger.info(f"Visual sentiment: {visual_results.get('dominant_emotion', 'unknown')} "
                                   f"(Score: {self.visual_score:.2f}, Confidence: {self.visual_confidence:.2f})")
                        
                        # Apply happy heuristic to improve sentiment detection
                        if self.apply_happy_heuristic:
                            original_score = visual_results.get('sentiment_score', 0)
                            adjusted_score = self._apply_happy_heuristic(visual_results)
                            
                            # Update the sentiment score with the adjusted value
                            if adjusted_score != original_score:
                                visual_results['original_sentiment_score'] = original_score
                                visual_results['sentiment_score'] = adjusted_score
                                self.visual_score = adjusted_score  # Update stored score with adjusted value
                                logger.info(f"Applied sentiment enhancement: {original_score:.2f} -> {adjusted_score:.2f}")
                        
                        break
                    else:
                        logger.warning(f"Visual analysis with {backend} produced invalid results")
                except Exception as e:
                    logger.warning(f"Error with {backend} backend: {str(e)}")
            
            # If we still don't have valid visual results, set default values
            if not visual_results or not hasattr(self, 'visual_score'):
                logger.warning("No valid visual results obtained, using defaults")
                self.visual_score = 0.0
                self.visual_confidence = 0.0
                
                # Create minimal visual results to avoid NoneType errors
                visual_results = {
                    "dominant_emotion": "unknown",
                    "dominant_emotion_confidence": 0.0,
                    "emotion_distribution": {"neutral": 1.0},
                    "sentiment_score": 0.0
                }
            
            # Process video for audio sentiment
            audio_results = self.audio_analyzer.predict_from_video(video_input)
            
            # Transcribe speech from video
            logger.info("Transcribing speech from video...")
            transcribed_text = self.transcribe_video(video_input)
            text_results = None
            
            # Only analyze text if we have transcribed something
            if transcribed_text and transcribed_text.strip():
                logger.info(f"Analyzing transcribed text: {transcribed_text[:50]}...")
                text_results = self.analyze_text(transcribed_text)
                if text_results and "error" not in text_results:
                    # Add transcription to results
                    text_results["transcribed"] = True
            else:
                logger.warning("No speech transcribed from video")
            
            # Combine results
            combined_results = {
                "visual": visual_results,
                "audio": audio_results,
                "text": text_results,
                "combined_score": None
            }
            
            # Calculate combined sentiment across all available modalities
            modality_scores = []
            modality_weights = []
            modality_names = []
            
            if visual_results:
                visual_score = visual_results.get("sentiment_score", 0)
                visual_weight = 0.45  # Updated weight for visual
                modality_scores.append(visual_score)
                modality_weights.append(visual_weight)
                modality_names.append("visual")
            
            if audio_results:
                audio_score = audio_results.get("sentiment_score", 0)
                audio_weight = 0.45  # Updated weight for audio
                modality_scores.append(audio_score)
                modality_weights.append(audio_weight)
                modality_names.append("audio")
            
            if text_results:
                text_score = text_results.get("sentiment_score", 0)
                text_weight = 0.10  # Updated weight for text
                modality_scores.append(text_score)
                modality_weights.append(text_weight)
                modality_names.append("text")
            
            # Calculate weighted sentiment if we have scores
            if modality_scores:
                # Normalize weights to sum to 1.0
                total_weight = sum(modality_weights)
                normalized_weights = [w/total_weight for w in modality_weights]
                
                # Calculate weighted score
                combined_score = sum(s * w for s, w in zip(modality_scores, normalized_weights))
                
                # Store results
                combined_results["combined_score"] = combined_score
                combined_results["combined_sentiment"] = self._score_to_sentiment(combined_score)
                combined_results["weights"] = dict(zip(modality_names, normalized_weights))
                combined_results["transcribed_text"] = transcribed_text if text_results else ""
            elif visual_results:
                # Fall back to just visual if no other modalities
                combined_results["combined_score"] = visual_results.get("sentiment_score", 0)
                combined_results["combined_sentiment"] = self._score_to_sentiment(visual_results.get("sentiment_score", 0))
            elif audio_results:
                # Fall back to just audio if no other modalities
                combined_results["combined_score"] = audio_results.get("sentiment_score", 0)
                combined_results["combined_sentiment"] = self._score_to_sentiment(audio_results.get("sentiment_score", 0))
            
            return combined_results
        
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}", exc_info=True)
            return {"error": f"Error analyzing video: {str(e)}"}
    
    def analyze_text(self, text_input):
        """
        Analyze text sentiment using the text sentiment analyzer
        
        Args:
            text_input (str): Text to analyze
            
        Returns:
            dict: Analysis results with sentiment score, classification, and confidence
        """
        try:
            # Use the text sentiment analyzer
            if self.text_analyzer:
                result = self.text_analyzer.analyze(text_input)
                
                # Check for errors
                if 'error' in result:
                    logger.warning(f"Text analysis error: {result['error']}")
                    return {
                        'sentiment_score': 0,
                        'classification': 'neutral',
                        'confidence': 0.5,
                        'error': result['error']
                    }
                
                # Process valid results
                sentiment_score = result.get('sentiment_score', 0)
                confidence = result.get('confidence', 0)
                classification = result.get('sentiment', 'Neutral')
                
                # Store for later use
                self.text_score = sentiment_score
                self.text_confidence = confidence
                self.last_transcript = text_input
                
                # Log results
                logger.info(f"Text sentiment: {classification} (Confidence: {confidence:.2f}, Score: {sentiment_score:.2f})")
                
                # Return formatted results
                return {
                    'sentiment_score': sentiment_score,
                    'classification': classification.lower(),
                    'confidence': confidence
                }
            else:
                logger.warning("No text analyzer available")
                return {
                    'sentiment_score': 0,
                    'classification': 'neutral',
                    'confidence': 0.5,
                    'error': 'Text analysis not available'
                }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
            return {
                'sentiment_score': 0,
                'classification': 'neutral',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def _score_to_sentiment(self, score):
        """Convert numerical score to sentiment category"""
        if score >= 0.2:
            return "Positive"
        elif score <= -0.2:
            return "Negative"
        else:
            return "Neutral"
    
    def _get_emoticon(self, sentiment):
        """Get emoticon for sentiment category - no longer used in the redesigned interface"""
        return ""
    
    def process_multimodal(self, video_input, audio_input=None, text_input=None):
        """
        Process multimodal inputs (video, audio, text) for sentiment analysis
        
        Args:
            video_input: Video file from Gradio
            audio_input: Audio file from Gradio (optional)
            text_input: Text input from Gradio (optional)
            
        Returns:
            tuple: (result_html, sentiment_score, chart_output)
        """
        logger.info(f"Processing inputs: Video={video_input is not None}, Audio={audio_input is not None}, Text={text_input is not None}")
        
        # Initialize results
        visual_results = None
        audio_results = None
        text_results = None
        transcript = None
        final_score = 0
        final_confidence = 0
        
        # Track start time
        start_time = time.time()
        
        # If video is provided, extract visual, audio and transcript
        if video_input is not None:
            # 1. Process video for visual sentiment
            logger.info("Processing video input...")
            visual_results = self.analyze_video(video_input)
            
            # Store video path
            self.last_video_path = video_input if isinstance(video_input, str) else (
                video_input.get('path', None) if isinstance(video_input, dict) else None
            )
            
            # 2. Extract audio for audio sentiment if no separate audio provided
            if audio_input is None and self.audio_analyzer is not None:
                # Get the actual path
                video_path = self._get_path_from_input(video_input)
                
                if video_path:
                    try:
                        audio_results = self.audio_analyzer.predict_from_video(video_path)
                        if audio_results:
                            # Store audio scores
                            self.audio_score = audio_results.get('sentiment_score', 0)
                            self.audio_confidence = audio_results.get('confidence', 0)
                            logger.info(f"Audio sentiment: {audio_results.get('emotion', 'unknown')} "
                                      f"(Score: {self.audio_score:.2f}, Confidence: {self.audio_confidence:.2f})")
                        else:
                            logger.warning("No audio results returned from analyzer")
                    except Exception as e:
                        logger.error(f"Error processing audio from video: {str(e)}")
            
            # 3. Transcribe speech if no separate text provided
            if text_input is None and self.speech_transcriber:
                logger.info("Transcribing speech from video...")
                transcript = self._transcribe_video(video_input)
                
                if transcript:
                    # Store transcript
                    self.last_transcript = transcript
                    
                    # Analyze transcript for sentiment
                    if self.text_analyzer:
                        logger.info(f"Analyzing transcribed text: {transcript}")
                        text_results = self.analyze_text(transcript)
                        # Store text scores (already stored in analyze_text)
                
        # If separate audio input provided
        if audio_input is not None and self.audio_analyzer is not None:
            try:
                audio_results = self.audio_analyzer.predict_from_file(audio_input)
                if audio_results:
                    # Store audio scores
                    self.audio_score = audio_results.get('sentiment_score', 0)
                    self.audio_confidence = audio_results.get('confidence', 0)
                    logger.info(f"Audio sentiment: {audio_results.get('emotion', 'unknown')} "
                              f"(Score: {self.audio_score:.2f}, Confidence: {self.audio_confidence:.2f})")
            except Exception as e:
                logger.error(f"Error processing audio input: {str(e)}")
        
        # If separate text input provided
        if text_input is not None and self.text_analyzer is not None:
            text_results = self.analyze_text(text_input)
            # Text scores are stored in analyze_text
        
        # Calculate combined sentiment score with modality weights
        if visual_results or audio_results or text_results:
            # Create array of available modality scores and confidences
            scores = []
            confidences = []
            modalities = []
            
            # Visual contribution
            if visual_results and hasattr(self, 'visual_score'):
                scores.append(self.visual_score)
                confidences.append(self.visual_confidence)
                modalities.append('visual')
            
            # Audio contribution
            if audio_results and hasattr(self, 'audio_score'):
                scores.append(self.audio_score)
                confidences.append(self.audio_confidence)
                modalities.append('audio')
            
            # Text contribution
            if text_results and hasattr(self, 'text_score'):
                scores.append(self.text_score)
                confidences.append(self.text_confidence)
                modalities.append('text')
            
            # Ensure we have at least one valid score
            if len(scores) > 0:
                # For confidence weighted average, normalize confidences to sum to 1
                confidence_sum = sum(confidences)
                
                if confidence_sum > 0:
                    # Weighted average of scores by confidence
                    final_score = sum(s * c for s, c in zip(scores, confidences)) / confidence_sum
                    final_confidence = confidence_sum / len(confidences)  # Average confidence
                else:
                    # Simple average if confidences are all zero
                    final_score = sum(scores) / len(scores)
                    final_confidence = 0.5  # Default confidence
            else:
                final_score = 0
                final_confidence = 0
        
        # Final score should be between -1 and 1
        final_score = max(-1.0, min(1.0, final_score))
        
        # Record processing time
        processing_time = time.time() - start_time
        logger.info(f"Multimodal processing completed in {processing_time:.2f} seconds")
        
        # Clean up any temporary files
        temp_files = []
        if hasattr(self, 'temp_files'):
            temp_files = self.temp_files
            
        if temp_files:
            logger.info(f"Cleaning up {len(temp_files)} temporary files")
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Error cleaning up temp file {temp_file}: {str(e)}")
            self.temp_files = []
        
        # Generate result HTML, sentiment score, and chart
        result_html = self._generate_result_html(visual_results, audio_results, text_results, transcript)
        chart_output = self._generate_chart(scores, modalities, confidences, final_score)
        
        # Store the final score and confidence
        self.overall_score = final_score
        self.overall_confidence = final_confidence
        
        # Return the HTML, score, and chart
        return result_html, final_score, chart_output

    def _generate_result_html(self, visual_results, audio_results, text_results, transcript=None):
        """
        Generate HTML for sentiment analysis results
        
        Args:
            visual_results: Visual sentiment analysis results
            audio_results: Audio sentiment analysis results
            text_results: Text sentiment analysis results
            transcript: Speech transcription (optional)
            
        Returns:
            str: HTML for displaying results
        """
        html_parts = []
        
        # Function to create a nice result table
        def create_result_table(title, rows):
            table_html = f"""
            <div style="margin-bottom: 20px;">
                <h2>{title}</h2>
                <div style="height: 2px; background: linear-gradient(90deg, #3a7bd5, #00d2ff); margin-bottom: 15px;"></div>
                <table style="width: 100%; border-collapse: collapse;">
            """
            
            for label, value in rows:
                table_html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold; width: 50%;">{label}:</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{value}</td>
                    </tr>
                """
            
            table_html += """
                </table>
            </div>
            """
            return table_html
        
        # Visual results
        if visual_results:
            try:
                # Format the dominant expression with capital first letter
                dominant_expression = visual_results.get('dominant_emotion', 'unknown')
                if dominant_expression:
                    dominant_expression = dominant_expression[0].upper() + dominant_expression[1:]
                
                rows = [
                    ("Sentiment Score", f"{self.visual_score:.2f}"),
                    ("Dominant Expression", dominant_expression),
                    ("Confidence", f"{self.visual_confidence:.2f}")
                ]
                html_parts.append(create_result_table("Visual Analysis", rows))
            except Exception as e:
                logger.error(f"Error formatting visual results: {str(e)}")
                html_parts.append(f"<p>Error formatting visual results: {str(e)}</p>")
        
        # Audio results
        if audio_results:
            try:
                rows = [
                    ("Sentiment Score", f"{self.audio_score:.2f}"),
                    ("Detected Emotion", audio_results.get('emotion', 'unknown')),
                    ("Confidence", f"{self.audio_confidence:.2f}")
                ]
                html_parts.append(create_result_table("Audio Analysis", rows))
            except Exception as e:
                logger.error(f"Error formatting audio results: {str(e)}")
                html_parts.append(f"<p>Error formatting audio results: {str(e)}</p>")
        else:
            # Check if we have audio score and confidence attributes, but audio_results failed
            if hasattr(self, 'audio_score') and hasattr(self, 'audio_confidence'):
                try:
                    # Use fallback values when audio_results is not available but we have scores
                    rows = [
                        ("Sentiment Score", f"{self.audio_score:.2f}"),
                        ("Detected Emotion", "unknown"),
                        ("Confidence", f"{self.audio_confidence:.2f}")
                    ]
                    html_parts.append(create_result_table("Audio Analysis", rows))
                except Exception as e:
                    logger.error(f"Error formatting fallback audio results: {str(e)}")
        
        # Text results
        if text_results:
            try:
                # Check if text_results has the expected structure
                if isinstance(text_results, dict) and ('classification' in text_results or 'sentiment' in text_results):
                    # Extract values with appropriate fallbacks
                    classification = text_results.get('classification', text_results.get('sentiment', 'neutral'))
                    if isinstance(classification, str):
                        classification = classification.capitalize()
                    
                    sentiment_score = text_results.get('sentiment_score', 0)
                    confidence = text_results.get('confidence', 0.5)
                    
                    rows = [
                        ("Sentiment Score", f"{sentiment_score:.2f}"),
                        ("Classification", classification),
                        ("Confidence", f"{confidence:.2f}")
                    ]
                    html_parts.append(create_result_table("Text Analysis", rows))
                else:
                    # Fallback when text_results structure is unexpected
                    logger.warning(f"Unexpected text_results structure: {text_results}")
                    # Use stored values if available
                    if hasattr(self, 'text_score') and hasattr(self, 'text_confidence'):
                        sentiment = self._score_to_sentiment(self.text_score).capitalize()
                        rows = [
                            ("Sentiment Score", f"{self.text_score:.2f}"),
                            ("Classification", sentiment),
                            ("Confidence", f"{self.text_confidence:.2f}")
                        ]
                        html_parts.append(create_result_table("Text Analysis", rows))
            except Exception as e:
                logger.error(f"Error formatting text results: {str(e)}")
                html_parts.append(f"<p>Error formatting text results: {str(e)}</p>")
        
        # Add transcript if available
        if transcript:
            html_parts.append(f"""
            <div style="margin-bottom: 20px;">
                <h2>Speech Transcript</h2>
                <div style="height: 2px; background: linear-gradient(90deg, #3a7bd5, #00d2ff); margin-bottom: 15px;"></div>
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; font-style: italic;">"{transcript}"</div>
            </div>
            """)
        
        # Join all parts
        return "".join(html_parts)

    def _get_path_from_input(self, input_data):
        """
        Extract a file path from different types of input data
        
        Args:
            input_data: Input data (path string, file object, gradio file input)
            
        Returns:
            str: File path or None if not found
        """
        try:
            # Handle string paths
            if isinstance(input_data, str):
                return input_data
                
            # Handle dict-like objects (like gradio inputs)
            if isinstance(input_data, dict) and 'path' in input_data:
                return input_data['path']
                
            # Handle file-like objects
            if hasattr(input_data, 'name'):
                return input_data.name
                
            # Handle tuple from Gradio components
            if isinstance(input_data, tuple) and len(input_data) == 2:
                # Could be (sample_rate, audio_data) from Gradio Audio
                return None  # No path for raw audio data
                
            logger.warning(f"Couldn't extract path from input: {type(input_data)}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting path from input: {str(e)}")
            return None

    def _transcribe_video(self, video_path):
        """
        Wrapper method for transcribe_video to maintain consistency with other methods
        
        Args:
            video_path: Path to video file
            
        Returns:
            str: Transcribed text
        """
        return self.transcribe_video(video_path)

    def _generate_chart(self, scores, modalities, confidences, final_score):
        """
        Generate a chart showing sentiment scores across modalities
        
        Args:
            scores: List of sentiment scores for each modality
            modalities: List of modality names
            confidences: List of confidence values
            final_score: Final combined sentiment score
            
        Returns:
            matplotlib.Figure: Chart figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # If we don't have any scores, return empty chart
            if not scores or not modalities:
                ax.text(0.5, 0.5, "No sentiment data available", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_axis_off()
                return fig
            
            # Create x-positions for bars
            x_pos = np.arange(len(modalities))
            
            # Define colors based on sentiment (red for negative, green for positive)
            colors = ['#ff6b6b' if s < 0 else '#4ecdc4' for s in scores]
            
            # Create bars
            bars = ax.bar(x_pos, scores, align='center', alpha=0.7, color=colors, width=0.5)
            
            # Add confidence as text on bars
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                height = bar.get_height()
                if height < 0:
                    # For negative bars, show confidence above the x-axis
                    y_pos = 0.05
                else:
                    # For positive bars, show confidence above the bar
                    y_pos = height + 0.05
                
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                        f'{conf:.2f}', ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add final score as a star marker
            ax.scatter(len(modalities), final_score, marker='*', s=200, 
                      color='#f9c80e', edgecolor='black', zorder=10,
                      label='Overall Score')
            
            # Set chart properties
            ax.set_ylabel('Sentiment Score')
            ax.set_title('Sentiment Analysis by Modality')
            ax.set_xticks(list(x_pos) + [len(modalities)])
            ax.set_xticklabels(modalities + ['Combined'])
            ax.set_ylim(-1.1, 1.1)
            
            # Add grid for better readability
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # Add sentiment regions
            ax.axhspan(0.2, 1.1, facecolor='#4ecdc4', alpha=0.1, label='Positive')
            ax.axhspan(-0.2, 0.2, facecolor='#f9c80e', alpha=0.1, label='Neutral')
            ax.axhspan(-1.1, -0.2, facecolor='#ff6b6b', alpha=0.1, label='Negative')
            
            # Add legend
            ax.legend(loc='best')
            
            # Tight layout
            plt.tight_layout()
            
            return fig
        
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}", exc_info=True)
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, f"Error generating chart: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=10, wrap=True)
            ax.set_axis_off()
            return fig

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch the MMSA Gradio interface")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--detector", type=str, default="retinaface", 
                       help="Face detector backend (opencv, retinaface, mtcnn, ssd)")
    parser.add_argument("--audio-model", type=str, default=None,
                       help="Path to audio sentiment model")
    parser.add_argument("--norm-params", type=str, default=None,
                       help="Path to audio normalization parameters")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory for output files")
    parser.add_argument("--num-frames", type=int, default=15,
                       help="Number of frames to analyze in videos")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Disable automatic cleanup of temporary files")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--allow-fallback", action="store_true",
                       help="Allow direct audio feature analysis if model fails")
    parser.add_argument("--no-happy-heuristic", action="store_true",
                       help="Disable the happy heuristic for visual sentiment analysis")
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize the interface with custom parameters
    mmsa_gradio = MultimodalSentimentGradio(
        audio_model_path=args.audio_model,
        audio_norm_params_path=args.norm_params,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        detector_backend=args.detector,
        cleanup_temp=not args.no_cleanup,
        allow_fallback=args.allow_fallback
    )
    
    # Always enable happy heuristic
    mmsa_gradio.apply_happy_heuristic = True
    
    # Create a wrapper function that only takes video input
    def process_video_only(video_input):
        return mmsa_gradio.process_multimodal(video_input, None, None)
    
    logger.info("Launching Gradio interface...")
    with gr.Blocks(title="Multimodal Sentiment Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Multimodal Sentiment Analysis
            
            Analyze sentiment by automatically combining different modalities: visual, audio, and text from video.
            Upload a video file (MP4) to perform complete multimodal analysis.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video (MP4)")
                
                # Highlight multimodal approach
                gr.Markdown(
                    """
                    ### Automatic Multimodal Analysis
                    The system will automatically:
                    1. Analyze facial expressions for visual sentiment
                    2. Extract and analyze audio for speech emotion
                    3. Transcribe speech and analyze the text sentiment
                    
                    All three modalities will be combined into a unified sentiment score.
                    """
                )
                
                analyze_button = gr.Button("Analyze Sentiment", variant="primary")
            
            with gr.Column(scale=1):
                result_html = gr.HTML(label="Results")
                sentiment_score = gr.Number(label="Sentiment Score (-1 to +1)")
                chart_output = gr.Plot(label="Sentiment by Modality")
                
                # Note explaining what the tool does in more academic terms
                gr.Markdown(
                    """
                    ### Model Information
                    
                    This multimodal sentiment analysis system combines:
                    - Visual analysis using facial expression recognition
                    - Audio analysis using speech emotion recognition
                    - Text analysis using transcribed speech
                    
                    The system employs a weighted integration approach for multimodal fusion,
                    with contextual adaptation based on detected sentiment patterns.
                    """
                )
        
        analyze_button.click(
            fn=process_video_only,
            inputs=[video_input],
            outputs=[result_html, sentiment_score, chart_output]
        )
        
        gr.Markdown(
            """
            ## Methodology
            
            This system utilizes:
            
            - **Visual Analysis**: DeepFace for facial expression recognition
            - **Audio Analysis**: CNN model trained on the RAVDESS emotional speech dataset
            - **Text Analysis**: RoBERTa-based model with context enrichment for speech transcription
            - **Speech Recognition**: OpenAI Whisper model with Google Speech Recognition fallback
            
            The final sentiment score integrates all available modalities into a unified score 
            from -1 (strongly negative) to +1 (strongly positive), with dynamic weighting
            between modalities (Visual: 45%, Audio: 45%, Text: 10%).
            """
        )
    
    # Launch the interface
    demo.launch(share=args.share)

if __name__ == "__main__":
    main() 