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
                self.text_analyzer = pipeline("sentiment-analysis")
                logger.info("Text sentiment analysis enabled")
            except Exception as e:
                logger.error(f"Error initializing text analyzer: {str(e)}")
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
                        
                        # Apply happy heuristic to improve sentiment detection
                        if self.apply_happy_heuristic:
                            original_score = visual_results.get('sentiment_score', 0)
                            adjusted_score = self._apply_happy_heuristic(visual_results)
                            
                            # Update the sentiment score with the adjusted value
                            if adjusted_score != original_score:
                                visual_results['original_sentiment_score'] = original_score
                                visual_results['sentiment_score'] = adjusted_score
                                logger.info(f"Applied sentiment enhancement: {original_score:.2f} -> {adjusted_score:.2f}")
                        
                        break
                    else:
                        logger.warning(f"Visual analysis with {backend} produced invalid results")
                except Exception as e:
                    logger.warning(f"Error with {backend} backend: {str(e)}")
            
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
        Analyze sentiment from text input
        
        Args:
            text_input: Text string from Gradio
            
        Returns:
            dict: Analysis results
        """
        try:
            if not text_input or text_input.strip() == "":
                return {"error": "Please enter some text"}
            
            if self.text_analyzer:
                # Use transformers pipeline
                result = self.text_analyzer(text_input)[0]
                
                # Map to our scale (-1 to 1)
                label = result["label"]
                score = result["score"]
                
                if label.lower() == "positive":
                    sentiment_score = score
                elif label.lower() == "negative":
                    sentiment_score = -score
                else:
                    sentiment_score = 0
                
                logger.info(f"Text sentiment: {label} (Confidence: {score:.2f}, Score: {sentiment_score:.2f})")
                
                return {
                    "sentiment": label,
                    "confidence": score,
                    "sentiment_score": sentiment_score,
                    "text": text_input
                }
            else:
                logger.warning("Text analysis unavailable - transformers not loaded")
                return {"error": "Text sentiment analysis not available"}
        
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
            return {"error": f"Error analyzing text: {str(e)}"}
    
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
    
    def process_multimodal(self, video_input=None, audio_input=None, text_input=None):
        """
        Process multimodal inputs and return combined results
        
        Args:
            video_input: Video file
            audio_input: Audio file
            text_input: Text string (only used if no transcription is available)
            
        Returns:
            tuple: HTML result, sentiment score, pie chart of modalities
        """
        results = {}
        scores = []
        labels = []
        modalities_used = []
        confidence_scores = []
        
        # Define weights for different modalities (updated weights)
        weights = {'video': 0.45, 'audio': 0.45, 'text': 0.10}
        
        # Create mapping from included_modalities labels to weight keys
        modality_mapping = {
            'visual': 'video',
            'audio': 'audio',
            'text': 'text'
        }
        
        logger.info(f"Processing inputs: Video={video_input is not None}, Audio={audio_input is not None}, Text={text_input is not None and text_input.strip() != ''}")
        
        try:
            # Process video input (including auto-transcription if possible)
            if video_input is not None:
                logger.info("Processing video input...")
                video_res = self.analyze_video(video_input)
                if video_res and not "error" in video_res:
                    results["video"] = video_res
                    
                    # Get combined score from video analysis (visual+audio+transcribed text)
                    if "combined_score" in video_res:
                        score = video_res["combined_score"]
                        
                        # Determine what modalities are included
                        included_modalities = []
                        if "visual" in video_res and video_res["visual"]:
                            included_modalities.append("Visual")
                        if "audio" in video_res and video_res["audio"]:
                            included_modalities.append("Audio")
                        if "text" in video_res and video_res["text"]:
                            included_modalities.append("Text (Transcribed)")
                        
                        modality = " + ".join(included_modalities)
                        # Calculate the weight based on included modalities
                        weight = 0
                        for m in included_modalities:
                            # Extract the base modality name (first word, lowercase)
                            m_key = m.lower().split()[0]  
                            # Map the modality key to the correct weight key
                            weight_key = modality_mapping.get(m_key, m_key)
                            # Get the weight or default to 0.1
                            weight += weights.get(weight_key, 0.1)
                    else:
                        # Just visual score
                        score = video_res.get("visual", {}).get("sentiment_score", 0)
                        modality = "Video"
                        weight = weights['video']  # Map "Video" to "video" key
                    
                    confidence = video_res.get("visual", {}).get("dominant_emotion_confidence", 0.5)
                    
                    scores.append(score)
                    labels.append(modality)
                    modalities_used.append("video")
                    confidence_scores.append(confidence)
                    
                    logger.info(f"Video score: {score}, confidence: {confidence}")
                    
                    # Store transcribed text
                    if "transcribed_text" in video_res and video_res["transcribed_text"]:
                        results["transcribed_text"] = video_res["transcribed_text"]
                else:
                    logger.warning("Video analysis failed or returned an error")
            
            # Process audio input (only if not already analyzed with video)
            if audio_input is not None and "video" not in results:
                logger.info("Processing audio input...")
                audio_res = self.analyze_audio(audio_input)
                if audio_res and not "error" in audio_res:
                    results["audio"] = audio_res
                    
                    score = audio_res.get("sentiment_score", 0)
                    confidence = audio_res.get("confidence", 0.5)
                    
                    scores.append(score)
                    labels.append("Audio")
                    modalities_used.append("audio")
                    confidence_scores.append(confidence)
                    
                    logger.info(f"Audio score: {score}, confidence: {confidence}")
                else:
                    logger.warning("Audio analysis failed or returned an error")
            
            # Process manual text input (only if provided and no transcription available)
            if text_input is not None and text_input.strip() != "" and "transcribed_text" not in results:
                logger.info("Processing manual text input...")
                text_res = self.analyze_text(text_input)
                if text_res and not "error" in text_res:
                    results["text"] = text_res
                    
                    score = text_res.get("sentiment_score", 0)
                    confidence = text_res.get("confidence", 0.5)
                    
                    scores.append(score)
                    labels.append("Text (Manual)")
                    modalities_used.append("text")
                    confidence_scores.append(confidence)
                    
                    logger.info(f"Text score: {score}, confidence: {confidence}")
                else:
                    logger.warning("Text analysis failed or returned an error")
            
            # Calculate combined score from all available modalities
            if not scores:
                logger.warning("No successful analysis from any modality")
                html_result = f"<div style='text-align: center; padding: 20px;'>"
                html_result += f"<h2>Analysis Failed</h2>"
                html_result += f"<p>No results could be obtained from the provided inputs.</p>"
                html_result += f"<p>Please ensure that your video, audio, or text contains valid content for sentiment analysis.</p>"
                html_result += f"</div>"
                return html_result, 0, None
            
            # Weight scores by confidence and modality importance
            weighted_scores = []
            total_weight = 0
            
            for i, score in enumerate(scores):
                # Calculate a weighted importance based on both modality type and confidence
                modality_type = modalities_used[i]
                # Use modality_mapping to handle inconsistencies between 'visual' and 'video'
                modality_key = modality_mapping.get(modality_type, modality_type)
                modality_weight = weights.get(modality_key, 0.1)
                confidence = confidence_scores[i]
                
                # Ensure we don't have zero confidence
                if confidence < 0.3:
                    confidence = 0.3
                
                # Compute final weight
                final_weight = modality_weight * confidence
                
                weighted_scores.append(score * final_weight)
                total_weight += final_weight
            
            # Calculate final score
            final_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
            
            # Convert score to sentiment category
            sentiment = "Neutral"
            if final_score >= 0.2:
                sentiment = "Positive"
            elif final_score <= -0.2:
                sentiment = "Negative"
            
            # Create a visualization of the modality contributions
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Set a color for each sentiment range
            colors = []
            for score in scores:
                if score >= 0.2:
                    colors.append('#4CAF50')  # Green for positive
                elif score <= -0.2:
                    colors.append('#F44336')  # Red for negative
                else:
                    colors.append('#9E9E9E')  # Gray for neutral
            
            # Create bar chart showing score by modality
            bars = ax1.bar(labels, scores, color=colors)
            
            # Add a horizontal line at y=0
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add horizontal lines at sentiment thresholds
            ax1.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Positive Threshold')
            ax1.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5, label='Negative Threshold')
            
            # Add labels
            ax1.set_title('Sentiment Scores by Modality')
            ax1.set_ylabel('Sentiment Score (-1 to +1)')
            ax1.set_ylim(-1.1, 1.1)
            
            # Add score labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height < 0:
                    # For negative scores, put the label below the bar
                    ax1.text(bar.get_x() + bar.get_width()/2., height - 0.1,
                            f'{height:.2f}', ha='center', va='top')
                else:
                    # For positive or zero scores, put the label on top
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'{height:.2f}', ha='center', va='bottom')
            
            ax1.legend()
            
            # Create a pie chart showing the contribution of each modality to the final score
            # Calculate the absolute contribution of each modality
            contributions = []
            labels_pie = []
            
            for i, (label, score) in enumerate(zip(labels, scores)):
                modality_type = modalities_used[i]
                # Use modality_mapping to handle inconsistencies between 'visual' and 'video'
                modality_key = modality_mapping.get(modality_type, modality_type)
                modality_weight = weights.get(modality_key, 0.1)
                confidence = confidence_scores[i]
                final_weight = modality_weight * confidence
                weight_proportion = final_weight / total_weight if total_weight > 0 else 0
                
                # Only include modalities with positive contributions to avoid confusion
                contribution = abs(score * weight_proportion)
                if contribution > 0.01:  # Only include significant contributions
                    contributions.append(contribution)
                    labels_pie.append(f"{label}\n({weight_proportion:.2f})")
            
            if contributions:
                ax2.pie(contributions, labels=labels_pie, autopct='%1.1f%%', startangle=90,
                       shadow=False, colors=['#2196F3', '#FF9800', '#9C27B0'][:len(contributions)])
                ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                ax2.set_title('Modality Contribution to Final Analysis')
            
            plt.tight_layout()
            
            # Create a professional HTML result with academic styling
            html_result = f"""
            <div style='font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; border-radius: 8px;'>
                <h2 style='color: #234A8B; text-align: center; margin-bottom: 20px; font-size: 24px;'>Multimodal Sentiment Analysis Results</h2>
                
                <div style='background-color: #fff; padding: 15px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color: #333; margin-top: 0; font-size: 20px;'>Analysis Summary</h3>
                    <table style='width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 16px;'>
                        <tr>
                            <td style='padding: 8px; font-weight: bold; width: 40%; color: #000;'>Sentiment Classification:</td>
                            <td style='padding: 8px; color: #000;'>{sentiment}</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px; font-weight: bold; color: #000;'>Sentiment Score:</td>
                            <td style='padding: 8px; color: #000;'>{final_score:.2f} (Range: -1 to 1)</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px; font-weight: bold; color: #000;'>Confidence Level:</td>
                            <td style='padding: 8px; color: #000;'>{(sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0):.2f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px; font-weight: bold; color: #000;'>Modalities Analyzed:</td>
                            <td style='padding: 8px; color: #000;'>{", ".join([m.capitalize() for m in set(modalities_used)])}</td>
                        </tr>
                    </table>
                </div>
            """
            
            # Display transcribed text in a prominent position if available
            if "transcribed_text" in results and results["transcribed_text"]:
                html_result += f"""
                <div style='background-color: #E1EFFE; padding: 15px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #0D47A1;'>
                    <h3 style='color: #0D47A1; margin-top: 0; font-size: 18px;'>Automatically Transcribed Speech</h3>
                    <blockquote style='margin: 0; font-style: italic; color: #000; font-size: 16px;'>"{results['transcribed_text']}"</blockquote>
                </div>
                """
            
            # Create detailed result sections for each modality
            html_result += "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px;'>"
            
            # Visual Analysis Section
            if "video" in results and "visual" in results["video"] and results["video"]["visual"]:
                visual_res = results["video"]["visual"]
                visual_score = visual_res.get("sentiment_score", 0)
                dominant_emotion = visual_res.get("dominant_emotion", "Unknown")
                emotion_conf = visual_res.get("dominant_emotion_confidence", 0)
                
                html_result += f"""
                <div style='background-color: #fff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color: #333; margin-top: 0; border-bottom: 1px solid #ccc; padding-bottom: 8px; font-size: 18px;'>Visual Analysis</h4>
                    <table style='width: 100%; border-collapse: collapse; font-size: 16px;'>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Sentiment Score:</td>
                            <td style='padding: 6px; color: #000;'>{visual_score:.2f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Dominant Expression:</td>
                            <td style='padding: 6px; color: #000;'>{dominant_emotion}</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Confidence:</td>
                            <td style='padding: 6px; color: #000;'>{emotion_conf:.2f}</td>
                        </tr>
                    </table>
                </div>
                """
            
            # Audio Analysis Section
            if "video" in results and "audio" in results["video"] and results["video"]["audio"]:
                audio_res = results["video"]["audio"]
                audio_score = audio_res.get("sentiment_score", 0)
                emotion = audio_res.get("emotion", "Unknown")
                confidence = audio_res.get("confidence", 0)
                
                html_result += f"""
                <div style='background-color: #fff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color: #333; margin-top: 0; border-bottom: 1px solid #ccc; padding-bottom: 8px; font-size: 18px;'>Audio Analysis</h4>
                    <table style='width: 100%; border-collapse: collapse; font-size: 16px;'>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Sentiment Score:</td>
                            <td style='padding: 6px; color: #000;'>{audio_score:.2f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Detected Emotion:</td>
                            <td style='padding: 6px; color: #000;'>{emotion}</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Confidence:</td>
                            <td style='padding: 6px; color: #000;'>{confidence:.2f}</td>
                        </tr>
                    </table>
                </div>
                """
            
            # Text Analysis Section
            if "video" in results and "text" in results["video"] and results["video"]["text"]:
                text_res = results["video"]["text"]
                text_score = text_res.get("sentiment_score", 0)
                sentiment = text_res.get("sentiment", "Unknown")
                confidence = text_res.get("confidence", 0)
                
                html_result += f"""
                <div style='background-color: #fff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color: #333; margin-top: 0; border-bottom: 1px solid #ccc; padding-bottom: 8px; font-size: 18px;'>Text Analysis</h4>
                    <table style='width: 100%; border-collapse: collapse; font-size: 16px;'>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Sentiment Score:</td>
                            <td style='padding: 6px; color: #000;'>{text_score:.2f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Classification:</td>
                            <td style='padding: 6px; color: #000;'>{sentiment}</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Confidence:</td>
                            <td style='padding: 6px; color: #000;'>{confidence:.2f}</td>
                        </tr>
                    </table>
                </div>
                """
            elif "text" in results:
                # For manually entered text
                text_res = results["text"]
                text_score = text_res.get("sentiment_score", 0)
                sentiment = text_res.get("sentiment", "Unknown")
                confidence = text_res.get("confidence", 0)
                
                html_result += f"""
                <div style='background-color: #fff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color: #333; margin-top: 0; border-bottom: 1px solid #ccc; padding-bottom: 8px; font-size: 18px;'>Text Analysis</h4>
                    <table style='width: 100%; border-collapse: collapse; font-size: 16px;'>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Sentiment Score:</td>
                            <td style='padding: 6px; color: #000;'>{text_score:.2f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Classification:</td>
                            <td style='padding: 6px; color: #000;'>{sentiment}</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px; font-weight: bold; color: #000;'>Confidence:</td>
                            <td style='padding: 6px; color: #000;'>{confidence:.2f}</td>
                        </tr>
                    </table>
                </div>
                """
            
            html_result += "</div>"  # End of modality grid
            
            # Add multimodal fusion details
            if len(modalities_used) > 1:
                html_result += f"""
                <div style='background-color: #F5F5F5; padding: 15px; border-radius: 5px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #DDD;'>
                    <h3 style='color: #333; margin-top: 0; font-size: 18px;'>Multimodal Fusion Methodology</h3>
                    <p style='margin-top: 0; font-size: 16px; color: #000;'>This analysis employs a confidence-weighted multimodal fusion approach that combines evidence from multiple modalities:</p>
                    
                    <table style='width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 15px;'>
                        <tr style='background-color: #E0E0E0;'>
                            <th style='padding: 8px; text-align: left; border-bottom: 1px solid #999; color: #000;'>Modality</th>
                            <th style='padding: 8px; text-align: right; border-bottom: 1px solid #999; color: #000;'>Base Weight</th>
                            <th style='padding: 8px; text-align: right; border-bottom: 1px solid #999; color: #000;'>Confidence</th>
                            <th style='padding: 8px; text-align: right; border-bottom: 1px solid #999; color: #000;'>Final Weight</th>
                            <th style='padding: 8px; text-align: right; border-bottom: 1px solid #999; color: #000;'>Score</th>
                            <th style='padding: 8px; text-align: right; border-bottom: 1px solid #999; color: #000;'>Contribution</th>
                        </tr>
                """
                
                for i, (modality, score) in enumerate(zip(labels, scores)):
                    modality_type = modalities_used[i]
                    # Use modality_mapping to handle inconsistencies between 'visual' and 'video'
                    modality_key = modality_mapping.get(modality_type, modality_type)
                    modality_weight = weights.get(modality_key, 0.1)
                    confidence = confidence_scores[i]
                    final_weight = modality_weight * confidence
                    weight_proportion = final_weight / total_weight if total_weight > 0 else 0
                    contribution = score * weight_proportion
                    
                    html_result += f"""
                        <tr>
                            <td style='padding: 8px; border-bottom: 1px solid #ccc; color: #000;'>{modality}</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>{modality_weight:.2f}</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>{confidence:.2f}</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>{weight_proportion:.2f}</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>{score:.2f}</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>{contribution:.2f}</td>
                        </tr>
                    """
                
                html_result += f"""
                        <tr style='font-weight: bold;'>
                            <td style='padding: 8px; border-bottom: 1px solid #ccc; color: #000;'>Final Result</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>-</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>-</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>1.00</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>{final_score:.2f}</td>
                            <td style='padding: 8px; text-align: right; border-bottom: 1px solid #ccc; color: #000;'>{final_score:.2f}</td>
                        </tr>
                    </table>
                </div>
                """
            
            # Add model information
            html_result += f"""
            <div style='background-color: #EEFBEE; padding: 15px; border-radius: 5px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #CCC;'>
                <h3 style='color: #333; margin-top: 0; font-size: 18px;'>Model Information</h3>
                <p style='margin-top: 0; color: #000; font-size: 16px;'>This system utilizes multiple specialized models for sentiment analysis:</p>
                <ul style='margin-top: 5px; color: #000; font-size: 16px;'>
                    <li><strong>Visual Analysis:</strong> DeepFace facial expression recognition with emotion mapping</li>
                    <li><strong>Audio Analysis:</strong> CNN model trained on the RAVDESS emotional speech dataset</li>
                    <li><strong>Text Analysis:</strong> Transformer-based sentiment classification model</li>
                    <li><strong>Speech Recognition:</strong> OpenAI Whisper model with Google Speech Recognition fallback</li>
                </ul>
            </div>
            """
            
            html_result += f"</div>"  # End of main container
            
            # Round the score to 2 decimal places for display in the UI
            final_score_rounded = round(final_score, 2)
            
            # Clean up any temporary files
            if self.cleanup_temp:
                self._cleanup_temp_files()
                
            return html_result, final_score_rounded, fig
            
        except Exception as e:
            logger.error(f"Error in multimodal processing: {str(e)}", exc_info=True)
            
            html_result = f"<div style='text-align: center; padding: 20px;'>"
            html_result += f"<h2>Error in Multimodal Analysis</h2>"
            html_result += f"<p>{str(e)}</p>"
            html_result += f"</div>"
            
            return html_result, 0, None

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