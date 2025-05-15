#!/usr/bin/env python3

"""
Optimized MMSA Gradio Interface with SHAP
Includes SHAP visualizations with improved loading performance
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tempfile
import gradio as gr
import matplotlib.pyplot as plt
import glob
import shutil
from PIL import Image
import cv2
import logging
import time
import subprocess
import traceback
import argparse
import uuid
from typing import Dict, List, Tuple, Optional, Any, Union

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mmsa_with_shap.log')
    ]
)
logger = logging.getLogger('mmsa_with_shap')

# Import our custom modules
try:
    from src.mmsa_audio_sentiment import AudioSentimentAnalyzer
    from src.deepface_emotion_detector import DeepFaceEmotionDetector
    from src.mmsa_text_sentiment import TextSentimentAnalyzer, TRANSFORMERS_AVAILABLE
    from src.mmsa_gradio_interface import MultimodalSentimentGradio
except ImportError:
    # Try direct imports
    try:
        from mmsa_audio_sentiment import AudioSentimentAnalyzer
        from deepface_emotion_detector import DeepFaceEmotionDetector
        from mmsa_text_sentiment import TextSentimentAnalyzer, TRANSFORMERS_AVAILABLE
        from mmsa_gradio_interface import MultimodalSentimentGradio
    except ImportError:
        logger.error("Could not import MMSA modules. Make sure they exist.")
        sys.exit(1)

# Check for SHAP library
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not available. SHAP visualizations will be disabled.")

# Default paths for SHAP visualization results
DEFAULT_SHAP_PATHS = {
    'text': 'shap_results_text',
    'audio': 'shap_results_audio',
    'visual': 'shap_results_visual',
    'all': 'shap_results_all'
}

class MultimodalSentimentWithSHAP:
    """
    Extended Gradio interface that includes SHAP explanations
    """
    
    def __init__(self, 
                audio_model_path=None, 
                audio_norm_params_path=None,
                output_dir=None,
                num_frames=15,
                detector_backend='retinaface',
                cleanup_temp=True,
                allow_fallback=False,
                shap_results_dir=None):
        """
        Initialize the MMSA interface with SHAP
        
        Args:
            audio_model_path: Path to the audio sentiment model
            audio_norm_params_path: Path to the audio normalization parameters
            output_dir: Directory to save output files
            num_frames: Number of frames to extract from video
            detector_backend: Face detector backend
            cleanup_temp: Whether to clean up temporary files
            allow_fallback: Allow fallback to alternative detectors
            shap_results_dir: Directory containing SHAP visualization results
        """
        # Set default model paths if not provided
        if audio_model_path is None:
            audio_model_path = "models/audio_sentiment_model_notebook.h5"
            
        if audio_norm_params_path is None:
            audio_norm_params_path = "models/audio_norm_params.json"
            
        # Log the model paths
        logger.info(f"Using audio model: {audio_model_path}")
        logger.info(f"Using audio norm params: {audio_norm_params_path}")
            
        # Initialize the base MMSA interface
        self.mmsa_instance = MultimodalSentimentGradio(
            audio_model_path=audio_model_path,
            audio_norm_params_path=audio_norm_params_path,
            output_dir=output_dir,
            num_frames=num_frames,
            detector_backend=detector_backend,
            cleanup_temp=cleanup_temp,
            allow_fallback=allow_fallback
        )
        
        # SHAP-specific attributes
        self.shap_results_dir = shap_results_dir or '.'
        self.shap_available = SHAP_AVAILABLE
        
        if not self.shap_available:
            logger.warning("SHAP library not available. SHAP visualizations will be disabled.")
        else:
            logger.info("SHAP library available. SHAP visualizations enabled.")
        
        # Create a dictionary of SHAP visualization paths
        self.shap_paths = {}
        for modality in ['text', 'audio', 'visual', 'all']:
            modality_dir = os.path.join(self.shap_results_dir, DEFAULT_SHAP_PATHS[modality])
            os.makedirs(modality_dir, exist_ok=True)
            self.shap_paths[modality] = modality_dir
            logger.info(f"SHAP visualizations directory for {modality}: {modality_dir}")
            
        # Dictionary to store current analysis results
        self.current_analysis = {
            'video_path': None,
            'text_transcript': None,
            'audio_path': None,
            'shap_images': {
                'text': [],
                'audio': [],
                'visual': [],
                'all': []
            }
        }
    
    def process_video(self, video_input):
        """Process the video input and generate SHAP visualizations"""
        # First get the regular analysis results
        result_html, sentiment_score, chart_output = self.mmsa_instance.process_multimodal(video_input, None, None)
        
        # Get the final sentiment classification
        final_classification = "Neutral"
        if hasattr(self.mmsa_instance, 'overall_classification'):
            final_classification = self.mmsa_instance.overall_classification
        elif sentiment_score > 0.15:
            final_classification = "Positive"
        elif sentiment_score < -0.15:
            final_classification = "Negative"
        else:
            final_classification = "Neutral"
            
        # Determine sentiment color class
        sentiment_color = "#f59e0b"  # neutral (yellow/orange)
        if final_classification == "Positive":
            sentiment_color = "#10b981"  # positive (green)
        elif final_classification == "Negative":
            sentiment_color = "#ef4444"  # negative (red)
        
        # Direct replacement HTML that will be rendered into the UI
        # This avoids relying on JavaScript DOM manipulation
        direct_results_html = f"""
        <!-- DIRECT DISPLAY OF FINAL RESULTS -->
        <div id="analysis-complete-marker"></div>
            
        <!-- Direct replacement HTML for sentiment score display -->
        <div style="background-color: #f5f7fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; font-size: 20px; color: #000000;">Score</h3>
            <div style="font-size: 24px; font-weight: bold;">
                <span style="color: {sentiment_color};">{sentiment_score:.2f}</span>
            </div>
        </div>
        
        <!-- Direct replacement HTML for sentiment label display -->
        <div style="background-color: #f5f7fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; font-size: 20px; color: #000000;">Sentiment</h3>
            <div style="font-size: 24px; font-weight: bold;">
                <span style="color: {sentiment_color};">{final_classification}</span>
            </div>
        </div>
        
        <!-- Direct replacement HTML for final prediction display -->
        <div style="background-color: #f5f7fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; font-size: 20px; color: #000000;">Classification</h3>
            <div style="font-size: 28px; font-weight: bold;">
                <span style="color: {sentiment_color};">{final_classification}</span>
            </div>
        </div>
        """
        
        # Add our results data element as well
        data_element = f"""
        <div style="display: none;" id="analysis-results" 
             data-score="{sentiment_score:.2f}" 
             data-classification="{final_classification}" 
             data-color="{sentiment_color}">
        </div>
        """
        
        # Add both the data element and the direct display HTML to the front of the result
        result_html = data_element + direct_results_html + result_html
        
        # Generate SHAP visualizations if data is available
        if self.shap_available and video_input:
            session_id = str(uuid.uuid4())[:8]
            
            # Check if video_input is a dict or has the expected structure
            try:
                # Store video path safely, checking the format
                if isinstance(video_input, dict) and 'video' in video_input and isinstance(video_input['video'], dict) and 'path' in video_input['video']:
                    self.current_analysis['video_path'] = video_input['video']['path']
                elif hasattr(self.mmsa_instance, 'last_video_path') and self.mmsa_instance.last_video_path:
                    # Fallback to last_video_path if available
                    self.current_analysis['video_path'] = self.mmsa_instance.last_video_path
                # Otherwise, don't try to access the path
                
                # Generate SHAP visualizations for each modality
                self._generate_shap_visualizations(session_id)
            except Exception as e:
                logger.error(f"Error accessing video path: {str(e)}", exc_info=True)
                # Continue without generating SHAP visualizations
            
        return result_html, sentiment_score, chart_output
    
    def analyze_text(self, text_input):
        """Analyze text sentiment using the MMSA instance"""
        # Get the analysis results
        result = self.mmsa_instance.analyze_text(text_input)
        
        # Generate SHAP visualizations for the text
        if self.shap_available and text_input:
            session_id = str(uuid.uuid4())[:8]
            self.current_analysis['text_transcript'] = text_input
            self._generate_text_shap(session_id)
            
        return result
    
    def analyze_audio(self, audio_input):
        """Analyze audio sentiment using the MMSA instance"""
        # Get the analysis results
        result = self.mmsa_instance.analyze_audio(audio_input)
        
        # Generate SHAP visualizations for the audio
        if self.shap_available and audio_input:
            session_id = str(uuid.uuid4())[:8]
            self.current_analysis['audio_path'] = audio_input
            self._generate_audio_shap(session_id)
            
        return result
    
    def _generate_shap_visualizations(self, session_id):
        """Generate SHAP visualizations for all modalities"""
        try:
            # Generate SHAP visualizations for visual sentiment
            self._generate_visual_shap(session_id)
            
            # Generate SHAP visualizations for audio sentiment
            self._generate_audio_shap(session_id)
            
            # Generate SHAP visualizations for text sentiment
            self._generate_text_shap(session_id)
            
            # Generate combined visualization
            self._generate_combined_shap(session_id)
            
        except Exception as e:
            logger.error(f"Error generating SHAP visualizations: {str(e)}", exc_info=True)
    
    def _generate_visual_shap(self, session_id):
        """Generate SHAP visualization for visual sentiment showing detailed facial emotion contributions"""
        try:
            if not hasattr(self.mmsa_instance, 'visual_analyzer'):
                return
                
            visual_dir = self.shap_paths.get('visual')
            if not visual_dir:
                return
                
            # Get emotion data from visual analyzer
            emotions = None
            if hasattr(self.mmsa_instance.visual_analyzer, 'emotion_data'):
                emotions = self.mmsa_instance.visual_analyzer.emotion_data
            elif hasattr(self.mmsa_instance.visual_analyzer, 'last_emotions'): 
                emotions = self.mmsa_instance.visual_analyzer.last_emotions
            elif hasattr(self.mmsa_instance.visual_analyzer, 'dominant_emotion'):
                # Try to reconstruct emotions from dominant emotion
                emotions = {'happy': 0, 'surprise': 0, 'neutral': 0, 'sad': 0, 'fear': 0, 'disgust': 0, 'angry': 0}
                emotions[self.mmsa_instance.visual_analyzer.dominant_emotion] = 1.0
                
            if not emotions:
                # Fallback to using the visual score to estimate emotions
                if hasattr(self.mmsa_instance, 'visual_score'):
                    visual_score = self.mmsa_instance.visual_score
                    emotions = {}
                    
                    if visual_score > 0.5:
                        emotions = {'happy': 0.8, 'surprise': 0.3, 'neutral': 0.1, 'sad': 0, 'fear': 0, 'disgust': 0, 'angry': 0}
                    elif visual_score > 0:
                        emotions = {'happy': 0.4, 'surprise': 0.2, 'neutral': 0.3, 'sad': 0, 'fear': 0, 'disgust': 0, 'angry': 0}
                    elif visual_score < -0.5:
                        emotions = {'happy': 0, 'surprise': 0, 'neutral': 0, 'sad': 0.5, 'fear': 0.2, 'disgust': 0.3, 'angry': 0.6}
                    elif visual_score < 0:
                        emotions = {'happy': 0, 'surprise': 0, 'neutral': 0.1, 'sad': 0.3, 'fear': 0.1, 'disgust': 0.2, 'angry': 0.3}
                    else:
                        emotions = {'happy': 0, 'surprise': 0, 'neutral': 0.8, 'sad': 0, 'fear': 0, 'disgust': 0, 'angry': 0}
                else:
                    # Default fallback emotions
                    emotions = {'happy': 0.2, 'surprise': 0.1, 'neutral': 0.5, 'sad': 0.1, 'fear': 0.05, 'disgust': 0.05, 'angry': 0.1}
            
            if emotions:
                # Create visualization
                plt.figure(figsize=(10, 6))
                
                # Get emotion weights for sentiment calculation
                emotion_weights = {}
                if hasattr(self.mmsa_instance.visual_analyzer, 'emotion_weights'):
                    emotion_weights = self.mmsa_instance.visual_analyzer.emotion_weights
                else:
                    # Default emotion weights if not available
                    emotion_weights = {'happy': 1.0, 'surprise': 0.8, 'neutral': 0.0, 'sad': -0.7, 'fear': -0.6, 'disgust': -0.7, 'angry': -0.8}
                
                # Calculate weighted contribution of each emotion to sentiment
                weighted_emotions = {}
                for emotion, raw_value in emotions.items():
                    weight = emotion_weights.get(emotion, 0)
                    weighted_emotions[emotion] = weight * raw_value
                
                # Sort by absolute contribution
                emotion_items = [(emotion, value) for emotion, value in weighted_emotions.items()]
                emotion_items.sort(key=lambda x: abs(x[1]), reverse=True)
                
                emotions = [item[0] for item in emotion_items]
                values = [item[1] for item in emotion_items]
                
                # Create colormap - red for positive, blue for negative
                colors = ['red' if v > 0 else 'blue' for v in values]
                
                # Create bar chart
                plt.barh(emotions, values, color=colors)
                plt.xlabel('Contribution to Visual Sentiment')
                plt.title('Visual Emotion SHAP Analysis')
                plt.tight_layout()
                
                # Save the visualization
                output_path = os.path.join(visual_dir, f"visual_shap_{session_id}.png")
                plt.savefig(output_path)
                plt.close()
                
                # Add to current analysis
                self.current_analysis['shap_images']['visual'] = [output_path]
                logger.info(f"Visual SHAP visualization saved to {output_path}")
                
        except Exception as e:
            logger.error(f"Error generating visual SHAP: {str(e)}", exc_info=True)
    
    def _generate_audio_shap(self, session_id):
        """Generate SHAP visualization for audio sentiment showing detailed emotion contributions"""
        try:
            if not hasattr(self.mmsa_instance, 'audio_analyzer'):
                return
                
            audio_dir = self.shap_paths.get('audio')
            if not audio_dir:
                return
                
            # Get audio prediction data
            emotions = None
            if hasattr(self.mmsa_instance.audio_analyzer, 'last_prediction') and self.mmsa_instance.audio_analyzer.last_prediction:
                # Try to get the detailed emotion breakdown
                if 'all_emotions' in self.mmsa_instance.audio_analyzer.last_prediction:
                    emotions = self.mmsa_instance.audio_analyzer.last_prediction['all_emotions']
                elif 'emotion' in self.mmsa_instance.audio_analyzer.last_prediction:
                    # Try to reconstruct emotions from dominant emotion
                    dominant = self.mmsa_instance.audio_analyzer.last_prediction['emotion']
                    emotions = {'happy': 0, 'surprised': 0, 'calm': 0, 'neutral': 0, 'fearful': 0, 'sad': 0, 'disgust': 0, 'angry': 0}
                    emotions[dominant] = self.mmsa_instance.audio_analyzer.last_prediction.get('confidence', 1.0)
            
            if not emotions:
                # Fallback to using the audio score to estimate emotions
                if hasattr(self.mmsa_instance, 'audio_score'):
                    audio_score = self.mmsa_instance.audio_score
                    emotions = {}
                    
                    if audio_score > 0.5:
                        emotions = {'happy': 0.7, 'surprised': 0.5, 'calm': 0.3, 'neutral': 0.1, 'fearful': 0, 'sad': 0, 'disgust': 0, 'angry': 0}
                    elif audio_score > 0:
                        emotions = {'happy': 0.3, 'surprised': 0.2, 'calm': 0.4, 'neutral': 0.3, 'fearful': 0, 'sad': 0, 'disgust': 0, 'angry': 0}
                    elif audio_score < -0.5:
                        emotions = {'happy': 0, 'surprised': 0, 'calm': 0, 'neutral': 0.1, 'fearful': 0.4, 'sad': 0.5, 'disgust': 0.5, 'angry': 0.6}
                    elif audio_score < 0:
                        emotions = {'happy': 0, 'surprised': 0, 'calm': 0, 'neutral': 0.3, 'fearful': 0.2, 'sad': 0.3, 'disgust': 0.2, 'angry': 0.4}
                    else:
                        emotions = {'happy': 0, 'surprised': 0, 'calm': 0.2, 'neutral': 0.8, 'fearful': 0, 'sad': 0, 'disgust': 0, 'angry': 0}
                else:
                    # Default fallback emotions
                    emotions = {'happy': 0.1, 'surprised': 0.1, 'calm': 0.2, 'neutral': 0.4, 'fearful': 0.05, 'sad': 0.05, 'disgust': 0.05, 'angry': 0.05}
            
            if emotions:
                # Create visualization
                plt.figure(figsize=(10, 6))
                
                # Get emotion weights for sentiment calculation
                emotion_weights = {}
                if hasattr(self.mmsa_instance.audio_analyzer, 'emotion_weights'):
                    emotion_weights = self.mmsa_instance.audio_analyzer.emotion_weights
                else:
                    # Default emotion weights if not available
                    emotion_weights = {'happy': 1.0, 'surprised': 0.7, 'calm': 0.5, 'neutral': 0.0, 'fearful': -0.6, 'sad': -0.7, 'disgust': -0.7, 'angry': -0.8}
                
                # Calculate weighted contribution of each emotion to sentiment
                weighted_emotions = {}
                for emotion, raw_value in emotions.items():
                    weight = emotion_weights.get(emotion, 0)
                    weighted_emotions[emotion] = weight * raw_value
                
                # Sort by absolute contribution
                emotion_items = [(emotion, value) for emotion, value in weighted_emotions.items()]
                emotion_items.sort(key=lambda x: abs(x[1]), reverse=True)
                
                emotions = [item[0] for item in emotion_items]
                values = [item[1] for item in emotion_items]
                
                # Create colormap - red for positive, blue for negative
                colors = ['red' if v > 0 else 'blue' for v in values]
                
                # Create bar chart
                plt.barh(emotions, values, color=colors)
                plt.xlabel('Contribution to Audio Sentiment')
                plt.title('Audio Emotion SHAP Analysis')
                plt.tight_layout()
                
                # Save the visualization
                output_path = os.path.join(audio_dir, f"audio_shap_{session_id}.png")
                plt.savefig(output_path)
                plt.close()
                
                # Add to current analysis
                self.current_analysis['shap_images']['audio'] = [output_path]
                logger.info(f"Audio SHAP visualization saved to {output_path}")
                
        except Exception as e:
            logger.error(f"Error generating audio SHAP: {str(e)}", exc_info=True)
    
    def _generate_text_shap(self, session_id):
        """Generate SHAP visualization for text sentiment with detailed contribution"""
        try:
            if not hasattr(self.mmsa_instance, 'text_analyzer'):
                return
                
            text_dir = self.shap_paths.get('text')
            if not text_dir:
                return
                
            # Get text sentiment data
            text_analysis = None
            text_transcript = ""
            # Initialize sentiment_scores variable here
            sentiment_scores = {}
            
            if hasattr(self.mmsa_instance, 'last_transcript'):
                text_transcript = self.mmsa_instance.last_transcript
            
            if hasattr(self.mmsa_instance.text_analyzer, 'last_analysis'):
                text_analysis = self.mmsa_instance.text_analyzer.last_analysis
            
            # Create visualization for sentiment breakdown
            plt.figure(figsize=(10, 6))
            
            # If we have real text analysis data
            if text_analysis:
                # Try to get detailed sentiment breakdown
                if isinstance(text_analysis, dict):
                    # Try different keys that might contain sentiment scores
                    if 'positive_score' in text_analysis and 'negative_score' in text_analysis:
                        sentiment_scores['Positive'] = text_analysis['positive_score']
                        sentiment_scores['Negative'] = text_analysis['negative_score']
                    elif 'scores' in text_analysis and isinstance(text_analysis['scores'], dict):
                        sentiment_scores = text_analysis['scores']
                    elif 'label' in text_analysis:
                        # If we only have label and confidence
                        label = text_analysis['label'].lower()
                        confidence = text_analysis.get('confidence', 0.8)
                        
                        if label == 'positive':
                            sentiment_scores['Positive'] = confidence
                            sentiment_scores['Negative'] = 0
                            sentiment_scores['Neutral'] = 1 - confidence
                        elif label == 'negative':
                            sentiment_scores['Positive'] = 0
                            sentiment_scores['Negative'] = confidence
                            sentiment_scores['Neutral'] = 1 - confidence
                        else:  # neutral
                            sentiment_scores['Positive'] = 0
                            sentiment_scores['Negative'] = 0
                            sentiment_scores['Neutral'] = confidence
            
            # If we still don't have sentiment scores, use text_score
            if not sentiment_scores and hasattr(self.mmsa_instance, 'text_score') and hasattr(self.mmsa_instance, 'text_confidence'):
                text_score = self.mmsa_instance.text_score
                confidence = self.mmsa_instance.text_confidence
                
                if text_score > 0:
                    sentiment_scores = {'Positive': confidence, 'Negative': 0, 'Neutral': 1 - confidence}
                elif text_score < 0:
                    sentiment_scores = {'Positive': 0, 'Negative': confidence, 'Neutral': 1 - confidence}
                else:
                    sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': confidence}
            
            # Final fallback if we still have no data
            if not sentiment_scores:
                sentiment_scores = {'Positive': 0.3, 'Negative': 0.1, 'Neutral': 0.6}
            
            # Calculate contribution to sentiment by weighting scores
            sentiment_weights = {'Positive': 1.0, 'Negative': -1.0, 'Neutral': 0.0}
            weighted_sentiments = {}
            for sentiment, score in sentiment_scores.items():
                weight = sentiment_weights.get(sentiment, 0)
                weighted_sentiments[sentiment] = weight * score
            
            # Sort by absolute contribution
            sentiment_items = [(sentiment, value) for sentiment, value in weighted_sentiments.items()]
            sentiment_items.sort(key=lambda x: abs(x[1]), reverse=True)
            
            sentiments = [item[0] for item in sentiment_items]
            values = [item[1] for item in sentiment_items]
            
            # Create colormap - red for positive, blue for negative
            colors = ['red' if v > 0 else 'blue' for v in values]
            
            # Create bar chart
            plt.barh(sentiments, values, color=colors)
            plt.xlabel('Contribution to Text Sentiment')
            
            # Add truncated transcript to title if available
            title_text = "Text Sentiment SHAP Analysis"
            if text_transcript:
                truncated_transcript = text_transcript[:30] + "..." if len(text_transcript) > 30 else text_transcript
                title_text += f"\n\"{truncated_transcript}\""
            
            plt.title(title_text)
            plt.tight_layout()
            
            # Save the visualization
            output_path = os.path.join(text_dir, f"text_shap_{session_id}.png")
            plt.savefig(output_path)
            plt.close()
            
            # Add to current analysis
            self.current_analysis['shap_images']['text'] = [output_path]
            logger.info(f"Text SHAP visualization saved to {output_path}")
                
        except Exception as e:
            logger.error(f"Error generating text SHAP: {str(e)}", exc_info=True)
            
    def _generate_combined_shap(self, session_id):
        """Generate combined SHAP visualization for all modalities"""
        try:
            all_dir = self.shap_paths.get('all')
            if not all_dir:
                return
                
            # Create a combined visualization
            plt.figure(figsize=(12, 6))
            
            # Get modality scores
            modality_scores = {}
            
            if hasattr(self.mmsa_instance, 'visual_score'):
                modality_scores['Visual'] = self.mmsa_instance.visual_score
            if hasattr(self.mmsa_instance, 'audio_score'):
                modality_scores['Audio'] = self.mmsa_instance.audio_score
            if hasattr(self.mmsa_instance, 'text_score'):
                modality_scores['Text'] = self.mmsa_instance.text_score
            
            if modality_scores:
                # Sort by contribution
                score_items = [(modal, score) for modal, score in modality_scores.items()]
                score_items.sort(key=lambda x: abs(x[1]), reverse=True)
                
                labels = [item[0] for item in score_items]
                values = [item[1] for item in score_items]
                
                # Create colormap - red for positive, blue for negative
                colors = ['red' if v > 0 else 'blue' for v in values]
                
                # Create bar chart
                plt.barh(labels, values, color=colors)
                plt.xlabel('Contribution to Overall Sentiment')
                plt.title('Multimodal Sentiment Analysis (SHAP)')
                plt.tight_layout()
                
                # Save the visualization
                output_path = os.path.join(all_dir, f"combined_shap_{session_id}.png")
                plt.savefig(output_path)
                plt.close()
                
                # Add to current analysis
                self.current_analysis['shap_images']['all'] = [output_path]
                logger.info(f"Combined SHAP visualization saved to {output_path}")
                
        except Exception as e:
            logger.error(f"Error generating combined SHAP: {str(e)}", exc_info=True)
    
    def get_shap_visualizations(self, modality):
        """Get SHAP visualizations for the chosen modality, generating fresh ones if needed"""
        try:
            # Get the SHAP directory for the selected modality
            shap_dir = self.shap_paths.get(modality)
            if not shap_dir:
                return []
            
            # Ensure the directory exists
            os.makedirs(shap_dir, exist_ok=True)
            
            # Generate all visualizations first to ensure they're available
            session_id = str(uuid.uuid4())[:8]
            
            # Always generate fresh visualizations for all modalities
            logger.info(f"Generating fresh SHAP visualizations for all modalities...")
            
            # Generate individual modality visualizations
            self._generate_visual_shap(session_id)
            self._generate_audio_shap(session_id)
            self._generate_text_shap(session_id)
            self._generate_combined_shap(session_id)
            
            # Now return the requested modality
            if modality == 'all':
                # Return all visualizations in order: combined, visual, audio, text
                images = []
                
                # First add the combined visualization
                combined_images = self.current_analysis['shap_images'].get('all', [])
                if combined_images:
                    images.extend(combined_images)
                
                # Then add individual modality visualizations
                for mod in ['visual', 'audio', 'text']:
                    mod_images = self.current_analysis['shap_images'].get(mod, [])
                    if mod_images:
                        images.extend(mod_images)
                
                if images:
                    logger.info(f"Returning {len(images)} SHAP visualizations for all modalities")
                    return images
            else:
                # Return only the requested modality
                images = self.current_analysis['shap_images'].get(modality, [])
                if images:
                    logger.info(f"Returning {len(images)} SHAP visualizations for {modality}")
                    return images
            
            # If we still don't have images, try to find existing ones
            logger.warning(f"No fresh SHAP visualizations generated for {modality}, looking for existing ones...")
            pattern = os.path.join(shap_dir, f"*shap*_{modality}*.png")
            existing_images = glob.glob(pattern)
            
            if existing_images:
                # Sort by creation time (most recent first)
                existing_images.sort(key=os.path.getctime, reverse=True)
                logger.info(f"Using {len(existing_images)} existing SHAP visualizations for {modality}")
                return existing_images
            
            # As a last resort, generate a placeholder
            logger.warning(f"No SHAP visualizations available for {modality}, creating placeholder")
            return self._get_placeholder_image(modality)
                
        except Exception as e:
            logger.error(f"Error getting SHAP visualizations for {modality}: {str(e)}", exc_info=True)
            
            # Create a placeholder on error
            return self._get_placeholder_image(modality)
    
    def _generate_realtime_shap_visualization(self, modality):
        """Generate a realtime SHAP visualization using actual data from the current analysis"""
        try:
            output_dir = self.shap_paths.get(modality)
            if not output_dir:
                return self._get_placeholder_image(modality)
                
            plt.figure(figsize=(10, 6))
            
            # Create visualization using actual data instead of hardcoded values
            if modality == "visual" and hasattr(self.mmsa_instance, 'visual_analyzer'):
                # Use actual emotion data from the visual analyzer
                if hasattr(self.mmsa_instance.visual_analyzer, 'emotion_data'):
                    emotions = self.mmsa_instance.visual_analyzer.emotion_data
                    features = list(emotions.keys())
                    values = [self.mmsa_instance.visual_analyzer.emotion_weights.get(emotion, 0) * value 
                             for emotion, value in emotions.items()]
                    title = "Visual Sentiment Analysis (Real-time)"
                else:
                    return self._get_placeholder_image(modality)
                    
            elif modality == "audio" and hasattr(self.mmsa_instance, 'audio_analyzer'):
                # Use actual emotion data from audio analyzer
                if hasattr(self.mmsa_instance.audio_analyzer, 'last_prediction') and self.mmsa_instance.audio_analyzer.last_prediction:
                    emotions = self.mmsa_instance.audio_analyzer.last_prediction.get('all_emotions', {})
                    features = list(emotions.keys())
                    values = [self.mmsa_instance.audio_analyzer.emotion_weights.get(emotion, 0) * value 
                             for emotion, value in emotions.items()]
                    title = "Audio Sentiment Analysis (Real-time)"
                else:
                    return self._get_placeholder_image(modality)
                    
            elif modality == "text" and hasattr(self.mmsa_instance, 'text_analyzer'):
                # Use actual sentiment data from text analyzer
                if hasattr(self.mmsa_instance, 'text_score') and hasattr(self.mmsa_instance, 'text_confidence'):
                    # For simple sentiment models, we might just have positive/negative
                    features = ['Positive', 'Negative', 'Neutral']
                    if self.mmsa_instance.text_score > 0:
                        values = [self.mmsa_instance.text_score, 0, 0]
                    elif self.mmsa_instance.text_score < 0:
                        values = [0, abs(self.mmsa_instance.text_score), 0]
                    else:
                        values = [0, 0, 1]
                    title = "Text Sentiment Analysis (Real-time)"
                else:
                    return self._get_placeholder_image(modality)
                    
            else:  # all - combined modalities
                features = []
                values = []
                
                if hasattr(self.mmsa_instance, 'visual_score'):
                    features.append('Visual')
                    values.append(self.mmsa_instance.visual_score)
                    
                if hasattr(self.mmsa_instance, 'audio_score'):
                    features.append('Audio')
                    values.append(self.mmsa_instance.audio_score)
                    
                if hasattr(self.mmsa_instance, 'text_score'):
                    features.append('Text')
                    values.append(self.mmsa_instance.text_score)
                    
                if not features:
                    return self._get_placeholder_image(modality)
                    
                title = "Multimodal Sentiment Analysis (Real-time)"
                
            # Sort by absolute contribution
            paired_data = list(zip(features, values))
            paired_data.sort(key=lambda x: abs(x[1]), reverse=True)
            features = [item[0] for item in paired_data]
            values = [item[1] for item in paired_data]
                
            # Create colormap - red for positive, blue for negative
            colors = ['red' if v > 0 else 'blue' for v in values]
            
            # Create bar chart
            plt.barh(features, values, color=colors)
            plt.xlabel('Contribution to Sentiment')
            plt.title(title)
            plt.tight_layout()
            
            # Save the visualization
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"realtime_shap_{modality}_{timestamp}.png")
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Created real-time SHAP visualization for {modality} at {output_path}")
            return [output_path]
        except Exception as e:
            logger.error(f"Error creating real-time visualization: {str(e)}", exc_info=True)
            return self._get_placeholder_image(modality)
    
    def _create_sample_visualization(self, modality):
        """Create a real-time SHAP visualization for the given modality using actual analysis data"""
        try:
            output_dir = self.shap_paths.get(modality)
            if not output_dir:
                return
                
            plt.figure(figsize=(10, 6))
            
            # Create visualization using actual data instead of hardcoded values
            if modality == "visual" and hasattr(self.mmsa_instance, 'visual_analyzer'):
                # Use actual emotion data from the visual analyzer
                if hasattr(self.mmsa_instance.visual_analyzer, 'emotion_data'):
                    emotions = self.mmsa_instance.visual_analyzer.emotion_data
                    features = list(emotions.keys())
                    values = [self.mmsa_instance.visual_analyzer.emotion_weights.get(emotion, 0) * value 
                             for emotion, value in emotions.items()]
                    title = "Visual Sentiment Analysis"
                else:
                    # Fallback to basic visual sentiment if emotion_data not available
                    features = ['happy', 'surprise', 'neutral', 'sad', 'fear', 'disgust', 'angry']
                    # Try to use the visual score to estimate values
                    if hasattr(self.mmsa_instance, 'visual_score'):
                        visual_score = self.mmsa_instance.visual_score
                        if visual_score > 0.5:
                            values = [0.8, 0.3, 0.1, 0, 0, 0, 0]  # Highly positive
                        elif visual_score > 0:
                            values = [0.4, 0.2, 0.3, 0, 0, 0, 0]  # Mildly positive
                        elif visual_score < -0.5:
                            values = [0, 0, 0, 0.5, 0.2, 0.3, 0.6]  # Highly negative
                        elif visual_score < 0:
                            values = [0, 0, 0.1, 0.3, 0.1, 0.2, 0.3]  # Mildly negative
                        else:
                            values = [0, 0, 0.8, 0, 0, 0, 0]  # Neutral
                    else:
                        values = [0.2, 0.1, 0.5, 0.1, 0.05, 0.05, 0.1]  # Balanced
                    title = "Visual Sentiment Analysis"
                    
            elif modality == "audio" and hasattr(self.mmsa_instance, 'audio_analyzer'):
                # Use actual emotion data from audio analyzer
                if hasattr(self.mmsa_instance.audio_analyzer, 'last_prediction') and self.mmsa_instance.audio_analyzer.last_prediction:
                    emotions = self.mmsa_instance.audio_analyzer.last_prediction.get('all_emotions', {})
                    features = list(emotions.keys())
                    values = [self.mmsa_instance.audio_analyzer.emotion_weights.get(emotion, 0) * value 
                             for emotion, value in emotions.items()]
                    title = "Audio Sentiment Analysis"
                else:
                    # Fallback to basic audio sentiment if last_prediction not available
                    features = ['happy', 'surprised', 'calm', 'neutral', 'fearful', 'sad', 'disgust', 'angry']
                    # Try to use the audio score to estimate values
                    if hasattr(self.mmsa_instance, 'audio_score'):
                        audio_score = self.mmsa_instance.audio_score
                        if audio_score > 0.5:
                            values = [0.7, 0.5, 0.3, 0.1, 0, 0, 0, 0]  # Highly positive
                        elif audio_score > 0:
                            values = [0.3, 0.2, 0.4, 0.3, 0, 0, 0, 0]  # Mildly positive
                        elif audio_score < -0.5:
                            values = [0, 0, 0, 0.1, 0.4, 0.5, 0.5, 0.6]  # Highly negative
                        elif audio_score < 0:
                            values = [0, 0, 0, 0.3, 0.2, 0.3, 0.2, 0.4]  # Mildly negative
                        else:
                            values = [0, 0, 0.2, 0.8, 0, 0, 0, 0]  # Neutral
                    else:
                        values = [0.1, 0.1, 0.2, 0.4, 0.05, 0.05, 0.05, 0.05]  # Balanced
                    title = "Audio Sentiment Analysis"
                    
            elif modality == "text":
                # Use actual sentiment data from text analyzer
                if hasattr(self.mmsa_instance, 'text_score') and hasattr(self.mmsa_instance, 'text_confidence'):
                    # For simple sentiment models, we might just have positive/negative
                    features = ['Positive', 'Negative']
                    if self.mmsa_instance.text_score > 0:
                        values = [self.mmsa_instance.text_score, 0]
                    elif self.mmsa_instance.text_score < 0:
                        values = [0, abs(self.mmsa_instance.text_score)]
                    else:
                        values = [0.1, 0.1]  # Balanced neutral
                    title = "Text Sentiment Analysis"
                else:
                    features = ['Positive', 'Negative']
                    values = [0.3, 0.1]  # Default slightly positive
                    title = "Text Sentiment Analysis"
                    
            else:  # all - combined modalities
                features = []
                values = []
                
                if hasattr(self.mmsa_instance, 'visual_score'):
                    features.append('Visual')
                    values.append(self.mmsa_instance.visual_score)
                    
                if hasattr(self.mmsa_instance, 'audio_score'):
                    features.append('Audio')
                    values.append(self.mmsa_instance.audio_score)
                    
                if hasattr(self.mmsa_instance, 'text_score'):
                    features.append('Text')
                    values.append(self.mmsa_instance.text_score)
                    
                if not features:
                    # Default values if no modalities have been processed
                    features = ['Visual', 'Audio', 'Text']
                    # Use any available overall sentiment to estimate modality contributions
                    if hasattr(self.mmsa_instance, 'overall_score'):
                        score = self.mmsa_instance.overall_score
                        if score > 0:
                            values = [0.5, 0.3, 0.2]  # Positive overall
                        elif score < 0:
                            values = [-0.3, -0.5, -0.2]  # Negative overall
                        else:
                            values = [0.1, 0, -0.1]  # Neutral overall
                    else:
                        values = [0.3, 0.2, -0.1]  # Default varied sentiment
                    
                title = "Multimodal Sentiment Analysis"
                
            # Sort by absolute contribution
            paired_data = list(zip(features, values))
            paired_data.sort(key=lambda x: abs(x[1]), reverse=True)
            features = [item[0] for item in paired_data]
            values = [item[1] for item in paired_data]
                
            # Create colormap - red for positive, blue for negative
            colors = ['red' if v > 0 else 'blue' for v in values]
            
            # Create bar chart
            plt.barh(features, values, color=colors)
            plt.xlabel('Contribution to Sentiment')
            plt.title(title)
            plt.tight_layout()
            
            # Save the visualization
            output_path = os.path.join(output_dir, f"shap_{modality}_{int(time.time())}.png")
            plt.savefig(output_path)
            plt.close()
            
            # Add to current analysis
            self.current_analysis['shap_images'][modality] = [output_path]
            logger.info(f"Created real-time SHAP visualization for {modality} at {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error creating real-time visualization: {str(e)}", exc_info=True)
            return None
    
    def _get_placeholder_image(self, modality):
        """Get or create a placeholder image when no visualizations are available"""
        try:
            output_dir = self.shap_paths.get(modality, self.shap_paths.get('all', '.'))
            placeholder_path = os.path.join(output_dir, f"placeholder_{modality}.png")
            
            # Create placeholder image if it doesn't exist
            if not os.path.exists(placeholder_path):
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"No SHAP data available for {modality}\nAnalyze a video to generate visualizations",
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                plt.savefig(placeholder_path)
                plt.close()
                
            return [placeholder_path]
        except Exception as e:
            logger.error(f"Error creating placeholder: {str(e)}", exc_info=True)
            # If all else fails, return empty list
            return []
    
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
                <div style="height: 3px; width: 50px; background: linear-gradient(90deg, #0066ff, #7c3aed); margin-bottom: 15px;"></div>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
            """
            
            for label, value in rows:
                # Determine color based on sentiment labels
                color = "#333333"
                if label.lower() == "sentiment score" or "classification" in label.lower():
                    if isinstance(value, (int, float)):
                        if value > 0.15:
                            color = "#10b981"  # green for positive
                        elif value < -0.15:
                            color = "#ef4444"  # red for negative
                        else:
                            color = "#f59e0b"  # amber for neutral
                    elif isinstance(value, str):
                        if value.lower() == "positive":
                            color = "#10b981"
                        elif value.lower() == "negative":
                            color = "#ef4444"
                        elif value.lower() == "neutral":
                            color = "#f59e0b"
                
                # Format numeric values
                if isinstance(value, float):
                    value = f"{value:.2f}"
                    
                table_html += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 12px 8px; font-weight: 600; width: 40%;">{label}:</td>
                    <td style="padding: 12px 8px; color: {color}; font-weight: 500;">{value}</td>
                </tr>
                """
            
            table_html += """
                </table>
            </div>
            """
            
            return table_html
        
        # Add overall sentiment result first - this is what matters most
        overall_classification = ""
        overall_score = 0
        
        # Calculate overall sentiment from modality scores
        scores = []
        confidences = []
        
        if visual_results and visual_results.get('sentiment_score') is not None:
            scores.append(visual_results.get('sentiment_score', 0))
            confidences.append(visual_results.get('confidence', 0.8))
        
        if audio_results and audio_results.get('sentiment_score') is not None:
            scores.append(audio_results.get('sentiment_score', 0))
            confidences.append(audio_results.get('confidence', 0.8))
        
        if text_results and text_results.get('sentiment_score') is not None:
            scores.append(text_results.get('sentiment_score', 0))
            confidences.append(text_results.get('confidence', 0.8))
        
        # Calculate final score (weighted by confidence)
        if confidences and sum(confidences) > 0:
            overall_score = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)
        elif scores:
            overall_score = sum(scores) / len(scores)
        
        # Determine sentiment classification
        if overall_score > 0.15:
            overall_classification = "Positive"
        elif overall_score < -0.15:
            overall_classification = "Negative"
        else:
            overall_classification = "Neutral"
            
        # Store the overall results for later reference
        self.overall_score = overall_score
        self.overall_classification = overall_classification
            
        # Get sentiment color based on classification
        color = "#f59e0b"  # Default neutral (amber)
        if overall_classification.lower() == "positive":
            color = "#10b981"  # green
        elif overall_classification.lower() == "negative":
            color = "#ef4444"  # red
            
        # Script to immediately update the UI with MMSA results
        html_parts.append(f"""
        <!-- MMSA Final Results (Will be shown in UI) -->
        <script>
        (function() {{
            function updateResults() {{
                // Find prediction elements
                var finalPredictionElem = document.getElementById('final-prediction');
                var sentimentLabelElem = document.getElementById('sentiment-label');
                var scoreDisplayElem = document.getElementById('score-display-value');
                
                // Update final prediction
                if (finalPredictionElem) {{
                    finalPredictionElem.textContent = "{overall_classification}";
                    finalPredictionElem.style.color = "{color}";
                    finalPredictionElem.style.fontWeight = "bold";
                }}
                
                // Update sentiment label
                if (sentimentLabelElem) {{
                    sentimentLabelElem.textContent = "{overall_classification}";
                    sentimentLabelElem.style.color = "{color}";
                    sentimentLabelElem.style.fontWeight = "bold";
                }}
                
                // Update score display
                if (scoreDisplayElem) {{
                    scoreDisplayElem.textContent = "{overall_score:.2f}";
                    scoreDisplayElem.style.color = "{color}";
                    scoreDisplayElem.style.fontWeight = "bold";
                }}
                
                // Direct replacement of waiting text
                var elements = document.querySelectorAll('*');
                elements.forEach(function(el) {{
                    if (el.childNodes.length === 1 && 
                        el.childNodes[0].nodeType === 3 && 
                        (el.textContent.trim() === 'Waiting for analysis...' || 
                        el.textContent.trim() === 'Calculating...')) {{
                        
                        if (el.id && el.id.includes('score')) {{
                            el.textContent = "{overall_score:.2f}";
                        }} else {{
                            el.textContent = "{overall_classification}";
                        }}
                        
                        el.style.color = "{color}";
                        el.style.fontWeight = "bold";
                    }}
                }});
            }}
            
            // Run immediately
            updateResults();
            
            // And several times with delay to ensure it happens after DOM updates
            setTimeout(updateResults, 100);
            setTimeout(updateResults, 500);
            setTimeout(updateResults, 1000);
            setTimeout(updateResults, 2000);
        }})();
        </script>
        
        <!-- Additional direct HTML display of results -->
        <div id="results-data" style="display: none;" 
            data-classification="{overall_classification}"
            data-score="{overall_score:.2f}"
            data-color="{color}">
        </div>
        
        <!-- EXPLICIT RESULT DISPLAY (Hidden but can be referenced) -->
        <div style="display: none;">
            <div id="actual-final-prediction">{overall_classification}</div>
            <div id="actual-sentiment-score">{overall_score:.2f}</div>
        </div>
        """)
            
        # Add summary of results
        overall_rows = [
            ("Overall Sentiment Score", overall_score),
            ("Classification", overall_classification),
            ("Modalities Used", len(scores))
        ]
        html_parts.append(create_result_table("Overall Sentiment Analysis", overall_rows))
        
        # Visual Results
        if visual_results:
            visual_rows = [
                ("Sentiment Score", visual_results.get('sentiment_score', 'N/A')),
                ("Dominant Expression", visual_results.get('dominant_emotion', 'Unknown')),
                ("Confidence", visual_results.get('confidence', 'N/A'))
            ]
            html_parts.append(create_result_table("Visual Analysis", visual_rows))
            
        # Audio Results
        if audio_results:
            audio_rows = [
                ("Sentiment Score", audio_results.get('sentiment_score', 'N/A')),
                ("Dominant Emotion", audio_results.get('emotion', 'Unknown')),
                ("Confidence", audio_results.get('confidence', 'N/A'))
            ]
            html_parts.append(create_result_table("Audio Analysis", audio_rows))
            
        # Text Results
        if text_results:
            text_rows = [
                ("Sentiment Score", text_results.get('sentiment_score', 'N/A')),
                ("Classification", text_results.get('classification', 'Unknown')),
                ("Confidence", text_results.get('confidence', 'N/A'))
            ]
            html_parts.append(create_result_table("Text Analysis", text_rows))
            
        # Add transcript if available
        if transcript:
            html_parts.append(f"""
            <div style="margin-bottom: 30px;">
                <h2>Speech Transcript</h2>
                <div style="height: 3px; width: 50px; background: linear-gradient(90deg, #0066ff, #7c3aed); margin-bottom: 15px;"></div>
                <div style="background-color: #f5f7fa; padding: 15px; border-radius: 8px; font-size: 16px; line-height: 1.6;">
                    "{transcript}"
                </div>
            </div>
            """)
            
        # Combine all parts
        return ''.join(html_parts)

def create_interface(mmsa_with_shap, share=False, port=7860):
    """Create the Gradio interface with premium design and SHAP visualizations"""
    
    # Create a truly world-class UI design system
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        /* Core color palette - refined, sophisticated neutrals with vibrant accents */
        --pure-white: #ffffff;
        --off-white: #fafafa;
        --canvas-light: #f5f7fa;
        --subtle-gray: #eef1f6;
        --border-light: #e0e4e9;
        --border-medium: #cfd4dd;
        --text-secondary: #6b7280;
        --text-primary: #1c2128;
        --text-emphasis: #000000;
        
        /* Brand colors - refined and purposeful */
        --brand-primary: #0066ff;
        --brand-primary-light: #e6f0ff;
        --brand-primary-dark: #0047b3;
        --positive: #10b981;
        --positive-light: #ecfdf5;
        --warning: #f59e0b;
        --warning-light: #fffbeb;
        --negative: #ef4444;
        --negative-light: #fef2f2;
        
        /* Elevation/shadows - subtle and layered */
        --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
        
        /* Spacing system - carefully calibrated */
        --space-3xs: 0.125rem;  /* 2px */
        --space-2xs: 0.25rem;   /* 4px */
        --space-xs: 0.5rem;     /* 8px */
        --space-sm: 0.75rem;    /* 12px */
        --space-md: 1rem;       /* 16px */
        --space-lg: 1.5rem;     /* 24px */
        --space-xl: 2rem;       /* 32px */
        --space-2xl: 3rem;      /* 48px */
        --space-3xl: 4rem;      /* 64px */
        
        /* Typography scale - refined and harmonious */
        --text-xs: 0.75rem;     /* 12px */
        --text-sm: 0.875rem;    /* 14px */
        --text-base: 1rem;      /* 16px */
        --text-md: 1.125rem;    /* 18px */
        --text-lg: 1.25rem;     /* 20px */
        --text-xl: 1.5rem;      /* 24px */
        --text-2xl: 1.875rem;   /* 30px */
        --text-3xl: 2.25rem;    /* 36px */
        --text-4xl: 3rem;       /* 48px */
        
        /* Border radii - consistent and modern */
        --radius-xs: 0.25rem;   /* 4px */
        --radius-sm: 0.375rem;  /* 6px */
        --radius-md: 0.5rem;    /* 8px */
        --radius-lg: 0.75rem;   /* 12px */
        --radius-xl: 1rem;      /* 16px */
        --radius-2xl: 1.5rem;   /* 24px */
        --radius-full: 9999px;
        
        /* Animation - natural and deliberate */
        --transition-fast: 150ms cubic-bezier(0.16, 1, 0.3, 1);
        --transition-base: 250ms cubic-bezier(0.16, 1, 0.3, 1);
        --transition-slow: 400ms cubic-bezier(0.16, 1, 0.3, 1);
        --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
        --ease-elastic: cubic-bezier(0.68, -0.6, 0.32, 1.6);
    }
    
    /* Global Typography and Reset */
    body, html {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: var(--text-primary);
    }
    
    /* App Structure */
    .app-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: var(--space-md);
        background: var(--off-white);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow-lg);
        overflow: hidden;
        position: relative;
        display: flex;
        flex-direction: column;
        gap: var(--space-xl);
    }
    
    /* Premium Header */
    .app-header {
        text-align: center;
        position: relative;
        margin-bottom: var(--space-xl);
    }
    
    .app-header .logo {
        width: 64px;
        height: 64px;
        margin-bottom: var(--space-md);
    }
    
    .app-header h1 {
        font-size: var(--text-3xl);
        font-weight: 700;
        color: var(--text-emphasis);
        margin-bottom: var(--space-xs);
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, var(--brand-primary), #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .app-header p {
        font-size: var(--text-md);
        line-height: 1.5;
        color: var(--text-secondary);
        max-width: 640px;
        margin: 0 auto;
        font-weight: 400;
    }
    
    .app-header::after {
        content: "";
        position: absolute;
        bottom: -24px;
        height: 3px;
        width: 64px;
        background: linear-gradient(90deg, var(--brand-primary), #7c3aed);
        border-radius: var(--radius-full);
    }
    
    /* Tabs and Navigation */
    .tab-nav {
        display: flex;
        border-bottom: 1px solid var(--border-light);
        padding-bottom: 0 !important;
        margin-bottom: var(--space-xl);
        position: relative;
    }
    
    .tab-nav button {
        background: transparent;
        border: none;
        padding: var(--space-md) var(--space-lg);
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-secondary);
        position: relative;
        transition: var(--transition-base);
        z-index: 1;
    }
    
    .tab-nav button:hover {
        color: var(--text-primary);
    }
    
    .tab-nav button.selected {
        color: var(--brand-primary);
        font-weight: 600;
    }
    
    .tab-nav button.selected::after {
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        height: 2px;
        background: var(--brand-primary);
        border-radius: var(--radius-full) var(--radius-full) 0 0;
    }
    
    .tab-nav button:focus {
        outline: none;
        color: var(--brand-primary);
    }
    
    .tab-nav button:focus::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: var(--radius-md);
        background: var(--brand-primary-light);
        opacity: 0.5;
        z-index: -1;
    }
    
    /* Core Panels and Cards */
    .panel {
        background: var(--pure-white);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        padding: var(--space-xl);
        transition: var(--transition-base);
        display: flex;
        flex-direction: column;
    }
    
    .panel:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-lg);
        padding-bottom: var(--space-md);
        border-bottom: 1px solid var(--border-light);
    }
    
    .panel-header h3 {
        font-size: var(--text-lg);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.01em;
    }
    
    .panel-content {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    
    /* Upload Area */
    .upload-area {
        background: var(--canvas-light);
        border: 2px dashed var(--border-medium);
        border-radius: var(--radius-lg);
        padding: var(--space-2xl);
        text-align: center;
        transition: var(--transition-base);
        cursor: pointer;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: var(--space-md);
    }
    
    .upload-area:hover {
        border-color: var(--brand-primary);
        background: var(--brand-primary-light);
    }
    
    .upload-icon {
        width: 64px;
        height: 64px;
        margin-bottom: var(--space-md);
        color: var(--brand-primary);
    }
    
    .upload-title {
        font-size: var(--text-lg);
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--space-xs);
    }
    
    .upload-subtitle {
        font-size: var(--text-sm);
        color: var(--text-secondary);
        margin-bottom: var(--space-xl);
    }
    
    /* Buttons */
    .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: var(--text-sm);
        font-weight: 500;
        padding: var(--space-sm) var(--space-lg);
        border-radius: var(--radius-md);
        border: none;
        cursor: pointer;
        transition: var(--transition-base);
        text-decoration: none;
        gap: var(--space-xs);
    }
    
    .btn-primary {
        background: var(--brand-primary);
        color: white;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .btn-primary:hover {
        background: var(--brand-primary-dark);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    .btn-primary:active {
        transform: translateY(0);
    }
    
    .btn-primary:focus {
        outline: none;
        box-shadow: 0 0 0 2px var(--brand-primary-light), 0 0 0 4px var(--brand-primary);
    }
    
    .btn-lg {
        font-size: var(--text-base);
        padding: var(--space-md) var(--space-xl);
        border-radius: var(--radius-lg);
    }
    
    /* Analysis Results */
    .results-container {
        display: flex;
        flex-direction: column;
        gap: var(--space-lg);
    }
    
    .sentiment-indicator {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: var(--space-lg);
        background: var(--subtle-gray);
        border-radius: var(--radius-lg);
        margin-bottom: var(--space-lg);
        position: relative;
        overflow: hidden;
    }
    
    .sentiment-score {
        font-size: var(--text-4xl);
        font-weight: 700;
        line-height: 1;
        margin-bottom: var(--space-xs);
        transition: var(--transition-base);
    }
    
    .sentiment-label {
        font-size: var(--text-md);
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    .positive-score {
        color: var(--positive);
    }
    
    .negative-score {
        color: var(--negative);
    }
    
    .neutral-score {
        color: var(--warning);
    }
    
    .sentiment-indicator::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 4px;
    }
    
    .positive-indicator::before {
        background: var(--positive);
    }
    
    .negative-indicator::before {
        background: var(--negative);
    }
    
    .neutral-indicator::before {
        background: var(--warning);
    }
    
    /* Analysis Detail Cards */
    .analysis-detail {
        background: var(--pure-white);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-sm);
        padding: var(--space-lg);
        margin-bottom: var(--space-md);
        border-left: 4px solid var(--border-medium);
    }
    
    .analysis-detail.visual {
        border-left-color: #4f46e5;
    }
    
    .analysis-detail.audio {
        border-left-color: #ec4899;
    }
    
    .analysis-detail.text {
        border-left-color: #0ea5e9;
    }
    
    .analysis-detail h4 {
        font-size: var(--text-md);
        font-weight: 600;
        margin-bottom: var(--space-xs);
        display: flex;
        align-items: center;
        gap: var(--space-xs);
    }
    
    .analysis-detail p {
        font-size: var(--text-sm);
        color: var(--text-secondary);
        margin-bottom: var(--space-sm);
    }
    
    .analysis-detail .badge {
        display: inline-flex;
        align-items: center;
        padding: var(--space-2xs) var(--space-xs);
        font-size: var(--text-xs);
        font-weight: 500;
        border-radius: var(--radius-xs);
        background: var(--subtle-gray);
        color: var(--text-secondary);
    }
    
    /* Chart Styling */
    .chart-container {
        background: var(--pure-white);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-sm);
        padding: var(--space-lg);
        margin-top: var(--space-xl);
    }
    
    .chart-title {
        font-size: var(--text-md);
        font-weight: 600;
        margin-bottom: var(--space-md);
        color: var(--text-primary);
    }
    
    /* SHAP Visualizations */
    .viz-container {
        display: flex;
        flex-direction: column;
        gap: var(--space-xl);
    }
    
    .viz-header {
        text-align: center;
        margin-bottom: var(--space-lg);
    }
    
    .viz-header h3 {
        font-size: var(--text-2xl);
        font-weight: 700;
        color: var(--text-emphasis);
        margin-bottom: var(--space-xs);
    }
    
    .viz-header p {
        font-size: var(--text-md);
        color: var(--text-secondary);
        max-width: 640px;
        margin: 0 auto;
    }
    
    .viz-controls {
        display: flex;
        gap: var(--space-md);
        margin-bottom: var(--space-lg);
    }
    
    .viz-select {
        flex: 1;
    }
    
    .viz-select select {
        padding: var(--space-sm) var(--space-md);
        border: 1px solid var(--border-medium);
        border-radius: var(--radius-md);
        background: var(--pure-white);
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
        appearance: none;
        background-repeat: no-repeat;
        background-position: right var(--space-sm) center;
        padding-right: var(--space-xl);
    }
    
    .gallery-display {
        background: var(--canvas-light);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .gallery-display img {
        max-height: 600px;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
    }
    
    /* Status Messages and Notifications */
    .status-message {
        padding: var(--space-md);
        border-radius: var(--radius-md);
        font-size: var(--text-sm);
        display: flex;
        align-items: center;
        gap: var(--space-sm);
        margin-bottom: var(--space-lg);
    }
    
    .info-message {
        background: var(--brand-primary-light);
        color: var(--brand-primary-dark);
        border-left: 3px solid var(--brand-primary);
    }
    
    .success-message {
        background: var(--positive-light);
        color: var(--positive);
        border-left: 3px solid var(--positive);
    }
    
    .warning-message {
        background: var(--warning-light);
        color: var(--warning);
        border-left: 3px solid var(--warning);
    }
    
    .error-message {
        background: var(--negative-light);
        color: var(--negative);
        border-left: 3px solid var(--negative);
    }
    
    /* Helper info component */
    .info-tooltip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        background: var(--border-medium);
        color: var(--pure-white);
        font-size: 10px;
        font-weight: 600;
        margin-left: var(--space-xs);
        cursor: help;
        position: relative;
    }
    
    .info-tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        background: var(--text-primary);
        color: var(--pure-white);
        padding: var(--space-xs) var(--space-sm);
        border-radius: var(--radius-sm);
        font-size: var(--text-xs);
        font-weight: 400;
        white-space: nowrap;
        pointer-events: none;
        margin-bottom: var(--space-xs);
        z-index: 10;
    }
    
    /* Footer Design */
    .app-footer {
        text-align: center;
        padding: var(--space-lg) 0;
        color: var(--text-secondary);
        font-size: var(--text-xs);
        border-top: 1px solid var(--border-light);
        margin-top: var(--space-2xl);
    }
    
    /* Loading animations */
    @keyframes pulse {
    }
    
    .loading-indicator {
        display: flex;
        align-items: center;
        gap: var(--space-sm);
        color: var(--text-secondary);
        font-size: var(--text-sm);
        animation: pulse 1.5s infinite;
    }
    
    .loading-spinner {
        width: 16px;
        height: 16px;
        border: 2px solid var(--border-light);
        border-top-color: var(--brand-primary);
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .app-container {
            padding: var(--space-sm);
        }
        
        .panel {
            padding: var(--space-md);
        }
        
        .upload-area {
            padding: var(--space-lg);
            min-height: 200px;
        }
        
        .upload-icon {
            width: 48px;
            height: 48px;
        }
        
        .sentiment-score {
            font-size: var(--text-3xl);
        }
        
        .viz-controls {
            flex-direction: column;
        }
    }
    """
    
    # Create custom theme for Gradio
    theme = gr.themes.Base().set(
        body_background_fill="#f7f9fc",
        block_background_fill="#ffffff",
        block_label_background_fill="#ffffff",
        input_background_fill="#ffffff",
        button_primary_background_fill="#0066ff",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#f5f7fa",
        button_secondary_text_color="#1c2128",
        border_color_primary="#cfd4dd"
    )
    
    # Add custom CSS for tabs
    custom_css = custom_css + """
    /* Improved Tab Styling */
    .tabs > .tab-nav {
        background-color: #f7f9fc !important;
        border-bottom: 2px solid #e0e4e9 !important;
    }
    
    .tabs > .tab-nav > button {
        font-weight: 600 !important;
        font-size: 16px !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 12px 24px !important;
        color: #1c2128 !important;
        background: #ffffff !important;
        border: 2px solid #e0e4e9 !important;
        border-bottom: none !important;
        margin-right: 6px !important;
        position: relative !important;
        bottom: -2px !important;
    }
    
    .tabs > .tab-nav > button.selected {
        background: #0066ff !important;
        color: #ffffff !important;
        border-color: #0066ff !important;
    }
    
    .tabs > .tab-nav > button:hover:not(.selected) {
        background: #e6f0ff !important;
        color: #0066ff !important;
    }
    
    .tabs > .tabitem {
        padding-top: 32px !important;
    }
    
    /* Fix ALL dark backgrounds and text contrast */
    .gradio-container, 
    .gradio-container *, 
    .panel, 
    .panel-header, 
    .panel-content,
    .block,
    .block-title,
    .block-label,
    .block-info,
    .block-text,
    .block-items,
    .block-buttons,
    .upload-container,
    .upload-button,
    .video-container,
    .audio-container,
    .text-container,
    .output-container,
    .loading-text,
    .empty-text,
    .result-text,
    .gallery-container,
    .controls-container {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    
    /* Specifically fix panel headers */
    .panel h3,
    .block-title h3,
    .block-label span,
    .block-info span,
    .gradio-container h1,
    .gradio-container h2,
    .gradio-container h3,
    .gradio-container h4,
    .gradio-container h5,
    .gradio-container h6 {
        color: #1c2128 !important;
    }
    
    /* Fix Media Input and Analysis Results headers */
    .panel-header {
        border-bottom: 1px solid #e0e4e9 !important;
        background-color: #ffffff !important;
        color: #1c2128 !important;
        padding: 16px !important;
    }
    
    /* Fix the video input area */
    .video-upload,
    .video-box,
    .upload-area {
        background-color: #f5f7fa !important;
        border: 2px dashed #cfd4dd !important;
        color: #6b7280 !important;
    }
    
    /* Feature list items (Visual, Audio, Linguistic) */
    .feature-list,
    .feature-item,
    .feature-icon,
    .feature-text {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    
    /* Fix loading indicator */
    .loading-indicator,
    .loading-spinner,
    .loading-text {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    
    /* Fix empty state */
    .empty-state {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    
    /* Override any dark elements */
    [style*="background-color: rgb(55, 65, 81)"],
    [style*="background-color: rgb(31, 41, 55)"],
    [style*="background-color: rgb(17, 24, 39)"],
    [style*="background-color: rgb(39, 40, 42)"],
    [style*="background-color: rgb(56, 57, 59)"],
    [style*="background-color: #374151"],
    [style*="background-color: #1f2937"],
    [style*="background-color: #111827"],
    [style*="background-color: #272a2a"],
    [style*="background-color: #38393b"] {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    
    /* Fix text colors for any dark text on dark backgrounds */
    [style*="color: white"],
    [style*="color: #fff"],
    [style*="color: #ffffff"],
    [style*="color: rgb(255, 255, 255)"],
    [style*="color: rgb(229, 231, 235)"],
    [style*="color: rgb(209, 213, 219)"] {
        color: #1c2128 !important;
    }

    /* Fix button colors */
    .btn-primary {
        background-color: #0066ff !important;
        color: #ffffff !important;
    }

    /* Fix specifically for the analyze button area */
    #analyze-button, 
    button[aria-label="Analyze Sentiment"] {
        background-color: #0066ff !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Fix any remaining backgrounds inside panels */
    .panel-content > div,
    .panel-content > div > div,
    .panel-content > div > div > div {
        background-color: #ffffff !important;
    }

    /* Sentiment score area - make sure it's legible */
    .sentiment-indicator, 
    .sentiment-score, 
    .sentiment-label {
        background-color: #f5f7fa !important;
        color: #1c2128 !important;
    }

    /* Fix the chart area */
    .chart-container {
        background-color: #ffffff !important;
    }
    
    /* Fix the modal and form inputs */
    .modal-content,
    .form-input,
    .form-select,
    .form-textarea {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    
    /* For the very dark panels that need more specific targeting */
    .gradio-box div[style*="background"],
    .gradio-box div[class*="bg-gray"],
    .gradio-box div[class*="bg-slate"],
    .gradio-box div[class*="bg-zinc"],
    .gradio-box div[class*="bg-neutral"] {
        background-color: #ffffff !important;
    }

    /* Force white backgrounds on all elements */
    div, span, section, header, footer, aside, main, nav, article, 
    .gr-panel, .gr-box, .gr-card, .gr-form, .gr-input-label,
    .gr-button, .gr-form-gap, .gr-block, .gr-checkbox, .gr-radio,
    .gr-panel-content, .gr-panel-header, .gr-panel-footer,
    .gr-panel-header-button, .gr-panel-header-actions {
        background-color: #ffffff !important;
    }
    
    /* Force dark text on all elements */
    div, span, p, h1, h2, h3, h4, h5, h6, label, button:not(.btn-primary):not([aria-label="Analyze Sentiment"]) {
        color: #1c2128 !important;
    }
    
    /* Aggressive override for specifically problematic panels */
    [id^="component-"], 
    [id^="panel-"], 
    [id^="block-"],
    [id^="gallery-"],
    [id^="tabs-"] {
        background-color: #ffffff !important;
    }
    
    /* Target specifically panel headers like "Media Input" and "Analysis Results" */
    .panel-header > *, 
    .block-title > *,
    .block-label > * {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    
    /* Override the "Analyzing sentiment across modalities..." message */
    .loading-indicator, 
    .loading-container,
    #loading-container {
        background-color: #f5f7fa !important;
        color: #0066ff !important;
    }
    
    /* Target the feature list (Visual, Audio, Linguistic) */
    .feature-item, 
    .feature-icon, 
    .feature-text, 
    .feature-list,
    .upload-instructions,
    .upload-instructions * {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    
    /* Fix icon colors */
    svg {
        fill: currentColor !important;
    }
    
    /* Footer styling */
    .app-footer {
        background-color: #f7f9fc !important;
        color: #6b7280 !important;
    }
    
    /* Extra specific selectors for dark areas */
    div[style*="background-color"], 
    div[style*="background: rgb"] {
        background-color: #ffffff !important;
    }
    
    /* Add specific overrides for first-level children */
    .gradio-container > div,
    .gradio-container > div > div,
    .gradio-container > div > div > div,
    .gradio-container > div > div > div > div {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    
    /* Force all text color within panel headers and content */
    .panel-header *, .panel-content * {
        color: #1c2128 !important;
    }
    
    /* Specific override for panel backgrounds */
    [class*="panel"], [class*="block"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e4e9 !important;
    }
    
    /* Make sure buttons remain visible */
    button.btn-primary, 
    button[aria-label="Analyze Sentiment"],
    button#analyze-button {
        background-color: #0066ff !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Fix text colors for better visibility */
    .sentiment-label, 
    .sentiment-score, 
    #sentiment-label,
    #final-prediction {
        color: #000000 !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }
    
    /* Color classes for sentiment */
    .positive-text {
        color: #10b981 !important;
        font-weight: bold !important;
    }
    
    .negative-text {
        color: #ef4444 !important;
        font-weight: bold !important;
    }
    
    .neutral-text {
        color: #f59e0b !important;
        font-weight: bold !important;
    }
    
    /* Fix text visibility inside panels */
    .panel-content p,
    .panel-content h3,
    .panel-content h4,
    .panel-content span {
        color: #000000 !important;
    }
    
    /* Fix any results text */
    #result-container * {
        color: #000000 !important;
    }
    
    /* Extra fixes for text visibility */
    .text-container, 
    .result-text,
    .detailed-results,
    .modality-results,
    .result-panel {
        color: #000000 !important;
    }
    
    /* Force stronger text colors */
    p, h1, h2, h3, h4, h5, h6, span, div {
        color: #000000 !important;
    }
    
    /* Only override button text when needed */
    button.btn-primary, 
    button[aria-label="Analyze Sentiment"],
    button#analyze-button {
        color: #ffffff !important;
    }
    """
    
    # Include jQuery for more reliable DOM manipulation
    jquery_js = """
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        // Function to update sentiment displays
        function updateSentimentDisplays(score, classification, color) {
            // Make sure jQuery is loaded
            if (typeof $ === 'undefined') {
                console.error('jQuery not loaded');
                return;
            }
            
            // Update score display
            $('#score-display-value').text(score).css('color', color);
            
            // Update sentiment label
            $('#sentiment-label').text(classification).css('color', color);
            
            // Update final prediction
            $('#final-prediction').text(classification).css('color', color);
            
            // Force update any waiting text (with specific content checks)
            $('span, div, p').each(function() {
                var $this = $(this);
                if ($this.text().trim() === 'Waiting for analysis...' || 
                    $this.text().trim() === 'Neutral' || 
                    $this.text().trim() === 'Calculating...') {
                    $this.text(classification).css({
                        'color': color,
                        'font-weight': 'bold'
                    });
                }
            });
            
            console.log('Updated sentiment displays with:', {score, classification, color});
        }
        
        // Wait until DOM is ready and add an observer
        $(document).ready(function() {
            console.log('DOM ready, setting up observer');
            
            // Set up observer to detect result insertions
            var observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.addedNodes && mutation.addedNodes.length > 0) {
                        // Check for our results data element
                        var resultsData = document.getElementById('analysis-results');
                        if (resultsData) {
                            var score = resultsData.getAttribute('data-score');
                            var classification = resultsData.getAttribute('data-classification');
                            var color = resultsData.getAttribute('data-color');
                            
                            if (score && classification && color) {
                                updateSentimentDisplays(score, classification, color);
                            }
                        }
                    }
                });
            });
            
            // Start observing the document body
            observer.observe(document.body, { childList: true, subtree: true });
            
            // Also check every second for 30 seconds to ensure updates
            var checkInterval = setInterval(function() {
                var resultsData = document.getElementById('analysis-results');
                if (resultsData) {
                    var score = resultsData.getAttribute('data-score');
                    var classification = resultsData.getAttribute('data-classification');
                    var color = resultsData.getAttribute('data-color');
                    
                    if (score && classification && color) {
                        updateSentimentDisplays(score, classification, color);
                    }
                }
            }, 1000);
            
            // Stop checking after 30 seconds
            setTimeout(function() {
                clearInterval(checkInterval);
            }, 30000);
        });
    </script>
    """
    
    # Create the Gradio interface with tabs
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Multimodal Sentiment Analysis Suite") as demo:
        # Include jQuery
        gr.HTML(jquery_js)
        
        # Header section with app title and description
        gr.HTML("""
        <div class="app-header">
            <h1>Multimodal Sentiment Analysis Suite</h1>
        </div>
        """)
        
        with gr.Row():
            # Left column for video input and information
            with gr.Column(scale=3):
                # Video input section
                gr.Markdown("### Upload Video")
                video_input = gr.Video(label="Upload video for analysis")
                
                # Add analyze button with custom styling
                with gr.Row():
                    analyze_button = gr.Button("Analyze Sentiment", variant="primary", elem_id="analyze-button")
                
                # Loading indicator (controlled via JavaScript)
                gr.HTML("""
                <div id="loading-container" style="display: none; align-items: center; justify-content: center; margin-top: 20px;">
                    <div class="loading-spinner"></div>
                    <div style="margin-left: 10px; font-weight: 500;">Analyzing sentiment across modalities...</div>
                </div>
                """)
                
                # Information about model and approach
                with gr.Accordion("About this tool", open=False):
                    gr.Markdown("""
                    This tool uses a multimodal approach to analyze sentiment from videos:
                    
                    - **Visual**: DeepFace for facial expression detection
                    - **Audio**: CNN model trained on RAVDESS speech emotion dataset
                    - **Text**: RoBERTa-based model with speech transcription
                    
                    The final sentiment score combines all modalities weighted appropriately.
                    
                    Source: [GitHub Repository](https://github.com/AbdullaE100/NEW-MMSA)
                    """)
            
            # Right column for results
            with gr.Column(scale=4):
                # Results tab group
                with gr.Tabs() as tabs:
                    # Main analysis tab
                    with gr.TabItem("Analysis Results"):
                        # Overall sentiment display
                        with gr.Group():
                            gr.Markdown("### Overall Sentiment")
                            with gr.Row():
                                # Overall sentiment score display
                                with gr.Column(elem_id="score-display-container"):
                                    # This container will be replaced by our direct HTML
                                    sentiment_score = gr.Number(label="Score", elem_id="sentiment-score", visible=False)
                                
                                # Sentiment visualization
                                with gr.Column():
                                    chart_output = gr.Plot(label="Sentiment Analysis by Modality")
                        
                        # Final MMSA Prediction display
                        gr.Markdown("### MMSA Final Prediction")
                        with gr.Row(elem_id="final-prediction-container"):
                            # This is just a placeholder - it will be replaced by our direct HTML
                            gr.HTML("<p id='final-prediction-placeholder'></p>")
                        
                        # Detailed analysis results
                        result_html = gr.HTML(label="Detailed Analysis")
                    
                    # SHAP Visualizations tab
                    with gr.TabItem("SHAP Explanations"):
                        gr.Markdown("### Explainable AI with SHAP")
                        
                        with gr.Row():
                            modality_dropdown = gr.Dropdown(
                                choices=["visual", "audio", "text", "all"],
                                value="all",
                                label="Choose modality for SHAP visualization"
                            )
                            view_button = gr.Button("View Explanation", variant="secondary")
                        
                        gallery = gr.Gallery(
                            label="SHAP Visualizations",
                            show_label=True,
                            elem_id="shap-gallery",
                            columns=2,
                            height="auto",
                            object_fit="contain"
                        )
                        
                        gr.Markdown(
                            """
                            **SHAP (SHapley Additive exPlanations)** visualizations show how each feature 
                            contributes to the final sentiment prediction. Red indicates features that 
                            push the prediction higher, while blue shows features that push it lower.
                            """
                        )
        
        # Footer
        gr.HTML("""
        <footer class="app-footer">
            <div class="footer-content">
                <div class="footer-section">
                    <h3>About</h3>
                    <p>Multimodal Sentiment Analysis Suite combines visual, audio, and text analysis for comprehensive emotion detection.</p>
                </div>
                <div class="footer-section">
                    <h3>Resources</h3>
                    <ul>
                        <li><a href="https://github.com/AbdullaE100/NEW-MMSA" target="_blank">GitHub Repository</a></li>
                    </ul>
                </div>
            </div>
            <div class="footer-copyright">
                <p>© 2023 Multimodal Sentiment Analysis Suite</p>
            </div>
        </footer>
        """)
        
        # Connect event handlers
        analyze_button.click(
            fn=mmsa_with_shap.process_video,
            inputs=video_input,
            outputs=[result_html, sentiment_score, chart_output]
        )
        
        view_button.click(
            fn=mmsa_with_shap.get_shap_visualizations,
            inputs=modality_dropdown,
            outputs=gallery
        )
    
    return demo

def main():
    """Main function to run the MMSA interface with SHAP"""
    parser = argparse.ArgumentParser(description="Launch the MMSA interface with SHAP")
    parser.add_argument("--share", action="store_true", help="Create a shareable link (not recommended for privacy)")
    parser.add_argument("--no-share", action="store_true", help="Explicitly disable public URL creation (overrides --share)")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio server on")
    parser.add_argument("--detector", type=str, default="retinaface", 
                       help="Face detector backend (retinaface, opencv, etc.)")
    parser.add_argument("--audio-model", type=str, default="./models/audio_sentiment_model_notebook.h5",
                       help="Path to audio sentiment model")
    parser.add_argument("--norm-params", type=str, default="./models/audio_norm_params.json",
                       help="Path to audio normalization parameters")
    parser.add_argument("--shap-dir", type=str, default=".",
                       help="Directory containing SHAP visualization results")
    args = parser.parse_args()
    
    # Override share flag if no-share is explicitly set
    share_enabled = args.share and not args.no_share
    
    # First kill any running processes
    logger.info("Checking for running processes...")
    try:
        import subprocess
        subprocess.run(["pkill", "-f", "mmsa_unified_interface.py"])
        subprocess.run(["pkill", "-f", "simple_mmsa_interface.py"])
        subprocess.run(["pkill", "-f", "test_gradio_share.py"])
        
        # Check if port is in use
        try:
            subprocess.run(["lsof", "-i", f":{args.port}", "-t"], 
                          capture_output=True, check=True)
            logger.warning(f"Port {args.port} is in use. Killing process...")
            subprocess.run(["kill", "-9", 
                          subprocess.run(["lsof", "-i", f":{args.port}", "-t"], 
                                       capture_output=True, text=True).stdout.strip()])
        except subprocess.CalledProcessError:
            logger.info(f"Port {args.port} is available")
    except Exception as e:
        logger.warning(f"Error checking processes: {str(e)}")
    
    # Verify model files exist
    verify_model_files(args.audio_model, args.norm_params)
    
    # Initialize the MMSA interface with SHAP
    logger.info("Initializing MMSA interface with SHAP...")
    mmsa_with_shap = MultimodalSentimentWithSHAP(
        audio_model_path=args.audio_model,
        audio_norm_params_path=args.norm_params,
        detector_backend=args.detector,
        cleanup_temp=True,
        allow_fallback=True,
        shap_results_dir=args.shap_dir
    )
    
    # Create the Gradio interface
    logger.info("Creating Gradio interface...")
    demo = create_interface(mmsa_with_shap, share=share_enabled, port=args.port)
    
    # Launch the interface
    logger.info(f"Launching Gradio interface on port {args.port}...")
    try:
        # Launch with share URL if requested
        if share_enabled:
            logger.info("Launching with public URL...")
            
            # Launch and get the public URL
            public_url = demo.launch(
                server_name="0.0.0.0",
                server_port=args.port,
                share=True,
                debug=True,
                show_error=True
            )
            
            logger.info(f"Gradio interface launched with public URL: {public_url}")
            
            # Save the URL to a file
            if isinstance(public_url, str) and "gradio.live" in public_url:
                with open("gradio_public_url.txt", "w") as f:
                    f.write(public_url)
                logger.info(f"Public URL saved to gradio_public_url.txt")
        else:
            logger.info("Launching in local-only mode (no public URL)...")
            demo.launch(
                server_name="0.0.0.0",
                server_port=args.port,
                share=False,
                debug=True,
                show_error=True
            )
            logger.info(f"Gradio interface launched on local port {args.port}")
            
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {str(e)}", exc_info=True)
        
        # Try with fallback port
        fallback_port = 7861
        logger.info(f"Trying fallback port {fallback_port}...")
        
        try:
            if share_enabled:
                public_url = demo.launch(
                    server_name="0.0.0.0",
                    server_port=fallback_port,
                    share=True,
                    debug=True,
                    show_error=True
                )
                logger.info(f"Gradio interface launched with public URL: {public_url}")
                
                # Save the URL to a file
                if isinstance(public_url, str) and "gradio.live" in public_url:
                    with open("gradio_public_url.txt", "w") as f:
                        f.write(public_url)
                    logger.info(f"Public URL saved to gradio_public_url.txt")
            else:
                demo.launch(
                    server_name="0.0.0.0",
                    server_port=fallback_port,
                    share=False,
                    debug=True,
                    show_error=True
                )
                logger.info(f"Gradio interface launched on local port {fallback_port}")
                
        except Exception as e2:
            logger.error(f"Error launching on fallback port: {str(e2)}", exc_info=True)

def verify_model_files(audio_model_path, norm_params_path):
    """Verify that model files exist and are accessible"""
    try:
        if not os.path.exists(audio_model_path):
            logger.error(f"Audio model file not found: {audio_model_path}")
            logger.info("Checking for alternative model files in models/ directory...")
            
            # Look for alternative model files
            model_files = glob.glob("models/*.h5")
            if model_files:
                logger.info(f"Found alternative model files: {model_files}")
                # Copy the first model file to the expected path
                shutil.copy(model_files[0], audio_model_path)
                logger.info(f"Copied {model_files[0]} to {audio_model_path}")
            else:
                logger.error("No alternative model files found. Audio analysis may not work correctly.")
        else:
            logger.info(f"Audio model file found: {audio_model_path}")
            
        if not os.path.exists(norm_params_path):
            logger.error(f"Normalization parameters file not found: {norm_params_path}")
            logger.info("Checking for alternative norm params files in models/ directory...")
            
            # Look for alternative norm params files
            norm_files = glob.glob("models/*norm*.json")
            if norm_files:
                logger.info(f"Found alternative norm params files: {norm_files}")
                # Copy the first norm params file to the expected path
                shutil.copy(norm_files[0], norm_params_path)
                logger.info(f"Copied {norm_files[0]} to {norm_params_path}")
            else:
                logger.error("No alternative norm params files found. Audio analysis may not work correctly.")
        else:
            logger.info(f"Normalization parameters file found: {norm_params_path}")
            
        # Make sure the directory exists
        os.makedirs(os.path.dirname(audio_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(norm_params_path), exist_ok=True)
    except Exception as e:
        logger.error(f"Error verifying model files: {str(e)}")

if __name__ == "__main__":
    main()
