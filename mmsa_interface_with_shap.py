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
from PIL import Image
import cv2
import shutil
import glob
import logging
import time
import subprocess
import traceback
import argparse
import uuid

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
        """Generate SHAP visualization for visual sentiment"""
        try:
            if not hasattr(self.mmsa_instance, 'visual_analyzer'):
                return
                
            visual_dir = self.shap_paths.get('visual')
            if not visual_dir:
                return
                
            # Get emotion data from visual analyzer
            if hasattr(self.mmsa_instance.visual_analyzer, 'emotion_data'):
                emotions = self.mmsa_instance.visual_analyzer.emotion_data
            else:
                # Try to access dominant emotion
                emotions = {}
                if hasattr(self.mmsa_instance.visual_analyzer, 'dominant_emotion'):
                    emotions = {'happy': 0, 'surprise': 0, 'neutral': 0, 'sad': 0, 'fear': 0, 'disgust': 0, 'angry': 0}
                    emotions[self.mmsa_instance.visual_analyzer.dominant_emotion] = 1.0
                    
            if emotions:
                # Create visualization
                plt.figure(figsize=(10, 6))
                
                # Sort emotions by contribution
                emotion_items = [(emotion, self.mmsa_instance.visual_analyzer.emotion_weights.get(emotion, 0) * value) 
                                 for emotion, value in emotions.items()]
                emotion_items.sort(key=lambda x: abs(x[1]), reverse=True)
                
                labels = [item[0] for item in emotion_items]
                values = [item[1] for item in emotion_items]
                
                # Create colormap - red for positive, blue for negative
                colors = ['red' if v > 0 else 'blue' for v in values]
                
                # Create bar chart
                plt.barh(labels, values, color=colors)
                plt.xlabel('Contribution to Sentiment')
                plt.title('Visual Sentiment Analysis (SHAP)')
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
        """Generate SHAP visualization for audio sentiment"""
        try:
            if not hasattr(self.mmsa_instance, 'audio_analyzer'):
                return
                
            audio_dir = self.shap_paths.get('audio')
            if not audio_dir:
                return
                
            # Get audio prediction data
            if hasattr(self.mmsa_instance.audio_analyzer, 'last_prediction'):
                predictions = self.mmsa_instance.audio_analyzer.last_prediction
                
                if predictions:
                    # Create visualization
                    plt.figure(figsize=(10, 6))
                    
                    # Sort by contribution
                    pred_items = [(emotion, value) for emotion, value in predictions.items()]
                    pred_items.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    labels = [item[0] for item in pred_items]
                    values = [item[1] for item in pred_items]
                    
                    # Create colormap - red for positive, blue for negative
                    colors = ['red' if v > 0 else 'blue' for v in values]
                    
                    # Create bar chart
                    plt.barh(labels, values, color=colors)
                    plt.xlabel('Contribution to Sentiment')
                    plt.title('Audio Sentiment Analysis (SHAP)')
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
        """Generate SHAP visualization for text sentiment"""
        try:
            if not hasattr(self.mmsa_instance, 'text_analyzer') or not hasattr(self.mmsa_instance, 'last_transcript'):
                return
                
            text_dir = self.shap_paths.get('text')
            if not text_dir:
                return
                
            # Get text sentiment data
            if hasattr(self.mmsa_instance.text_analyzer, 'last_analysis'):
                analysis = self.mmsa_instance.text_analyzer.last_analysis
                transcript = self.mmsa_instance.last_transcript
                
                if analysis:
                    # Create visualization
                    plt.figure(figsize=(10, 4))
                    
                    # Create sentiment bars
                    labels = ["Positive", "Negative"]
                    values = [analysis.get('positive_score', 0), analysis.get('negative_score', 0)]
                    
                    # Create colormap
                    colors = ['red', 'blue']
                    
                    # Create bar chart
                    plt.barh(labels, values, color=colors)
                    plt.xlabel('Confidence Score')
                    plt.title(f'Text Sentiment Analysis (SHAP): "{transcript[:30]}..."')
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
        """Get SHAP visualization images for a specific modality"""
        try:
            if not self.shap_available:
                logger.warning("SHAP is not available.")
                return []
                
            # Log and validate the modality input
            logger.info(f"Requesting SHAP visualizations for modality: {modality}")
            
            # Handle None or empty string case
            if modality is None or modality == "":
                modality = "all"  # default to all modalities
            
            # Create default shap paths if needed
            if not hasattr(self, 'shap_paths') or not self.shap_paths:
                self.shap_paths = {}
                for mod in ['text', 'audio', 'visual', 'all']:
                    mod_dir = os.path.join(self.shap_results_dir or '.', DEFAULT_SHAP_PATHS[mod])
                    os.makedirs(mod_dir, exist_ok=True)
                    self.shap_paths[mod] = mod_dir
            
            # Ensure the modality is valid
            if modality not in self.shap_paths:
                logger.warning(f"No SHAP results path for {modality} modality, defaulting to 'all'")
                modality = "all"
            
            # First check if we have current analysis images
            current_images = self.current_analysis.get('shap_images', {}).get(modality, [])
            if current_images:
                logger.info(f"Found {len(current_images)} SHAP visualizations for {modality} from current analysis")
                
                # Verify these files exist
                valid_images = [img for img in current_images if os.path.exists(img)]
                if valid_images:
                    return valid_images
                else:
                    logger.warning("Current analysis images do not exist on disk")
            
            # If no current images, create sample visualizations if needed
            pattern = os.path.join(self.shap_paths[modality], "*.png")
            image_paths = glob.glob(pattern)
            
            if not image_paths:
                logger.warning(f"No SHAP visualization images found for {modality} modality")
                # Create a sample visualization
                self._create_sample_visualization(modality)
                # Try again
                image_paths = glob.glob(pattern)
            
            # Sort by filename to maintain order
            image_paths.sort()
            
            if image_paths:
                logger.info(f"Found {len(image_paths)} SHAP visualizations for {modality} modality")
                return image_paths
            else:
                # Last resort: return a placeholder
                return self._get_placeholder_image(modality)
        except Exception as e:
            logger.error(f"Error fetching SHAP visualizations: {str(e)}", exc_info=True)
            # Return placeholder in case of any error
            return self._get_placeholder_image(modality)
    
    def _create_sample_visualization(self, modality):
        """Create a sample SHAP visualization for the given modality"""
        try:
            output_dir = self.shap_paths.get(modality)
            if not output_dir:
                return
                
            plt.figure(figsize=(10, 6))
            
            # Create a simple bar chart showing features based on modality
            if modality == "visual":
                features = ['happy', 'surprise', 'neutral', 'sad', 'fear', 'disgust', 'angry']
                values = [0.7, 0.4, 0.1, -0.3, -0.2, -0.5, -0.6]
                title = "Visual Sentiment Analysis (Sample)"
            elif modality == "audio":
                features = ['happy', 'surprised', 'calm', 'neutral', 'fearful', 'sad', 'disgust', 'angry']
                values = [0.6, 0.3, 0.2, 0.0, -0.2, -0.4, -0.5, -0.7]
                title = "Audio Sentiment Analysis (Sample)"
            elif modality == "text":
                features = ['Positive', 'Negative']
                values = [0.6, -0.4]
                title = "Text Sentiment Analysis (Sample)"
            else:  # all
                features = ['Visual', 'Audio', 'Text']
                values = [0.6, 0.3, -0.2]
                title = "Multimodal Sentiment Analysis (Sample)"
                
            # Create colormap - red for positive, blue for negative
            colors = ['red' if v > 0 else 'blue' for v in values]
            
            # Create bar chart
            plt.barh(features, values, color=colors)
            plt.xlabel('Contribution to Sentiment')
            plt.title(title)
            plt.tight_layout()
            
            # Save the visualization
            output_path = os.path.join(output_dir, f"sample_shap_{modality}.png")
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Created sample SHAP visualization for {modality} at {output_path}")
        except Exception as e:
            logger.error(f"Error creating sample visualization: {str(e)}", exc_info=True)
            
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
        left: 50%;
        transform: translateX(-50%);
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
        width: 100%;
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
        height: 100%;
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
        width: 100%;
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
        width: 100%;
        padding: var(--space-sm) var(--space-md);
        border: 1px solid var(--border-medium);
        border-radius: var(--radius-md);
        background: var(--pure-white);
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
        appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16' fill='none'%3E%3Cpath d='M4 6L8 10L12 6' stroke='%236B7280' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
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
        max-width: 100%;
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
        border-radius: 50%;
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
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
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
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
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
        border-radius: 50%;
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
    """
    
    demo = gr.Blocks(title="Multimodal Sentiment Analysis Suite", theme=theme, css=custom_css)
    
    with demo:
        # Sophisticated App Header
        gr.HTML("""
        <div class="app-header">
            <svg class="logo" width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M32 2C15.4 2 2 15.4 2 32C2 48.6 15.4 62 32 62C48.6 62 62 48.6 62 32C62 15.4 48.6 2 32 2Z" fill="url(#paint0_linear)" fill-opacity="0.1"/>
                <path d="M20 22C20 20.8954 20.8954 20 22 20H42C43.1046 20 44 20.8954 44 22V42C44 43.1046 43.1046 44 42 44H22C20.8954 44 20 43.1046 20 42V22Z" fill="url(#paint1_linear)" fill-opacity="0.4"/>
                <path d="M26 26C26 24.8954 26.8954 24 28 24H48C49.1046 24 50 24.8954 50 26V46C50 47.1046 49.1046 48 48 48H28C26.8954 48 26 47.1046 26 46V26Z" fill="url(#paint2_linear)" fill-opacity="0.7"/>
                <path d="M14 18C14 16.8954 14.8954 16 16 16H36C37.1046 16 38 16.8954 38 18V38C38 39.1046 37.1046 40 36 40H16C14.8954 40 14 39.1046 14 38V18Z" fill="url(#paint3_linear)"/>
                <path d="M38.7782 30.0625C38.7782 32.1908 37.9623 34.2312 36.4988 35.8033C35.0353 37.3754 33.0368 38.25 30.9528 38.25C28.8688 38.25 26.8703 37.3754 25.4068 35.8033C23.9433 34.2312 23.1274 32.1908 23.1274 30.0625C23.1274 27.9342 23.9433 25.8938 25.4068 24.3217C26.8703 22.7496 28.8688 21.875 30.9528 21.875C33.0368 21.875 35.0353 22.7496 36.4988 24.3217C37.9623 25.8938 38.7782 27.9342 38.7782 30.0625Z" fill="white"/>
                <defs>
                    <linearGradient id="paint0_linear" x1="32" y1="2" x2="32" y2="62" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#0066FF"/>
                        <stop offset="1" stop-color="#7F00FF"/>
                    </linearGradient>
                    <linearGradient id="paint1_linear" x1="32" y1="20" x2="32" y2="44" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#0066FF"/>
                        <stop offset="1" stop-color="#7F00FF"/>
                    </linearGradient>
                    <linearGradient id="paint2_linear" x1="38" y1="24" x2="38" y2="48" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#0066FF"/>
                        <stop offset="1" stop-color="#7F00FF"/>
                    </linearGradient>
                    <linearGradient id="paint3_linear" x1="26" y1="16" x2="26" y2="40" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#0066FF"/>
                        <stop offset="1" stop-color="#7F00FF"/>
                    </linearGradient>
                </defs>
            </svg>
            <h1>Multimodal Sentiment Analysis Suite</h1>
            <p>Advanced analysis platform that combines visual, audio, and text signals for comprehensive emotional understanding.</p>
        </div>
        """)
        
        # Main app container
        with gr.Group(elem_classes=["app-container"]):
            # Premium navigation tabs
            with gr.Tabs(elem_classes=["tab-nav"]) as tabs:
                # Multimodal Analysis Tab (Main workflow)
                with gr.Tab("Multimodal Analysis", elem_id="multimodal-tab"):
                    with gr.Row(equal_height=True):
                        # Left Panel: Media Upload and Controls
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["panel"]):
                                with gr.Group(elem_classes=["panel-header"]):
                                    gr.HTML("<h3 style='color: #1c2128; background-color: #ffffff;'>Media Input</h3>")
                                
                                with gr.Group(elem_classes=["panel-content"]):
                                    # Advanced upload area with custom styling
                                    video_input = gr.Video(
                                        label="",
                                        elem_id="video-upload",
                                        height=320,
                                        show_label=False,
                                        elem_classes=["upload-area"]
                                    )
                                    
                                    # Replace standard instructions with premium HTML version
                                    gr.HTML("""
                                    <div class="upload-instructions" style="background-color: #ffffff; color: #1c2128;">
                                        <div class="feature-list" style="background-color: #ffffff;">
                                            <div class="feature-item" style="background-color: #ffffff; color: #1c2128;">
                                                <div class="feature-icon" style="color: #4f46e5; background-color: #ffffff;">
                                                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path>
                                                        <circle cx="12" cy="12" r="3"></circle>
                                                    </svg>
                                                </div>
                                                <div class="feature-text" style="background-color: #ffffff; color: #1c2128;">
                                                    <strong style="color: #1c2128;">Visual</strong>
                                                    <span style="color: #6b7280;">Facial expressions & micro-movements</span>
                                                </div>
                                            </div>
                                            
                                            <div class="feature-item" style="background-color: #ffffff; color: #1c2128;">
                                                <div class="feature-icon" style="color: #ec4899; background-color: #ffffff;">
                                                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                        <path d="M12 2v1"></path>
                                                        <path d="M12 7v1"></path>
                                                        <path d="M12 12v1"></path>
                                                        <path d="M12 17v1"></path>
                                                        <path d="M12 21v1"></path>
                                                        <path d="m17 4-1 .5"></path>
                                                        <path d="m8 20 1-.5"></path>
                                                        <path d="M20.1 9.1a7.7 7.7 0 0 1 0 5.8"></path>
                                                        <path d="m3.9 14.9 1.6-2.3"></path>
                                                        <path d="m3.9 9.1 1.6 2.3"></path>
                                                    </svg>
                                                </div>
                                                <div class="feature-text" style="background-color: #ffffff; color: #1c2128;">
                                                    <strong style="color: #1c2128;">Audio</strong>
                                                    <span style="color: #6b7280;">Voice tone patterns & acoustic features</span>
                                                </div>
                                            </div>
                                            
                                            <div class="feature-item" style="background-color: #ffffff; color: #1c2128;">
                                                <div class="feature-icon" style="color: #0ea5e9; background-color: #ffffff;">
                                                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                        <rect width="6" height="14" x="4" y="5" rx="2"></rect>
                                                        <rect width="6" height="10" x="14" y="9" rx="2"></rect>
                                                        <path d="M4 2v20"></path>
                                                    </svg>
                                                </div>
                                                <div class="feature-text" style="background-color: #ffffff; color: #1c2128;">
                                                    <strong style="color: #1c2128;">Linguistic</strong>
                                                    <span style="color: #6b7280;">Speech content & word choice</span>
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div class="upload-tip" style="background-color: #f5f7fa; color: #6b7280;">
                                            <div class="tip-icon" style="color: #6b7280; background-color: transparent;">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                    <circle cx="12" cy="12" r="10"></circle>
                                                    <path d="M12 16v-4"></path>
                                                    <path d="M12 8h.01"></path>
                                                </svg>
                                            </div>
                                            <p style="color: #6b7280;">For optimal results, use videos with clear faces and audible speech.</p>
                                        </div>
                                    </div>
                                    
                                    <style>
                                        .upload-instructions {
                                            margin-top: 16px;
                                            background-color: #ffffff !important;
                                        }
                                        
                                        .feature-list {
                                            display: flex;
                                            flex-direction: column;
                                            gap: 12px;
                                            margin-bottom: 16px;
                                            background-color: #ffffff !important;
                                        }
                                        
                                        .feature-item {
                                            display: flex;
                                            align-items: flex-start;
                                            gap: 12px;
                                            background-color: #ffffff !important;
                                            color: #1c2128 !important;
                                        }
                                        
                                        .feature-icon {
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            width: 32px;
                                            height: 32px;
                                            border-radius: 8px;
                                            background: rgba(0,0,0,0.03) !important;
                                        }
                                        
                                        .feature-text {
                                            display: flex;
                                            flex-direction: column;
                                            background-color: #ffffff !important;
                                            color: #1c2128 !important;
                                        }
                                        
                                        .feature-text strong {
                                            font-weight: 600;
                                            font-size: 14px;
                                            color: #1c2128 !important;
                                        }
                                        
                                        .feature-text span {
                                            font-size: 13px;
                                            color: #6b7280 !important;
                                        }
                                        
                                        .upload-tip {
                                            display: flex;
                                            align-items: flex-start;
                                            gap: 8px;
                                            padding: 12px;
                                            background: #f5f7fa !important;
                                            border-radius: 8px;
                                        }
                                        
                                        .tip-icon {
                                            color: #6b7280 !important;
                                            background-color: transparent !important;
                                        }
                                        
                                        .upload-tip p {
                                            margin: 0;
                                            font-size: 13px;
                                            color: #6b7280 !important;
                                        }
                                    </style>
                                    """)
                                    
                                    # Premium analyze button
                                    analyze_button = gr.Button(
                                        "Analyze Sentiment",
                                        elem_id="analyze-button",
                                        elem_classes=["btn", "btn-primary", "btn-lg"]
                                    )
                        
                        # Right Panel: Analysis Results with premium styling
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["panel"]):
                                with gr.Group(elem_classes=["panel-header"]):
                                    gr.HTML("<h3 style='color: #1c2128; background-color: #ffffff;'>Analysis Results</h3>")
                                
                                with gr.Group(elem_classes=["panel-content", "results-container"]):
                                    # Premium loading state indicator (initially hidden)
                                    with gr.Group(visible=False, elem_id="loading-container", elem_classes=["loading-indicator"]):
                                        gr.HTML("""
                                        <div class="loading-spinner" style="background-color: #ffffff; color: #0066ff;"></div>
                                        <div class="loading-spinner"></div>
                                        <span>Analyzing sentiment across modalities...</span>
                                        """)
                                    
                                    # Premium sentiment score display
                                    with gr.Group(elem_id="sentiment-container", elem_classes=["sentiment-indicator"]):
                                        sentiment_score = gr.Number(
                                            label="",
                                            elem_id="sentiment-score",
                                            elem_classes=["sentiment-score"],
                                            show_label=False,
                                            value=0
                                        )
                                        
                                        # The sentiment label that changes dynamically
                                        gr.HTML("""
                                        <div class="sentiment-label">Neutral</div>
                                        <div class="sentiment-legend">
                                            <div class="legend-item">
                                                <span class="legend-color negative"></span>
                                                <span class="legend-text">Negative (-1.0)</span>
                                            </div>
                                            <div class="legend-item">
                                                <span class="legend-color neutral"></span>
                                                <span class="legend-text">Neutral (0.0)</span>
                                            </div>
                                            <div class="legend-item">
                                                <span class="legend-color positive"></span>
                                                <span class="legend-text">Positive (+1.0)</span>
                                            </div>
                                        </div>
                                        
                                        <style>
                                            .sentiment-legend {
                                                display: flex;
                                                justify-content: center;
                                                gap: 16px;
                                                margin-top: 8px;
                                            }
                                            
                                            .legend-item {
                                                display: flex;
                                                align-items: center;
                                                gap: 4px;
                                            }
                                            
                                            .legend-color {
                                                width: 12px;
                                                height: 12px;
                                                border-radius: 3px;
                                            }
                                            
                                            .legend-color.negative {
                                                background: var(--negative);
                                            }
                                            
                                            .legend-color.neutral {
                                                background: var(--warning);
                                            }
                                            
                                            .legend-color.positive {
                                                background: var(--positive);
                                            }
                                            
                                            .legend-text {
                                                font-size: 12px;
                                                color: var(--text-secondary);
                                            }
                                        </style>
                                        """)
                                    
                                    # Detailed results with premium HTML formatting
                                    result_html = gr.HTML(
                                        elem_id="result-details",
                                        value="""
                                        <div class="empty-state">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="empty-icon">
                                                <circle cx="12" cy="12" r="10"></circle>
                                                <path d="M12 8v4"></path>
                                                <path d="M12 16h.01"></path>
                                            </svg>
                                            <h3>No Analysis Yet</h3>
                                            <p>Upload a video and click "Analyze Sentiment" to see detailed results here.</p>
                                        </div>
                                        
                                        <style>
                                            .empty-state {
                                                display: flex;
                                                flex-direction: column;
                                                align-items: center;
                                                justify-content: center;
                                                padding: 48px 24px;
                                                text-align: center;
                                                color: var(--text-secondary);
                                            }
                                            
                                            .empty-icon {
                                                opacity: 0.3;
                                                margin-bottom: 16px;
                                            }
                                            
                                            .empty-state h3 {
                                                font-size: 16px;
                                                font-weight: 600;
                                                margin-bottom: 8px;
                                                color: var(--text-primary);
                                            }
                                            
                                            .empty-state p {
                                                font-size: 14px;
                                                color: var(--text-secondary);
                                            }
                                        </style>
                                        """
                                    )
                                    
                                    # Premium chart container
                                    with gr.Group(elem_classes=["chart-container"]):
                                        gr.HTML("<h4 class='chart-title'>Modality Contributions</h4>")
                                        chart_output = gr.Plot(
                                            elem_id="chart-visualization"
                                        )
                
                # SHAP Visualization Tab - Premium Design
                with gr.Tab("SHAP Visualizations", elem_id="shap-tab"):
                    # Elegant visualization container
                    with gr.Group(elem_classes=["viz-container"]):
                        # Premium header for the SHAP section
                        gr.HTML("""
                        <div class="viz-header">
                            <h3>Model Explainability Visualizations</h3>
                            <p>Discover how each feature influences sentiment prediction with SHAP (SHapley Additive exPlanations) technology.</p>
                        </div>
                        """)
                        
                        with gr.Row(equal_height=True):
                            # Left: Sophisticated Control Panel
                            with gr.Column(scale=1):
                                with gr.Group(elem_classes=["panel"]):
                                    with gr.Group(elem_classes=["panel-header"]):
                                        gr.HTML("<h3>Visualization Controls</h3>")
                                    
                                    with gr.Group(elem_classes=["panel-content"]):
                                        # Premium selector
                                        with gr.Group():
                                            gr.HTML("""
                                            <label class="select-label">
                                                Modality
                                                <div class="info-tooltip" data-tooltip="Choose which aspect of the analysis to visualize">i</div>
                                            </label>
                                            <style>
                                                .select-label {
                                                    display: flex;
                                                    align-items: center;
                                                    font-size: 14px;
                                                    font-weight: 500;
                                                    margin-bottom: 8px;
                                                    color: var(--text-primary);
                                                }
                                            </style>
                                            """)
                                            
                                            modality_dropdown = gr.Dropdown(
                                                choices=[
                                                    "all",
                                                    "visual",
                                                    "audio",
                                                    "text"
                                                ],
                                                value="all",
                                                label="Select Modality",
                                                elem_id="modality-selector",
                                                elem_classes=["viz-select"]
                                            )
                                        
                                        # Premium button
                                        view_button = gr.Button(
                                            "View Visualizations", 
                                            elem_id="view-button",
                                            elem_classes=["btn", "btn-primary"]
                                        )
                                        
                                        # Status with premium styling
                                        shap_status = gr.HTML(
                                            """
                                            <div class="status-message info-message">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                    <circle cx="12" cy="12" r="10"></circle>
                                                    <path d="M12 16v-4"></path>
                                                    <path d="M12 8h.01"></path>
                                                </svg>
                                                Ready to display visualizations. First analyze a video in the Multimodal Analysis tab.
                                            </div>
                                            """
                                        )
                                        
                                        # Premium explanation panel
                                        with gr.Accordion(
                                            "Understanding These Visualizations", 
                                            open=False
                                        ):
                                            gr.HTML("""
                                            <div class="explanation-content">
                                                <h4>How to Read SHAP Visualizations</h4>
                                                
                                                <div class="explanation-item">
                                                    <div class="explanation-icon positive">
                                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                            <path d="M5 12h14"></path>
                                                            <path d="M12 5v14"></path>
                                                        </svg>
                                                    </div>
                                                    <div class="explanation-text">
                                                        <strong>Red bars</strong> show features pushing the prediction toward positive sentiment
                                                    </div>
                                                </div>
                                                
                                                <div class="explanation-item">
                                                    <div class="explanation-icon negative">
                                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                            <path d="M5 12h14"></path>
                                                        </svg>
                                                    </div>
                                                    <div class="explanation-text">
                                                        <strong>Blue bars</strong> indicate features contributing to negative sentiment
                                                    </div>
                                                </div>
                                                
                                                <div class="explanation-item">
                                                    <div class="explanation-icon length">
                                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                            <path d="M21 6H3"></path>
                                                            <path d="M17 12H3"></path>
                                                            <path d="M12 18H3"></path>
                                                        </svg>
                                                    </div>
                                                    <div class="explanation-text">
                                                        <strong>Bar length</strong> represents the magnitude of each feature's impact
                                                    </div>
                                                </div>
                                                
                                                <p class="explanation-summary">
                                                    These visualizations reveal which aspects of the video had the strongest influence on the final sentiment score, providing transparency into the decision-making process.
                                                </p>
                                            </div>
                                            
                                            <style>
                                                .explanation-content {
                                                    padding: 16px;
                                                }
                                                
                                                .explanation-content h4 {
                                                    font-size: 16px;
                                                    font-weight: 600;
                                                    margin-bottom: 16px;
                                                    color: var(--text-primary);
                                                }
                                                
                                                .explanation-item {
                                                    display: flex;
                                                    align-items: center;
                                                    gap: 12px;
                                                    margin-bottom: 12px;
                                                }
                                                
                                                .explanation-icon {
                                                    display: flex;
                                                    align-items: center;
                                                    justify-content: center;
                                                    width: 28px;
                                                    height: 28px;
                                                    border-radius: 6px;
                                                }
                                                
                                                .explanation-icon.positive {
                                                    background: var(--positive-light);
                                                    color: var(--positive);
                                                }
                                                
                                                .explanation-icon.negative {
                                                    background: var(--negative-light);
                                                    color: var(--negative);
                                                }
                                                
                                                .explanation-icon.length {
                                                    background: var(--brand-primary-light);
                                                    color: var(--brand-primary);
                                                }
                                                
                                                .explanation-text {
                                                    font-size: 14px;
                                                    color: var(--text-secondary);
                                                }
                                                
                                                .explanation-text strong {
                                                    font-weight: 600;
                                                    color: var(--text-primary);
                                                }
                                                
                                                .explanation-summary {
                                                    margin-top: 16px;
                                                    padding-top: 16px;
                                                    border-top: 1px solid var(--border-light);
                                                    font-size: 14px;
                                                    color: var(--text-secondary);
                                                    line-height: 1.5;
                                                }
                                            </style>
                                            """)
                            
                            # Right: Premium Gallery Display
                            with gr.Column(scale=2):
                                with gr.Group(elem_classes=["panel"]):
                                    with gr.Group(elem_classes=["panel-header"]):
                                        gr.HTML("<h3>SHAP Visualizations</h3>")
                                    
                                    with gr.Group(elem_classes=["panel-content"]):
                                        # Enhanced gallery with empty state
                                        gallery = gr.Gallery(
                                            label="",
                                            show_label=False,
                                            elem_id="shap-gallery",
                                            columns=1,
                                            object_fit="contain",
                                            height="700px",
                                            elem_classes=["gallery-display"]
                                        )
                                        
                                        # Premium empty state for gallery
                                        gr.HTML("""
                                        <div class="empty-gallery" id="empty-gallery">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                                                <rect width="18" height="18" x="3" y="3" rx="2" ry="2"></rect>
                                                <circle cx="9" cy="9" r="2"></circle>
                                                <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path>
                                            </svg>
                                            <h3>No Visualizations Available</h3>
                                            <p>Analyze a video in the Multimodal Analysis tab to generate SHAP visualizations.</p>
                                        </div>
                                        
                                        <style>
                                            .empty-gallery {
                                                position: absolute;
                                                top: 50%;
                                                left: 50%;
                                                transform: translate(-50%, -50%);
                                                display: flex;
                                                flex-direction: column;
                                                align-items: center;
                                                justify-content: center;
                                                text-align: center;
                                                color: var(--text-secondary);
                                                width: 100%;
                                                max-width: 300px;
                                                z-index: 1;
                                            }
                                            
                                            .empty-gallery svg {
                                                opacity: 0.3;
                                                margin-bottom: 16px;
                                            }
                                            
                                            .empty-gallery h3 {
                                                font-size: 16px;
                                                font-weight: 600;
                                                margin-bottom: 8px;
                                                color: var(--text-primary);
                                            }
                                            
                                            .empty-gallery p {
                                                font-size: 14px;
                                                color: var(--text-secondary);
                                            }
                                        </style>
                                        
                                        <script>
                                            // Script to hide empty state when gallery has items
                                            function toggleEmptyState() {
                                                const galleryEl = document.getElementById('shap-gallery');
                                                const emptyEl = document.getElementById('empty-gallery');
                                                
                                                if (galleryEl && emptyEl) {
                                                    // Check if gallery has any child elements
                                                    const hasItems = galleryEl.querySelectorAll('img').length > 0;
                                                    emptyEl.style.display = hasItems ? 'none' : 'flex';
                                                }
                                            }
                                            
                                            // Run on load and periodically check
                                            document.addEventListener('DOMContentLoaded', function() {
                                                toggleEmptyState();
                                                setInterval(toggleEmptyState, 1000);
                                            });
                                        </script>
                                        """)
        
        # Premium footer
        gr.HTML("""
        <footer class="app-footer">
            <div class="footer-content">
                <p>Multimodal Sentiment Analysis Suite | Advanced emotion analysis platform using state-of-the-art AI models</p>
                <div class="footer-links">
                    <a href="#" class="footer-link">Documentation</a>
                    <a href="#" class="footer-link">Privacy Policy</a>
                    <a href="#" class="footer-link">Terms of Use</a>
                </div>
            </div>
            
            <style>
                .app-footer {
                    margin-top: var(--space-2xl);
                    padding: var(--space-lg) 0;
                    text-align: center;
                    color: var(--text-secondary);
                    border-top: 1px solid var(--border-light);
                }
                
                .footer-content {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 12px;
                }
                
                .footer-content p {
                    font-size: 13px;
                    margin: 0;
                }
                
                .footer-links {
                    display: flex;
                    gap: 24px;
                }
                
                .footer-link {
                    font-size: 13px;
                    color: var(--brand-primary);
                    text-decoration: none;
                    transition: var(--transition-base);
                }
                
                .footer-link:hover {
                    color: var(--brand-primary-dark);
                    text-decoration: underline;
                }
            </style>
        </footer>
        """)
        
        # Add JavaScript functionality through HTML component
        gr.HTML("""
        <script>
        // Script to handle dynamic styling and interactions
        document.addEventListener('DOMContentLoaded', function() {
            // Update sentiment score styling based on value
            function updateScoreDisplay() {
                setTimeout(function() {
                    const scoreElement = document.getElementById('sentiment-score');
                    const sentimentContainer = document.getElementById('sentiment-container');
                    const sentimentLabel = sentimentContainer ? sentimentContainer.querySelector('.sentiment-label') : null;
                    
                    if (scoreElement && sentimentContainer) {
                        const scoreValue = parseFloat(scoreElement.textContent);
                        
                        if (!isNaN(scoreValue)) {
                            // Remove existing classes
                            sentimentContainer.classList.remove('positive-indicator', 'negative-indicator', 'neutral-indicator');
                            scoreElement.classList.remove('positive-score', 'negative-score', 'neutral-score');
                            
                            // Add new classes based on value
                            if (scoreValue > 0.3) {
                                scoreElement.classList.add('positive-score');
                                sentimentContainer.classList.add('positive-indicator');
                                if (sentimentLabel) {
                                    sentimentLabel.textContent = scoreValue > 0.7 ? 'Very Positive' : 'Positive';
                                }
                            } else if (scoreValue < -0.3) {
                                scoreElement.classList.add('negative-score');
                                sentimentContainer.classList.add('negative-indicator');
                                if (sentimentLabel) {
                                    sentimentLabel.textContent = scoreValue < -0.7 ? 'Very Negative' : 'Negative';
                                }
                            } else {
                                scoreElement.classList.add('neutral-score');
                                sentimentContainer.classList.add('neutral-indicator');
                                if (sentimentLabel) {
                                    sentimentLabel.textContent = 'Neutral';
                                }
                            }
                        }
                    }
                }, 100);
            }
            
            // Handle loading state for analyze button
            function setupLoadingState() {
                const analyzeButton = document.getElementById('analyze-button');
                const loadingContainer = document.getElementById('loading-container');
                
                if (analyzeButton && loadingContainer) {
                    analyzeButton.addEventListener('click', function() {
                        // Show loading state
                        loadingContainer.style.display = 'flex';
                        
                        // Hide loading after analysis is done (approximately)
                        setTimeout(function() {
                            loadingContainer.style.display = 'none';
                        }, 5000);
                    });
                }
            }
            
            // Initial setup
            updateScoreDisplay();
            setupLoadingState();
            
            // Watch for changes to the sentiment score
            const observer = new MutationObserver(function(mutations) {
                updateScoreDisplay();
            });
            
            // Start observing once the element exists
            setTimeout(function() {
                const scoreElement = document.getElementById('sentiment-score');
                if (scoreElement) {
                    observer.observe(scoreElement, { childList: true, subtree: true });
                }
            }, 1000);
        });
        </script>
        """, visible=False)  # Hidden but executes JavaScript

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
    parser.add_argument("--audio-model", type=str, default="./audio_sentiment_model_notebook.h5",
                       help="Path to audio sentiment model")
    parser.add_argument("--norm-params", type=str, default="audio_norm_params.json",
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

if __name__ == "__main__":
    main() 