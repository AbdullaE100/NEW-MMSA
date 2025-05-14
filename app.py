#!/usr/bin/env python3

"""
MMSA Hugging Face Spaces Entry Point
This file is used by Hugging Face Spaces to run the MMSA application.
"""

import os
import sys
import gradio as gr
import logging
import torch
import numpy as np
from PIL import Image
import tempfile
import librosa
import cv2
import time
import json
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mmsa_hf_spaces.log')
    ]
)
logger = logging.getLogger('mmsa_hf_spaces')

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create necessary directories
for dir_name in ['shap_results_text', 'shap_results_audio', 'shap_results_visual', 'shap_results_all', 'gradio_results']:
    os.makedirs(dir_name, exist_ok=True)

# Add paths to your MMSA model
sys.path.append(os.getcwd())
try:
    from mmsa_interface_with_shap import Interface
except ImportError:
    # If running in Hugging Face Space and model is in different location
    sys.path.append('MMSA-model')
    from mmsa_interface_with_shap import Interface

# Initialize the MMSA Interface
try:
    mmsa = Interface()
except Exception as e:
    print(f"Error initializing MMSA interface: {e}")
    mmsa = None

def analyze_sentiment(text_input=None, audio_file=None, video_file=None):
    """Process multimodal inputs and return sentiment analysis results"""
    results = {}
    error = None
    
    try:
        # Process text if provided
        if text_input and len(text_input.strip()) > 0:
            text_result = mmsa.predict_text(text_input)
            results['text'] = text_result
        
        # Process audio if provided
        if audio_file is not None:
            audio_path = audio_file.name if hasattr(audio_file, 'name') else audio_file
            audio_result = mmsa.predict_audio(audio_path)
            results['audio'] = audio_result
            
        # Process video if provided
        if video_file is not None:
            video_path = video_file.name if hasattr(video_file, 'name') else video_file
            video_result = mmsa.predict_video(video_path)
            results['video'] = video_result
            
        # Return formatted results
        if results:
            output_html = "<h3>Sentiment Analysis Results</h3>"
            
            for modality, result in results.items():
                output_html += f"<h4>{modality.capitalize()} Analysis:</h4>"
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, (np.ndarray, np.float32, np.int64)):
                            v = float(v)
                        output_html += f"<p><b>{k}:</b> {v}</p>"
                else:
                    output_html += f"<p>{result}</p>"
            
            return output_html
        else:
            return "No input provided for analysis."
    
    except Exception as e:
        error = f"Error during analysis: {str(e)}"
        print(error)
        return error or "An error occurred during processing."

# Create Gradio interface
with gr.Blocks(title="Multimodal Sentiment Analysis") as demo:
    gr.Markdown("# Multimodal Sentiment Analysis")
    gr.Markdown("Upload or input text, audio, or video for sentiment analysis")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text Input", placeholder="Enter text for sentiment analysis...")
            audio_input = gr.Audio(label="Audio Input", type="filepath")
            video_input = gr.Video(label="Video Input")
            analyze_button = gr.Button("Analyze Sentiment")
        
        with gr.Column():
            output = gr.HTML(label="Analysis Results")
    
    analyze_button.click(
        analyze_sentiment,
        inputs=[text_input, audio_input, video_input],
        outputs=output
    )
    
    gr.Markdown("## About")
    gr.Markdown("""
    This is a Multimodal Sentiment Analysis tool that can analyze sentiment from:
    - Text
    - Audio
    - Video
    
    The model analyzes emotional content and provides sentiment scores across different modalities.
    """)

# This is needed for Hugging Face Spaces to properly launch the app
if __name__ == "__main__":
    demo.launch()
else:
    # For Hugging Face Spaces compatibility
    app = demo 