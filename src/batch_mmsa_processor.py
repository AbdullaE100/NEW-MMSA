#!/usr/bin/env python3

import os
import sys
import json
import argparse
import glob
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our custom MMSA modules
from mmsa_gradio_interface import MultimodalSentimentGradio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_mmsa.log')
    ]
)
logger = logging.getLogger('batch_mmsa')

class BatchMMSAProcessor:
    """
    Batch processor for Multimodal Sentiment Analysis on multiple videos
    """
    
    def __init__(self, 
                output_dir="./batch_results",
                audio_model_path=None,
                audio_norm_params_path=None,
                num_frames=15,
                detector_backend='retinaface',
                enable_transcription=True):
        """
        Initialize the batch processor
        
        Args:
            output_dir (str): Directory to save results
            audio_model_path (str): Path to audio sentiment model
            audio_norm_params_path (str): Path to audio normalization parameters
            num_frames (int): Number of frames to analyze per video
            detector_backend (str): Face detector backend for DeepFace
            enable_transcription (bool): Whether to enable speech transcription
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initializing MMSA processor with output directory: {output_dir}")
        logger.info(f"Using detector backend: {detector_backend}")
        
        # Initialize the MMSA analyzer
        self.analyzer = MultimodalSentimentGradio(
            audio_model_path=audio_model_path,
            audio_norm_params_path=audio_norm_params_path,
            output_dir=output_dir,
            num_frames=num_frames,
            detector_backend=detector_backend,
            cleanup_temp=True
        )
        
        self.enable_transcription = enable_transcription
        logger.info(f"Speech transcription enabled: {enable_transcription}")
    
    def process_video(self, video_path):
        """
        Process a single video file
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Processing video: {video_path}")
        
        try:
            # Analyze the video
            results = self.analyzer.analyze_video(video_path)
            
            # Save individual result
            output_file = os.path.join(
                self.output_dir, 
                f"{os.path.splitext(os.path.basename(video_path))[0]}_sentiment.json"
            )
            
            # Use custom encoder for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (
                        float, 
                        int, 
                        str, 
                        bool, 
                        list, 
                        dict, 
                        tuple, 
                        type(None)
                    )):
                        return super(NumpyEncoder, self).default(obj)
                    import numpy as np
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return str(obj)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Results saved to: {output_file}")
            
            # Return results summary
            summary = {
                "file": os.path.basename(video_path),
                "combined_score": results.get("combined_score", 0),
                "combined_sentiment": results.get("combined_sentiment", "neutral"),
                "modalities": []
            }
            
            if "visual" in results and results["visual"]:
                summary["modalities"].append("visual")
                summary["visual_score"] = results["visual"].get("sentiment_score", 0)
                summary["visual_emotion"] = results["visual"].get("dominant_emotion", "unknown")
            
            if "audio" in results and results["audio"]:
                summary["modalities"].append("audio")
                summary["audio_score"] = results["audio"].get("sentiment_score", 0)
                summary["audio_emotion"] = results["audio"].get("emotion", "unknown")
            
            if "text" in results and results["text"]:
                summary["modalities"].append("text")
                summary["text_score"] = results["text"].get("sentiment_score", 0)
                
            if "transcribed_text" in results:
                summary["transcribed_text"] = results["transcribed_text"]
            
            return summary
        
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return {
                "file": os.path.basename(video_path),
                "error": str(e),
                "combined_score": 0,
                "combined_sentiment": "error"
            }
    
    def process_directory(self, directory, pattern="*.mp4"):
        """
        Process all videos in a directory
        
        Args:
            directory (str): Directory containing videos
            pattern (str): File pattern to match
            
        Returns:
            list: Results for all processed videos
        """
        logger.info(f"Processing directory: {directory} with pattern: {pattern}")
        
        # Find all matching video files
        if os.path.isdir(directory):
            video_files = glob.glob(os.path.join(directory, pattern))
        else:
            video_files = glob.glob(pattern)
        
        if not video_files:
            logger.warning(f"No videos found matching pattern: {pattern}")
            return []
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        # Process each video
        results = []
        for video_file in tqdm(video_files, desc="Processing videos"):
            result = self.process_video(video_file)
            results.append(result)
        
        # Save overall results
        self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """
        Save batch results to CSV and generate summary visualizations
        
        Args:
            results (list): List of result dictionaries
        """
        if not results:
            logger.warning("No results to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        csv_file = os.path.join(self.output_dir, "batch_results.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved batch results to: {csv_file}")
        
        # Generate visualizations
        self.generate_visualizations(df)
    
    def generate_visualizations(self, df):
        """
        Generate summary visualizations for batch results
        
        Args:
            df (DataFrame): Results dataframe
        """
        if len(df) == 0:
            return
        
        # Create output directory for plots
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Distribution of sentiment scores
        plt.figure(figsize=(10, 6))
        plt.hist(df['combined_score'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score (-1 to 1)')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Sentiment category counts
        sentiment_counts = df['combined_sentiment'].value_counts()
        plt.figure(figsize=(8, 6))
        colors = {'Positive': '#4CAF50', 'Neutral': '#9E9E9E', 'Negative': '#F44336'}
        bar_colors = [colors.get(s, '#1976D2') for s in sentiment_counts.index]
        sentiment_counts.plot(kind='bar', color=bar_colors)
        plt.title('Sentiment Category Distribution')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'sentiment_categories.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Compare modalities (if available)
        if 'visual_score' in df.columns and 'audio_score' in df.columns:
            # Create scatter plot of visual vs audio scores
            plt.figure(figsize=(8, 8))
            plt.scatter(df['visual_score'], df['audio_score'], alpha=0.7, 
                       c=df['combined_score'], cmap='RdYlGn', s=100)
            plt.colorbar(label='Combined Score')
            plt.title('Visual vs Audio Sentiment Scores')
            plt.xlabel('Visual Score')
            plt.ylabel('Audio Score')
            plt.grid(alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'modality_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        if 'text_score' in df.columns:
            # Comparison with text modality
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Visual vs Text
            if 'visual_score' in df.columns:
                axes[0].scatter(df['visual_score'], df['text_score'], alpha=0.7,
                              c=df['combined_score'], cmap='RdYlGn', s=100)
                axes[0].set_title('Visual vs Text Sentiment Scores')
                axes[0].set_xlabel('Visual Score')
                axes[0].set_ylabel('Text Score')
                axes[0].grid(alpha=0.3)
                axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Audio vs Text
            if 'audio_score' in df.columns:
                axes[1].scatter(df['audio_score'], df['text_score'], alpha=0.7,
                              c=df['combined_score'], cmap='RdYlGn', s=100)
                axes[1].set_title('Audio vs Text Sentiment Scores')
                axes[1].set_xlabel('Audio Score')
                axes[1].set_ylabel('Text Score')
                axes[1].grid(alpha=0.3)
                axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'text_modality_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Generated visualization plots in: {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description="Batch process videos with Multimodal Sentiment Analysis")
    parser.add_argument("input_dir", help="Input directory containing videos or path to a single video file")
    parser.add_argument("--output-dir", "-o", default="./batch_results", help="Output directory for results")
    parser.add_argument("--pattern", "-p", default="*.mp4", help="File pattern to match (e.g., *.mp4, *.avi)")
    parser.add_argument("--detector", "-d", default="retinaface", 
                       choices=["opencv", "retinaface", "mtcnn", "ssd"], 
                       help="Face detector backend")
    parser.add_argument("--frames", "-f", type=int, default=15, 
                       help="Number of frames to analyze per video")
    parser.add_argument("--audio-model", type=str, default=None, 
                       help="Path to audio sentiment model")
    parser.add_argument("--no-transcription", action="store_true",
                       help="Disable speech transcription")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Initialize processor
    processor = BatchMMSAProcessor(
        output_dir=args.output_dir,
        audio_model_path=args.audio_model,
        num_frames=args.frames,
        detector_backend=args.detector,
        enable_transcription=not args.no_transcription
    )
    
    # Process input
    if os.path.isdir(args.input_dir):
        processor.process_directory(args.input_dir, args.pattern)
    elif os.path.isfile(args.input_dir):
        # Single file mode
        result = processor.process_video(args.input_dir)
        processor.save_results([result])
    else:
        logger.error(f"Input path not found: {args.input_dir}")
        sys.exit(1)

if __name__ == "__main__":
    main() 