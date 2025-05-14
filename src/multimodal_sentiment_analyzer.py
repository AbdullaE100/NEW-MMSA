#!/usr/bin/env python3

import os
import sys
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our custom modules
from deepface_emotion_detector import DeepFaceEmotionDetector
from audio_sentiment import AudioSentimentAnalyzer

class MultimodalSentimentAnalyzer:
    """
    Multimodal sentiment analyzer that combines visual (DeepFace) and auditory (CNN)
    analysis for more accurate sentiment prediction
    """
    
    def __init__(self, 
                 output_dir='./multimodal_results', 
                 num_frames=10,
                 audio_model_path=None,
                 visual_weight=0.8,
                 audio_weight=0.2):
        """
        Initialize the multimodal sentiment analyzer
        
        Args:
            output_dir (str): Directory to save results
            num_frames (int): Number of frames to analyze per video
            audio_model_path (str): Path to pre-trained audio model
            visual_weight (float): Weight to give visual sentiment (0-1)
            audio_weight (float): Weight to give audio sentiment (0-1)
        """
        self.output_dir = output_dir
        self.num_frames = num_frames
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize component analyzers
        try:
            self.visual_analyzer = DeepFaceEmotionDetector(
                output_dir=os.path.join(output_dir, 'visual'),
                num_frames=num_frames
            )
            print("Visual analyzer (DeepFace) initialized")
        except ImportError as e:
            print(f"Warning: Could not initialize visual analyzer: {str(e)}")
            self.visual_analyzer = None
        
        try:
            self.audio_analyzer = AudioSentimentAnalyzer(
                model_path=audio_model_path
            )
            print("Audio analyzer initialized")
        except ImportError as e:
            print(f"Warning: Could not initialize audio analyzer: {str(e)}")
            self.audio_analyzer = None
        
        # Ensure at least one analyzer is available
        if not self.visual_analyzer and not self.audio_analyzer:
            raise ImportError("Neither visual nor audio analyzer could be initialized")
        
        # Set modality weights
        total_weight = visual_weight + audio_weight
        self.visual_weight = visual_weight / total_weight
        self.audio_weight = audio_weight / total_weight
        
        print(f"Multimodal Sentiment Analyzer initialized (Visual weight: {self.visual_weight:.2f}, Audio weight: {self.audio_weight:.2f})")
    
    def process_video(self, video_path, detector_backend='opencv'):
        """
        Process a video through both visual and audio analyzers
        
        Args:
            video_path (str): Path to video file
            detector_backend (str): Face detector backend for DeepFace
            
        Returns:
            dict: Combined analysis results
        """
        results = {
            'file': os.path.basename(video_path),
            'path': video_path,
            'visual_analysis': None,
            'audio_analysis': None,
            'combined_sentiment': None
        }
        
        # Run visual analysis if available
        if self.visual_analyzer:
            try:
                visual_results = self.visual_analyzer.process_video(
                    video_path, 
                    num_frames=self.num_frames,
                    detector_backend=detector_backend
                )
                
                if visual_results:
                    results['visual_analysis'] = visual_results
                    print(f"Visual analysis completed: {visual_results.get('dominant_emotion', 'unknown')} " + 
                          f"(sentiment: {visual_results.get('sentiment_score', 0):.2f})")
            except Exception as e:
                print(f"Error in visual analysis: {str(e)}")
        
        # Run audio analysis if available
        audio_results = None
        if self.audio_analyzer:
            try:
                audio_results = self.audio_analyzer.predict_from_video(video_path)
                
                if audio_results:
                    results['audio_analysis'] = audio_results
                    print(f"Audio analysis completed: {audio_results.get('emotion', 'unknown')} " + 
                          f"(sentiment: {audio_results.get('sentiment_score', 0):.2f})")
            except Exception as e:
                print(f"Error in audio analysis: {str(e)}")
        
        # Calculate combined sentiment score
        visual_score = results.get('visual_analysis', {}).get('sentiment_score', 0) if results.get('visual_analysis') else 0
        audio_score = results.get('audio_analysis', {}).get('sentiment_score', 0) if results.get('audio_analysis') else 0
        
        # If we have both modalities, do weighted combination
        if results.get('visual_analysis') and results.get('audio_analysis'):
            # Apply special weighting for happy/positive emotions to fix underdetection
            visual_emotion = results['visual_analysis'].get('dominant_emotion', 'neutral')
            audio_emotion = results['audio_analysis'].get('emotion', 'neutral') 
            
            # Check if either modality detected happiness or surprise
            is_visual_happy = visual_emotion in ['happy', 'surprise']
            is_audio_happy = audio_emotion in ['happy', 'surprised', 'calm']
            
            # If either modality detected happiness, give more weight to that modality
            if is_visual_happy and not is_audio_happy:
                # If visual shows happiness but audio doesn't, trust visual more
                dynamic_visual_weight = min(0.9, self.visual_weight + 0.1)  
                dynamic_audio_weight = 1.0 - dynamic_visual_weight
                print(f"Detecting visual happiness - increasing visual weight: {self.visual_weight:.2f} → {dynamic_visual_weight:.2f}")
            elif is_audio_happy and not is_visual_happy:
                # If audio shows happiness but visual doesn't, trust audio more
                dynamic_audio_weight = min(0.5, self.audio_weight + 0.2)
                dynamic_visual_weight = 1.0 - dynamic_audio_weight
                print(f"Detecting audio happiness - increasing audio weight: {self.audio_weight:.2f} → {dynamic_audio_weight:.2f}")
            else:
                # Default weights
                dynamic_visual_weight = self.visual_weight
                dynamic_audio_weight = self.audio_weight
            
            # Calculate weighted score
            combined_score = (
                visual_score * dynamic_visual_weight + 
                audio_score * dynamic_audio_weight
            )
            
            # Post-processing boost for happy videos
            # Look for mixed signals that might indicate happiness
            visual_emotions = results['visual_analysis'].get('emotion_distribution', {})
            visual_happy = visual_emotions.get('happy', 0)
            visual_surprise = visual_emotions.get('surprise', 0)
            
            audio_probs = results['audio_analysis'].get('probabilities', {})
            audio_happy = audio_probs.get('happy', 0) if audio_probs else 0
            audio_surprised = audio_probs.get('surprised', 0) if audio_probs else 0
            audio_calm = audio_probs.get('calm', 0) if audio_probs else 0
            
            # Calculate positive signals across modalities
            positive_visual = visual_happy + visual_surprise
            positive_audio = audio_happy + audio_surprised + audio_calm
            
            # Apply a boost if significant positive signals exist in both modalities
            # but the combined score doesn't reflect it
            if positive_visual > 0.2 and positive_audio > 0.2 and combined_score < 0.3:
                boost_amount = (positive_visual + positive_audio) * 0.15
                original_score = combined_score
                combined_score = min(0.7, combined_score + boost_amount)
                print(f"Applied cross-modal positive boost: {original_score:.2f} → {combined_score:.2f}")
                results['multimodal_boost_applied'] = True
            
            # Get most confident emotion from either modality
            visual_conf = results['visual_analysis'].get('dominant_emotion_confidence', 0)
            audio_conf = results['audio_analysis'].get('confidence', 0)
            
            if visual_conf >= audio_conf:
                dominant_emotion = results['visual_analysis'].get('dominant_emotion', 'neutral')
            else:
                dominant_emotion = results['audio_analysis'].get('emotion', 'neutral')
            
            results['combined_sentiment'] = {
                'score': combined_score,
                'dominant_emotion': dominant_emotion,
                'classification': 'positive' if combined_score > 0.1 else 
                                 'negative' if combined_score < -0.1 else 'neutral',
                'visual_weight': dynamic_visual_weight,
                'audio_weight': dynamic_audio_weight
            }
            
            print(f"Combined sentiment: {results['combined_sentiment']['classification']} ({combined_score:.2f})")
        
        # If we only have one modality, use that
        elif results.get('visual_analysis'):
            results['combined_sentiment'] = {
                'score': visual_score,
                'dominant_emotion': results['visual_analysis'].get('dominant_emotion', 'neutral'),
                'classification': 'positive' if visual_score > 0.1 else 
                                 'negative' if visual_score < -0.1 else 'neutral'
            }
            print(f"Using only visual sentiment: {results['combined_sentiment']['classification']} ({visual_score:.2f})")
        
        elif results.get('audio_analysis'):
            results['combined_sentiment'] = {
                'score': audio_score,
                'dominant_emotion': results['audio_analysis'].get('emotion', 'neutral'),
                'classification': 'positive' if audio_score > 0.1 else 
                                 'negative' if audio_score < -0.1 else 'neutral'
            }
            print(f"Using only audio sentiment: {results['combined_sentiment']['classification']} ({audio_score:.2f})")
        else:
            # If no analysis was successful, return default neutral values
            results['combined_sentiment'] = {
                'score': 0.0,
                'dominant_emotion': 'neutral',
                'classification': 'neutral'
            }
            print("Warning: Neither visual nor audio analysis was successful. Using default neutral values.")
        
        # Save results to JSON file
        results_file = os.path.join(
            self.output_dir, 
            f"{os.path.splitext(os.path.basename(video_path))[0]}_sentiment.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def process_directory(self, directory, file_pattern="*.mp4"):
        """
        Process all videos in a directory
        
        Args:
            directory (str): Directory containing videos
            file_pattern (str): Pattern to match video files
            
        Returns:
            list: Analysis results for all videos
        """
        from glob import glob
        
        # Find all video files in directory
        video_paths = glob(os.path.join(directory, file_pattern))
        
        if not video_paths:
            print(f"No videos found in {directory} matching pattern {file_pattern}")
            return []
        
        results = []
        
        print(f"Processing {len(video_paths)} videos in {directory}...")
        for video_path in tqdm(video_paths):
            result = self.process_video(video_path)
            if result:
                results.append(result)
        
        # Save combined results
        all_results_file = os.path.join(self.output_dir, "all_results.json")
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"All results saved to {all_results_file}")
        
        return results
    
    def visualize_results(self, results, output_path=None):
        """
        Visualize sentiment analysis results
        
        Args:
            results (list): List of analysis results
            output_path (str): Path to save visualization
            
        Returns:
            None
        """
        if not results:
            print("No results to visualize")
            return
        
        # Extract sentiment scores
        filenames = [result['file'] for result in results]
        visual_scores = []
        audio_scores = []
        combined_scores = []
        
        for result in results:
            if result.get('visual_analysis'):
                visual_scores.append(result['visual_analysis'].get('sentiment_score', 0))
            else:
                visual_scores.append(0)
            
            if result.get('audio_analysis'):
                audio_scores.append(result['audio_analysis'].get('sentiment_score', 0))
            else:
                audio_scores.append(0)
            
            if result.get('combined_sentiment'):
                combined_scores.append(result['combined_sentiment'].get('score', 0))
            else:
                combined_scores.append(0)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot sentiment scores
        x = range(len(filenames))
        plt.plot(x, visual_scores, 'o-', label='Visual Sentiment')
        plt.plot(x, audio_scores, 's-', label='Audio Sentiment')
        plt.plot(x, combined_scores, '^-', label='Combined Sentiment')
        
        # Add horizontal lines for positive/negative threshold
        plt.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Positive Threshold')
        plt.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5, label='Negative Threshold')
        
        # Add labels and legend
        plt.xticks(x, filenames, rotation=45, ha='right')
        plt.ylabel('Sentiment Score')
        plt.title('Multimodal Sentiment Analysis Results')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()

def main():
    """Main function to run the multimodal sentiment analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing videos")
    parser.add_argument("--output-dir", type=str, default="./multimodal_results", 
                        help="Directory to save results")
    parser.add_argument("--num-frames", type=int, default=10, 
                        help="Number of frames to analyze per video")
    parser.add_argument("--audio-model", type=str, default=None, 
                        help="Path to pre-trained audio model")
    parser.add_argument("--visual-weight", type=float, default=0.8, 
                        help="Weight to give visual sentiment (0-1)")
    parser.add_argument("--audio-weight", type=float, default=0.2, 
                        help="Weight to give audio sentiment (0-1)")
    parser.add_argument("--detector", type=str, default="opencv", 
                        help="Face detector backend for DeepFace")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize results")
    
    args = parser.parse_args()
    
    if not args.video and not args.dir:
        parser.error("Either --video or --dir must be specified")
    
    # Initialize analyzer
    analyzer = MultimodalSentimentAnalyzer(
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        audio_model_path=args.audio_model,
        visual_weight=args.visual_weight,
        audio_weight=args.audio_weight
    )
    
    # Process video or directory
    if args.video:
        results = [analyzer.process_video(args.video, detector_backend=args.detector)]
    else:
        results = analyzer.process_directory(args.dir)
    
    # Visualize results if requested
    if args.visualize and results:
        vis_path = os.path.join(args.output_dir, "sentiment_visualization.png")
        analyzer.visualize_results(results, vis_path)

if __name__ == "__main__":
    main() 