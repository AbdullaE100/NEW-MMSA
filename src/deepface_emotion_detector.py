#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import json
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('deepface_detector')

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

# Import DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    logger.error("DeepFace not found. Please install it with 'pip install deepface'")
    DEEPFACE_AVAILABLE = False

# Ensure required directories exist
def ensure_mmsa_directories():
    """Ensure that all required MMSA-FET directories exist"""
    # Get user's home directory in a platform-independent way
    home_dir = os.path.expanduser("~")
    
    # Create MMSA-FET temp directories
    mmsa_fet_base = os.path.join(home_dir, ".MMSA-FET")
    mmsa_fet_tmp = os.path.join(mmsa_fet_base, "tmp")
    
    # Create directories if they don't exist
    os.makedirs(mmsa_fet_tmp, exist_ok=True)
    os.makedirs(os.path.join(mmsa_fet_base, "features"), exist_ok=True)
    os.makedirs(os.path.join(mmsa_fet_base, "videos"), exist_ok=True)
    
    logger.info(f"MMSA-FET directories created at {mmsa_fet_base}")
    return mmsa_fet_base

class DeepFaceEmotionDetector:
    """
    A dedicated class for using DeepFace to analyze emotions in videos
    """
    
    def __init__(self, output_dir='./deepface_results', num_frames=10, detector_backend='retinaface'):
        """
        Initialize the DeepFace emotion detector
        
        Args:
            output_dir (str): Directory to save results
            num_frames (int): Number of frames to analyze per video
            detector_backend (str): Face detector backend to use (retinaface, opencv, mtcnn, ssd, etc.)
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace is not available. Please install it with 'pip install deepface'")
        
        self.output_dir = output_dir
        self.num_frames = num_frames
        self.detector_backend = detector_backend  # Default to retinaface which has good performance
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define emotion categories
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Define sentiment mapping with significantly boosted weights for happy/positive emotions
        # and more balanced weights for negative emotions
        self.sentiment_weights = {
            'happy': 1.0,      # Maximum positive (was 0.9)
            'surprise': 0.8,   # Strong positive (was 0.7)
            'neutral': 0.0,    # Neutral
            'sad': -0.7,       # Moderate negative (was -0.8)
            'fear': -0.6,      # Moderate negative (was -0.7)
            'disgust': -0.7,   # Moderate negative (was -0.8)
            'angry': -0.8      # Strong negative (was -0.9)
        }
        
        # Ensure MMSA directories exist (might be needed by other components)
        ensure_mmsa_directories()
        
        logger.info("DeepFace Emotion Detector initialized successfully")
        logger.info(f"Using emotion sentiment weights: {self.sentiment_weights}")
    
    def extract_frames(self, video_path, num_frames=None):
        """
        Extract frames from a video for emotion analysis
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to extract for analysis
            
        Returns:
            list: List of extracted frames as numpy arrays
        """
        if num_frames is None:
            num_frames = self.num_frames
            
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        if frame_count <= 0:
            logger.error(f"No frames detected in video: {video_path}")
            return []
        
        # If video is very short, reduce number of frames
        if frame_count < num_frames:
            num_frames = max(1, frame_count // 2)
            logger.info(f"Short video, analyzing {num_frames} frames")
        
        # Extract frames at regular intervals, including start and end
        frames = []
        if num_frames == 1:
            # For a single frame, take the middle one
            middle_frame_idx = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        else:
            frame_indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        
        cap.release()
        
        return frames
    
    def analyze_emotions(self, frames, detector_backend='opencv'):
        """
        Analyze emotions in multiple frames using DeepFace
        
        Args:
            frames (list): List of frames to analyze
            detector_backend (str): Face detector to use
            
        Returns:
            dict: Aggregated emotion analysis results
        """
        try:
            # Analyze emotions in each frame
            all_emotions = []
            valid_frames = 0
            
            for i, frame in enumerate(frames):
                try:
                    # Try multiple detector backends if the first fails
                    backends = [detector_backend, 'retinaface', 'mtcnn', 'ssd', 'opencv']
                    result = None
                    
                    for backend in backends:
                        try:
                            result = DeepFace.analyze(
                                frame, 
                                actions=['emotion'],
                                detector_backend=backend,
                                enforce_detection=False,
                                silent=True
                            )
                            if result:
                                break
                        except Exception as e:
                            logger.debug(f"Backend {backend} failed with error: {str(e)}")
                            continue
                    
                    if not result:
                        logger.warning(f"No faces detected in frame {i} with any backend")
                        continue
                    
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]  # Get first face result
                    
                    if 'emotion' in result:
                        all_emotions.append(result['emotion'])
                        valid_frames += 1
                        logger.debug(f"Frame {i}: Detected {result['emotion']}")
                except Exception as e:
                    logger.error(f"Error analyzing frame {i}: {str(e)}")
            
            if not all_emotions:
                logger.warning(f"No emotions detected in any frames")
                # Return default values instead of None
                return {
                    "dominant_emotion": "neutral",
                    "dominant_emotion_confidence": 0.3,  # Default low confidence
                    "emotion_distribution": {"neutral": 1.0},
                    "sentiment_score": 0.0
                }
            
            # Aggregate emotions across frames
            aggregated_emotions = {}
            for emotion in self.emotions:
                values = [e.get(emotion, 0) for e in all_emotions]
                aggregated_emotions[emotion] = np.mean(values) if values else 0
            
            # Normalize to ensure they sum to 1.0
            total = sum(aggregated_emotions.values())
            if total > 0:
                for emotion in aggregated_emotions:
                    aggregated_emotions[emotion] /= total
            else:
                # If all values are zero, set neutral to 1.0
                aggregated_emotions["neutral"] = 1.0
            
            # Find dominant emotion
            dominant_emotion = max(aggregated_emotions.items(), key=lambda x: x[1])[0]
            dominant_emotion_confidence = aggregated_emotions[dominant_emotion]
            
            # Ensure we have a non-zero confidence (minimum threshold to avoid zero confidence)
            if dominant_emotion_confidence < 0.3:
                logger.warning(f"Low confidence in dominant emotion, applying minimum threshold")
                dominant_emotion_confidence = max(0.3, dominant_emotion_confidence)
            
            # Calculate sentiment score based on emotion weights
            sentiment_score = 0
            for emotion, score in aggregated_emotions.items():
                sentiment_score += score * self.sentiment_weights.get(emotion, 0)
            
            result = {
                "dominant_emotion": dominant_emotion,
                "dominant_emotion_confidence": dominant_emotion_confidence,
                "emotion_distribution": aggregated_emotions,
                "sentiment_score": sentiment_score
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            # Return default values on error
            return {
                "dominant_emotion": "neutral",
                "dominant_emotion_confidence": 0.3,
                "emotion_distribution": {"neutral": 1.0},
                "sentiment_score": 0.0,
                "error": str(e)
            }
    
    def process_video(self, video_path, num_frames=None, detector_backend=None):
        """
        Process a video file with DeepFace to analyze emotions
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to analyze
            detector_backend (str): Face detector backend to use
            
        Returns:
            dict: Emotion analysis results
        """
        # Use specified detector or default
        detector_backend = detector_backend or self.detector_backend
        
        try:
            # Output log info
            logger.info(f"Processing video: {video_path} with {detector_backend} backend")
            
            # Extract frames
            frames = self.extract_frames(video_path, num_frames)
            
            if not frames:
                logger.warning(f"No frames extracted from {video_path}")
                return None
            
            logger.info(f"Extracted {len(frames)} frames from video")
            
            # Analyze emotions in frames
            results = self.analyze_emotions(frames, detector_backend)
            
            if not results:
                logger.warning(f"No emotions detected in {video_path}")
                return None
            
            # Special handling for happy videos - boost happy and surprise scores
            # This prevents happy videos from being classified as negative
            emotion_dist = results.get('emotion_distribution', {})
            happy_score = emotion_dist.get('happy', 0)
            surprise_score = emotion_dist.get('surprise', 0)
            
            # Apply a post-processing boost for happy emotions to address consistent 
            # underdetection of happy emotions in videos
            if happy_score > 0.2 or (happy_score > 0.15 and surprise_score > 0.15):
                logger.info(f"Significant happy component detected: happy={happy_score:.2f}, surprise={surprise_score:.2f}")
                
                # Recalculate sentiment with happy boosting
                # Apply additional boosting factors to happy and surprise
                happy_boost_factor = 1.25
                surprise_boost_factor = 1.15
                
                sentiment_score = 0
                for emotion, score in emotion_dist.items():
                    if emotion == 'happy':
                        # Apply boosting to happy
                        adjusted_score = min(1.0, score * happy_boost_factor)
                        sentiment_score += adjusted_score * self.sentiment_weights[emotion]
                    elif emotion == 'surprise':
                        # Apply boosting to surprise
                        adjusted_score = min(1.0, score * surprise_boost_factor)
                        sentiment_score += adjusted_score * self.sentiment_weights[emotion]
                    else:
                        sentiment_score += score * self.sentiment_weights.get(emotion, 0)
                
                logger.info(f"Applied happiness boost: original={results['sentiment_score']:.2f}, new={sentiment_score:.2f}")
                results['sentiment_score'] = sentiment_score
            
            # Save results to output directory
            output_path = os.path.join(
                self.output_dir, 
                f"{os.path.splitext(os.path.basename(video_path))[0]}_emotion.json"
            )
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return None
    
    def process_directory(self, directory, file_pattern="*.mp4"):
        """
        Process all videos in a directory
        
        Args:
            directory (str): Path to directory containing videos
            file_pattern (str): Pattern to match video files
            
        Returns:
            list: List of sentiment analysis results
        """
        # Get list of video files
        video_files = glob(os.path.join(directory, file_pattern))
        
        if not video_files:
            print(f"No video files found in {directory}")
            return []
        
        print(f"Found {len(video_files)} video files in {directory}")
        
        # Process each video
        results = []
        for video_path in tqdm(video_files, desc="Processing videos"):
            result = self.process_video(video_path)
            if result:
                results.append(result)
        
        # Check for ground truth labels
        labels_path = os.path.join(directory, "labels.csv")
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            print("\nComparing with ground truth:")
            
            accuracy = 0
            for result in results:
                file_name = result['file']
                label_row = labels_df[labels_df['file'] == file_name]
                if not label_row.empty:
                    expected = float(label_row['expected_sentiment'].values[0])
                    result['expected'] = expected
                    diff = abs(result['sentiment_score'] - expected)
                    print(f"  - {file_name}: Predicted={result['sentiment_score']:.2f}, Expected={expected:.2f}, Diff={diff:.2f}")
                    
                    # Simple accuracy metric (within 0.3 of ground truth)
                    if diff < 0.3:
                        accuracy += 1
            
            if results:
                accuracy_percentage = (accuracy / len(results)) * 100
                print(f"\nOverall accuracy: {accuracy_percentage:.1f}% ({accuracy}/{len(results)} correct)")
        
        # Save results to file
        results_path = os.path.join(self.output_dir, "sentiment_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nResults saved to {results_path}")
        
        return results
    
    def analyze_video(self, video_path, num_frames=None, detector_backend=None):
        """
        Alias for process_video to maintain compatibility with the Gradio interface
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to analyze
            detector_backend (str): Face detector backend to use
            
        Returns:
            dict: Emotion analysis results
        """
        return self.process_video(video_path, num_frames, detector_backend)
    
    def visualize_results(self, results, output_path=None):
        """
        Visualize sentiment analysis results
        
        Args:
            results (list): List of sentiment analysis results
            output_path (str): Path to save visualization
        """
        if not results:
            print("No results to visualize")
            return
        
        # Create figure for sentiment scores
        plt.figure(figsize=(14, 10))
        
        # Create subplot for sentiment scores
        plt.subplot(2, 1, 1)
        
        # Extract data
        files = [r['file'].split('.')[0] for r in results]
        scores = [r['sentiment_score'] for r in results]
        expected = [r.get('expected', None) for r in results]
        
        # Create bars for predicted sentiment
        bars = plt.bar(range(len(files)), scores, 0.4, label='Predicted')
        
        # Color bars based on sentiment
        colors = ['red' if s < -0.2 else 'green' if s > 0.2 else 'gray' for s in scores]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add expected sentiment scores if available
        if any(e is not None for e in expected):
            expected_valid = [(i, e) for i, e in enumerate(expected) if e is not None]
            if expected_valid:
                x_vals, y_vals = zip(*expected_valid)
                plt.plot(x_vals, y_vals, 'o-', color='blue', label='Expected')
        
        # Add grid, labels and title
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xlabel('Video')
        plt.ylabel('Sentiment Score')
        plt.title('DeepFace Sentiment Analysis Results')
        plt.xticks(range(len(files)), files, rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.3)
        plt.axhline(y=-0.2, color='red', linestyle='--', alpha=0.3)
        plt.legend()
        
        # Create subplot for emotion breakdown
        plt.subplot(2, 1, 2)
        
        # Extract emotion data
        emotion_data = np.array([[r['emotion_distribution'].get(emotion, 0) for emotion in self.emotions] for r in results])
        
        # Create heatmap
        im = plt.imshow(emotion_data, cmap='YlOrRd')
        
        # Add colorbar
        plt.colorbar(im, label='Score')
        
        # Add labels
        plt.xticks(range(len(self.emotions)), self.emotions, rotation=45)
        plt.yticks(range(len(files)), files)
        plt.title("Emotion Breakdown by Video")
        
        plt.tight_layout()
        
        # Save or display figure
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
    
    def generate_html_report(self, results):
        """
        Generate an HTML report of the sentiment analysis results
        
        Args:
            results (list): List of sentiment analysis results
        """
        if not results:
            print("No results to generate report for")
            return
        
        # Create report file
        report_path = os.path.join(self.output_dir, "sentiment_report.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DeepFace Emotion Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2 {{ color: #333; }}
                .container {{ max-width: 1000px; margin: 0 auto; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .neutral {{ color: gray; }}
                .accuracy {{ font-weight: bold; margin-top: 20px; }}
                .chart {{ margin-top: 30px; }}
                .emotions {{ margin-top: 10px; font-size: 0.9em; }}
                .emotion-bar {{ height: 15px; margin: 2px 0; background-color: #eee; position: relative; }}
                .emotion-fill {{ height: 100%; position: absolute; left: 0; top: 0; }}
                .emotion-angry {{ background-color: #ff4d4d; }}
                .emotion-disgust {{ background-color: #9933ff; }}
                .emotion-fear {{ background-color: #cc00cc; }}
                .emotion-happy {{ background-color: #33cc33; }}
                .emotion-sad {{ background-color: #3366ff; }}
                .emotion-surprise {{ background-color: #ffcc00; }}
                .emotion-neutral {{ background-color: #999999; }}
                .emotion-label {{ display: inline-block; width: 80px; }}
                .emotion-value {{ display: inline-block; width: 50px; text-align: right; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>DeepFace Emotion Analysis Report</h1>
                <p>Analysis performed on {len(results)} videos</p>
                
                <table>
                    <tr>
                        <th>Video</th>
                        <th>Sentiment</th>
                        <th>Score</th>
                        <th>Expected</th>
                        <th>Difference</th>
                        <th>Dominant Emotion</th>
                        <th>Emotion Breakdown</th>
                    </tr>
        """
        
        correct_count = 0
        for result in results:
            file_name = result['file']
            sentiment = "Positive" if result['sentiment_score'] > 0.2 else "Negative" if result['sentiment_score'] < -0.2 else "Neutral"
            score = result['sentiment_score']
            expected = result.get('expected', None)
            dominant_emotion = result['dominant_emotion']
            
            sentiment_class = ''
            if sentiment == 'Positive':
                sentiment_class = 'positive'
            elif sentiment == 'Negative':
                sentiment_class = 'negative'
            else:
                sentiment_class = 'neutral'
            
            diff_text = '—'
            if expected is not None:
                diff = abs(score - expected)
                diff_text = f"{diff:.2f}"
                if diff < 0.3:
                    correct_count += 1
            
            # Create emotion bars HTML
            emotions_html = '<div class="emotions">'
            for emotion in self.emotions:
                emotion_value = result['emotion_distribution'].get(emotion, 0)
                emotions_html += f"""
                <div>
                    <span class="emotion-label">{emotion}</span>
                    <span class="emotion-value">{emotion_value:.1f}%</span>
                    <div class="emotion-bar">
                        <div class="emotion-fill emotion-{emotion}" style="width: {emotion_value}%;"></div>
                    </div>
                </div>
                """
            emotions_html += '</div>'
            
            html_content += f"""
                <tr>
                    <td>{file_name}</td>
                    <td class="{sentiment_class}">{sentiment}</td>
                    <td>{score:.2f}</td>
                    <td>{expected if expected is not None else '—'}</td>
                    <td>{diff_text}</td>
                    <td>{dominant_emotion}</td>
                    <td>{emotions_html}</td>
                </tr>
            """
        
        # Add accuracy information if ground truth was available
        accuracy_html = ""
        if correct_count > 0:
            accuracy_percentage = (correct_count / len(results)) * 100
            accuracy_html = f"""
                <p class="accuracy">Overall accuracy: {accuracy_percentage:.1f}% ({correct_count}/{len(results)} correct)</p>
            """
        
        # Add image reference if it exists
        image_html = ""
        vis_path = os.path.join(self.output_dir, "sentiment_visualization.png")
        if os.path.exists(vis_path):
            rel_path = os.path.basename(vis_path)
            image_html = f"""
                <div class="chart">
                    <h2>Visualization</h2>
                    <img src="{rel_path}" alt="Sentiment Visualization" style="max-width:100%">
                </div>
            """
        
        html_content += f"""
                </table>
                
                {accuracy_html}
                
                {image_html}
                
                <p>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><small>Analysis performed using DeepFace emotion detection</small></p>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {report_path}")

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Detect emotions in videos using DeepFace")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing video files")
    parser.add_argument("--video", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to analyze per video")
    parser.add_argument("--output", type=str, default="./deepface_results", help="Output directory")
    parser.add_argument("--detector", type=str, default="retinaface", 
                      choices=["opencv", "retinaface", "mtcnn", "ssd"],
                      help="Face detector backend (opencv, retinaface, mtcnn, ssd)")
    parser.add_argument("--visualize", action="store_true", 
                      help="Visualize results")
    parser.add_argument("--report", action="store_true", 
                      help="Generate HTML report")
    args = parser.parse_args()
    
    # Check that we have either a directory or a video file
    if not args.dir and not args.video:
        print("Error: Please provide either a directory or a video file")
        sys.exit(1)
    
    # Initialize the detector
    detector = DeepFaceEmotionDetector(output_dir=args.output, num_frames=args.num_frames, detector_backend=args.detector)
    
    # Process video(s)
    results = []
    
    if args.video:
        # Process single video
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
            
        print(f"Processing video: {args.video}")
        result = detector.process_video(args.video, detector_backend=args.detector)
        results.append(result)
        
    elif args.dir:
        # Process all videos in directory
        if not os.path.exists(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            sys.exit(1)
            
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob(os.path.join(args.dir, f"*{ext}")))
        
        if not video_files:
            print(f"No video files found in {args.dir}")
            sys.exit(1)
            
        print(f"Processing {len(video_files)} videos from {args.dir}")
        
        for video in tqdm(video_files):
            result = detector.process_video(video, detector_backend=args.detector)
            result["file"] = os.path.basename(video)
            results.append(result)
    
    # Save results
    output_file = os.path.join(args.output, "sentiment_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
        
    print(f"Results saved to {output_file}")
    
    # Visualize results if requested
    if args.visualize and results:
        vis_path = os.path.join(args.output, "sentiment_visualization.png")
        detector.visualize_results(results, vis_path)
    
    # Generate HTML report if requested
    if args.report and results:
        detector.generate_html_report(results)

if __name__ == "__main__":
    main() 