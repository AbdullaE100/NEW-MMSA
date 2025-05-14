#!/usr/bin/env python3

"""
MMSA - Multimodal Sentiment Analysis Tool
Entry point script to run the MMSA application.
"""

import os
import sys
import argparse

# Add src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

def run_gradio_interface(share=False, detector="retinaface"):
    """Run the Gradio web interface"""
    from mmsa_gradio_interface import MultimodalSentimentGradio, main
    
    # Get model paths
    audio_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "audio_sentiment_model_notebook.h5")
    audio_norm_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "audio_norm_params.json")
    
    # Set output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "gradio_results")
    
    # Initialize the analyzer
    analyzer = MultimodalSentimentGradio(
        audio_model_path=audio_model_path,
        audio_norm_params_path=audio_norm_params_path,
        output_dir=output_dir,
        detector_backend=detector
    )
    
    # Call the original main function with share parameter
    import sys
    sys.argv = ['mmsa_gradio_interface.py']
    if share:
        sys.argv.append('--share')
    if detector:
        sys.argv.extend(['--detector', detector])
    main()
    
def run_batch_processor(input_dir=None, output_dir=None):
    """Run batch processing on a directory of videos"""
    from batch_mmsa_processor import BatchMMSAProcessor
    
    # Get model paths
    audio_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "audio_sentiment_model_notebook.h5")
    audio_norm_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "audio_norm_params.json")

    # Set default directories if not provided
    if input_dir is None:
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "batch_results")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the batch processor
    processor = BatchMMSAProcessor(
        audio_model_path=audio_model_path,
        audio_norm_params_path=audio_norm_params_path,
        output_dir=output_dir
    )
    
    # Process all videos in the input directory
    processor.process_directory(input_dir)
    
    print(f"Processed videos from {input_dir}. Results saved to {output_dir}")

def main():
    """Main entry point for the MMSA tool"""
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Gradio web interface command
    gradio_parser = subparsers.add_parser("web", help="Run the web interface")
    gradio_parser.add_argument("--share", action="store_true", help="Create a public URL")
    gradio_parser.add_argument("--detector", default="retinaface", 
                            help="Face detector backend (opencv, retinaface, mtcnn, ssd)")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Process multiple videos")
    batch_parser.add_argument("--input", dest="input_dir", help="Directory containing videos")
    batch_parser.add_argument("--output", dest="output_dir", help="Directory for output results")
    
    args = parser.parse_args()
    
    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        return
    
    # Run the appropriate function based on the command
    if args.command == "web":
        run_gradio_interface(share=args.share, detector=args.detector)
    elif args.command == "batch":
        run_batch_processor(input_dir=args.input_dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main() 