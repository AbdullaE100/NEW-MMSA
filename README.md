# Multimodal Sentiment Analysis (MMSA)

A comprehensive tool for analyzing sentiment across multiple modalities (visual, audio, and text) from video input.

This repository contains a Multimodal Sentiment Analysis system that combines three modalities to analyze sentiment:

1. **Visual**: Analyzes facial expressions using DeepFace
2. **Audio**: Analyzes vocal emotions using a CNN model trained on RAVDESS
3. **Text**: Analyzes sentiment from speech transcripts using RoBERTa

## Overview

This Multimodal Sentiment Analysis (MMSA) tool analyzes videos through three modalities:
- **Visual**: Facial expressions using DeepFace
- **Audio**: Speech emotions using a CNN model trained on the RAVDESS dataset
- **Text**: Sentiment analysis on automatically transcribed speech using RoBERTa

The system combines the analysis from these three modalities with the following weights:
- Visual (facial expressions): 45%
- Audio (speech emotions): 45%
- Text (transcribed speech): 10%

## Enhanced Text Sentiment Analysis

We've implemented an enhanced text sentiment analyzer using the RoBERTa model fine-tuned on Twitter data (`cardiffnlp/twitter-roberta-base-sentiment-latest`). This component includes:

- **Context enrichment** for short transcriptions, which are common in speech analysis
- **Emotion hint extraction** to detect emotional cues in text
- **Filename-based hints** to leverage video filenames for additional context
- **Advanced processing** to handle neutral-sounding transcriptions

The enhanced text analyzer showed a 71.43% category accuracy on test videos, compared to only 14.29% for the baseline model, representing a significant improvement in accuracy.

## Enhanced UI with SHAP Visualizations

The system now features an enhanced user interface with premium design elements:

- **Improved Contrast and Readability**: Carefully chosen color schemes for optimal readability
- **Modern Tab Navigation**: Intuitive tab-based interface for easy access to different analyses
- **SHAP Visualizations**: Explainable AI features showing how each modality contributes to the final sentiment score

## Local-Only Mode

For privacy-conscious users, we've added a local-only mode that never creates a public URL:

```bash
# Run the system in local-only mode (no public URL)
python run_local_mmsa.py
```

This mode ensures all analysis happens only on your local machine without exposing the interface to the internet.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AbdullaE100/NEW-MMSA.git
cd NEW-MMSA
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained models:
   - Audio sentiment model: Place in `models/audio_sentiment_model_notebook.h5`
   - Audio normalization parameters: Place in `models/audio_norm_params.json`

## Running the System

You can run the system using the provided scripts:

```bash
# Run the enhanced MMSA system
./run_enhanced_mmsa.sh

# Test the enhanced text sentiment analyzer
./run_enhanced_test.sh
```

### Web Interface

Run the Gradio web interface for interactive analysis:

```bash
# Start the web interface
python run_mmsa.py web
```

To create a public shareable URL (active for 72 hours):

```bash
python run_mmsa.py web --share
```

### Batch Processing

Process multiple videos at once:

```bash
# Place videos in the examples folder, then run:
python run_mmsa.py batch

# Or specify input and output directories:
python run_mmsa.py batch --input /path/to/videos --output /path/to/results
```

## System Components

- `src/mmsa_text_sentiment.py`: Enhanced text sentiment analysis using RoBERTa
- `src/mmsa_gradio_interface.py`: Gradio web interface for the MMSA system
- `test_enhanced_roberta.py`: Test script for the enhanced text analyzer
- `test_roberta_sentiment.py`: Test script for the baseline RoBERTa model

## Test Results

Performance comparison between baseline and enhanced text sentiment analyzers:

| Metric | Baseline RoBERTa | Enhanced RoBERTa | Improvement |
|--------|-----------------|------------------|-------------|
| Mean Absolute Error | 0.5143 | 0.1714 | 66.67% |
| Mean Squared Error | 0.3400 | 0.0793 | 76.68% |
| Category Accuracy | 14.29% | 71.43% | +57.14 pts |

## Requirements

The system requires the following dependencies:

- transformers
- torch
- numpy
- scipy
- matplotlib
- pandas
- whisper
- gradio
- deepface

You can install these dependencies using the provided requirements files:

```bash
pip install -r requirements.txt
```

## Key Features

- **Visual Analysis**: Uses DeepFace for facial expression recognition with multiple detector backends
- **Audio Analysis**: Custom CNN model trained on the RAVDESS emotional speech dataset
- **Text Analysis**: RoBERTa-based model fine-tuned on tweets for accurate sentiment classification
- **Speech Transcription**: Uses OpenAI Whisper with Google Speech Recognition as fallback
- **Multimodal Fusion**: Weighted integration of all modalities for comprehensive sentiment analysis
- **Batch Processing**: Process multiple videos for research and analysis
- **Web Interface**: User-friendly Gradio interface for interactive analysis

## Troubleshooting

### Common Issues:

1. **"ModuleNotFoundError"**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

2. **"Model file not found"**: Check that Git LFS downloaded the model files properly. You should see files in the `models/` directory. If not:
   ```bash
   git lfs pull
   ```

3. **Facial detection errors**: Try using a different detector:
   ```bash
   python run_mmsa.py web --detector opencv
   ```

## Examples

Upload a video to the web interface to get analysis results including:
- Overall sentiment score (-1 to +1)
- Detailed visual, audio, and text sentiment analysis
- Charts showing contribution of each modality to the final sentiment score
- Automatically transcribed speech text

## References

- **RoBERTa Text Sentiment Model**: [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **DeepFace**: [serengil/deepface](https://github.com/serengil/deepface)
- **Whisper**: [openai/whisper](https://github.com/openai/whisper)

## License

MIT 