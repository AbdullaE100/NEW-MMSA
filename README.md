# Multimodal Sentiment Analysis (MMSA) System

This repository contains a Multimodal Sentiment Analysis system that combines three modalities to analyze sentiment:

1. **Visual**: Analyzes facial expressions using DeepFace
2. **Audio**: Analyzes vocal emotions using a CNN model trained on RAVDESS
3. **Text**: Analyzes sentiment from speech transcripts using RoBERTa

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

## Running the System

You can run the system using the provided scripts:

```bash
# Run the enhanced MMSA system
./run_enhanced_mmsa.sh

# Test the enhanced text sentiment analyzer
./run_enhanced_test.sh
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
pip install -r test_roberta_requirements.txt
```

## References

- **RoBERTa Text Sentiment Model**: [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **DeepFace**: [serengil/deepface](https://github.com/serengil/deepface)
- **Whisper**: [openai/whisper](https://github.com/openai/whisper)

## Overview

This Multimodal Sentiment Analysis (MMSA) tool analyzes videos through three modalities:
- **Visual**: Facial expressions using DeepFace
- **Audio**: Speech emotions using a CNN model trained on the RAVDESS dataset
- **Text**: Sentiment analysis on automatically transcribed speech using RoBERTa

The system combines the analysis from these three modalities with the following weights:
- Visual (facial expressions): 45%
- Audio (speech emotions): 45%
- Text (transcribed speech): 10%

### Key Features

- **Visual Analysis**: Uses DeepFace for facial expression recognition with multiple detector backends
- **Audio Analysis**: Custom CNN model trained on the RAVDESS emotional speech dataset
- **Text Analysis**: RoBERTa-based model fine-tuned on tweets for accurate sentiment classification
- **Speech Transcription**: Uses OpenAI Whisper with Google Speech Recognition as fallback
- **Multimodal Fusion**: Weighted integration of all modalities for comprehensive sentiment analysis
- **Batch Processing**: Process multiple videos for research and analysis
- **Web Interface**: User-friendly Gradio interface for interactive analysis

## 🔰 Beginner-Friendly Setup Guide

### Prerequisites

1. **Install Git** - Download and install from [git-scm.com](https://git-scm.com/downloads)
2. **Install Git LFS** - Follow instructions at [git-lfs.github.com](https://git-lfs.github.com/)
3. **Install Python** - Download Python 3.8 or later from [python.org](https://www.python.org/downloads/)

### Step-by-Step Setup

1. **Open Terminal/Command Prompt**
   - Windows: Search for "Command Prompt" or "PowerShell" 
   - Mac: Open Terminal from Applications/Utilities
   - Linux: Open Terminal using Ctrl+Alt+T

2. **Clone the Repository with Git LFS**
   ```bash
   # Initialize Git LFS
   git lfs install
   
   # Clone the repository (this will also download the large model files)
   git clone https://github.com/AbdullaE100/NEW-MMSA.git
   
   # Enter the project directory
   cd NEW-MMSA
   ```

3. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   # Install all required packages
   pip install -r requirements.txt
   ```

5. **Create Necessary Directories**
   ```bash
   # Run the included setup script
   bash setup_models.sh
   ```

6. **Test Setup**
   ```bash
   # Run the setup test script to verify installation
   python setup_test.py
   ```

## Running the Application

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

## Text Sentiment Analysis

The system uses a RoBERTa-based model fine-tuned on Twitter data for state-of-the-art text sentiment analysis. This model provides more accurate and nuanced sentiment detection compared to generic models.

### Features:
- Pre-trained on over 124M tweets from 2018-2021
- Fine-tuned specifically for sentiment analysis
- 3-class classification: Positive, Neutral, Negative
- Processes social media language patterns effectively
- Handles emojis, slang, and informal text

The model is automatically used when transcribing speech from videos, enhancing the overall sentiment analysis.

To test the text sentiment analysis independently:
```bash
python test_text_sentiment.py
```

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

## Model Information

- **Visual Analysis**: Uses DeepFace for facial expression recognition
- **Audio Analysis**: Uses a CNN model trained on the RAVDESS emotional speech dataset
- **Text Analysis**: Uses transformers for sentiment classification
- **Speech Recognition**: Uses OpenAI Whisper with Google Speech Recognition as fallback

## License

MIT 