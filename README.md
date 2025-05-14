# Multimodal Sentiment Analysis (MMSA)

A comprehensive tool for analyzing sentiment across multiple modalities (visual, audio, and text) from video input.

## Overview

This Multimodal Sentiment Analysis (MMSA) tool analyzes videos through three modalities:
- **Visual**: Facial expressions using DeepFace
- **Audio**: Speech emotions using a CNN model trained on the RAVDESS dataset
- **Text**: Sentiment analysis on automatically transcribed speech

The system combines the analysis from these three modalities with the following weights:
- Visual (facial expressions): 45%
- Audio (speech emotions): 45%
- Text (transcribed speech): 10%

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

## Usage

### Web Interface

Run the Gradio web interface:

```bash
python run_mmsa.py web
```

To create a public shareable URL:

```bash
python run_mmsa.py web --share
```

### Batch Processing

Process multiple videos at once:

```bash
python run_mmsa.py batch --input /path/to/videos --output /path/to/results
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