---
title: Multimodal Sentiment Analysis
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.17.0
app_file: app.py
pinned: false
---

# Multimodal Sentiment Analysis (MMSA)

A comprehensive tool for analyzing sentiment across multiple modalities (visual, audio, and text) from video input.

## Features

- **Visual Analysis**: Analyzes facial expressions using DeepFace with multiple detector backends
- **Audio Analysis**: Analyzes vocal emotions using a CNN model trained on RAVDESS dataset
- **Text Analysis**: Analyzes sentiment from speech transcripts using RoBERTa with context enrichment
- **Multimodal Fusion**: Combines all modalities for more accurate sentiment prediction
- **SHAP Visualizations**: Explains AI decisions by showing how each feature contributes to the final sentiment score
- **Enhanced UI**: Intuitive interface with tabs for analysis results and SHAP explanations

## User Interface

![Upload Media UI](images/upload%20media%20ui.png)
*The main interface where users can upload videos for analysis*

![Sentiment Breakdown](images/Sentiment%20breakdown.png)
*Detailed breakdown of sentiment analysis across all modalities*

![SHAP Analysis](images/Shap%20Analysis.png)
*SHAP visualizations showing how each feature contributes to the sentiment prediction*

## How to Run

### Standard Interface with SHAP Visualizations

```bash
# Run with SHAP visualizations
python mmsa_interface_with_shap.py
```

### Create a Public Shareable Link

```bash
# Run with a public share link (active for 72 hours)
python mmsa_interface_with_shap.py --share
```

### Additional Options

```bash
# Change the port (default is 7860)
python mmsa_interface_with_shap.py --port 8000

# Specify a different face detector backend
python mmsa_interface_with_shap.py --detector opencv

# Specify custom model paths
python mmsa_interface_with_shap.py --audio-model ./my_models/audio_model.h5 --norm-params ./my_models/norm_params.json
```

## How to Use

1. Upload a video using the interface
2. Click the "Analyze Sentiment" button
3. View the overall sentiment analysis with scores from all modalities
4. Explore SHAP visualizations to understand how each modality contributed to the prediction
5. See the detailed breakdown of visual, audio, and text sentiment analysis

## Technical Details

This application utilizes pre-trained models for each modality:
- **Visual Processing**: DeepFace for facial emotion detection
- **Audio Analysis**: CNN model trained on the RAVDESS emotional speech dataset
- **Text Processing**: RoBERTa with context enrichment for speech transcript analysis
- **SHAP Integration**: SHapley Additive exPlanations for AI interpretability

## Models

The models used in this application are stored in the repository:
- Audio sentiment model: `models/audio_sentiment_model_notebook.h5`
- Audio normalization parameters: `models/audio_norm_params.json`

## Installation

```bash
# Clone the repository
git clone https://github.com/AbdullaE100/NEW-MMSA.git
cd NEW-MMSA

# Install dependencies
pip install -r requirements.txt

# Run the interface
python mmsa_interface_with_shap.py
```

---

Created by AbdullaE 
Repository: https://github.com/AbdullaE100/NEW-MMSA 