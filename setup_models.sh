#!/bin/bash

# Create necessary directories
mkdir -p models results/gradio_results results/batch_results examples tmp_audio

echo "Directory structure created."
echo ""
echo "IMPORTANT: You need to manually copy the following model files:"
echo "1. Place the audio model at: models/audio_sentiment_model_notebook.h5"
echo "2. Place the normalization parameters at: models/audio_norm_params.json"
echo ""
echo "These files are not included in the repository due to their size."
echo "Please copy them from your original project or use Git LFS to download them." 