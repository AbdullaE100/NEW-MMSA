#!/usr/bin/env python3

"""
Hugging Face Spaces entry point for the Multimodal Sentiment Analysis app.

Spaces looks for a top-level ``app.py`` that exposes a Gradio ``demo``. This
module builds the same interface used by ``mmsa_interface_with_shap.py`` so the
hosted app and the local app stay identical.
"""

import os
import sys
import logging

# Make sure both the repo root and ``src`` are importable.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mmsa.spaces")

from mmsa_interface_with_shap import MultimodalSentimentWithSHAP, create_interface

# Build the analyzer and the Gradio interface. Any failure here should surface
# loudly in the Space logs rather than being swallowed into a half-dead app.
analyzer = MultimodalSentimentWithSHAP()
demo = create_interface(analyzer)

# Hugging Face Spaces imports this module and launches ``demo`` itself, but we
# also support running it directly with ``python app.py``.
if __name__ == "__main__":
    demo.launch()
