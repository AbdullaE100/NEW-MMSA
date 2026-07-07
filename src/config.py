"""
Central configuration for the Multimodal Sentiment Analysis pipeline.

Every tunable constant that affects a prediction lives here so that the values
are visible in one place, documented, and easy to change or sweep. Previously
these numbers were scattered as inline literals across several modules with
different values, which made the system's behaviour hard to reason about.

Important honesty note: the fusion weights below are HAND-TUNED, not learned
from data. They are a reasonable prior, not a validated result. See MODEL_CARD.md.
"""

# ---------------------------------------------------------------------------
# Multimodal fusion
# ---------------------------------------------------------------------------
# Relative weights for the late-fusion weighted average of per-modality
# sentiment scores. They do not need to sum to 1.0 — the fusion code
# renormalises over whichever modalities are actually present for a given input.
FUSION_WEIGHTS = {
    "visual": 0.45,
    "audio": 0.45,
    "text": 0.10,
}

# ---------------------------------------------------------------------------
# Sentiment score -> label thresholds
# ---------------------------------------------------------------------------
# A combined score in [-1, 1] is bucketed into a discrete label using these
# cut-offs. Scores strictly above POSITIVE_THRESHOLD are "Positive", strictly
# below NEGATIVE_THRESHOLD are "Negative", and everything in between "Neutral".
POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15

# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------
# Fixed input geometry the audio CNN was trained on. Feature extraction is
# deterministic (centered crop/pad) so a given clip always yields the same
# prediction.
AUDIO_SAMPLING_RATE = 44100
AUDIO_DURATION_SECONDS = 2.5
AUDIO_N_MFCC = 30
AUDIO_TARGET_FRAMES = 216
