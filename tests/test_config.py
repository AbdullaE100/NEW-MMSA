"""Sanity checks on the central configuration constants."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402


def test_fusion_weights_present():
    assert set(config.FUSION_WEIGHTS) == {"visual", "audio", "text"}


def test_fusion_weights_positive():
    assert all(w > 0 for w in config.FUSION_WEIGHTS.values())


def test_thresholds_ordered():
    assert config.NEGATIVE_THRESHOLD < 0 < config.POSITIVE_THRESHOLD


def test_audio_geometry_is_sane():
    assert config.AUDIO_SAMPLING_RATE > 0
    assert config.AUDIO_DURATION_SECONDS > 0
    assert config.AUDIO_N_MFCC > 0
    assert config.AUDIO_TARGET_FRAMES > 0
