"""Tests for deterministic audio length-normalisation.

These guard the regression where inference used a random crop/pad offset,
making the same audio produce different predictions on each call.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_features import fixed_length_signal  # noqa: E402


def test_crop_is_deterministic():
    signal = np.arange(1000, dtype=np.float32)
    a = fixed_length_signal(signal, 400)
    b = fixed_length_signal(signal, 400)
    assert np.array_equal(a, b)
    assert len(a) == 400


def test_pad_is_deterministic():
    signal = np.arange(100, dtype=np.float32)
    a = fixed_length_signal(signal, 400)
    b = fixed_length_signal(signal, 400)
    assert np.array_equal(a, b)
    assert len(a) == 400


def test_crop_is_centered():
    signal = np.arange(10, dtype=np.float32)
    out = fixed_length_signal(signal, 4)
    # (10 - 4) // 2 == 3 -> samples [3, 4, 5, 6]
    assert np.array_equal(out, np.array([3, 4, 5, 6], dtype=np.float32))


def test_pad_is_centered_and_symmetric():
    signal = np.array([5, 6], dtype=np.float32)
    out = fixed_length_signal(signal, 6)
    # two zeros on each side of the two real samples
    assert np.array_equal(out, np.array([0, 0, 5, 6, 0, 0], dtype=np.float32))


def test_exact_length_passthrough():
    signal = np.arange(216, dtype=np.float32)
    out = fixed_length_signal(signal, 216)
    assert np.array_equal(out, signal)


@pytest.mark.parametrize("length", [1, 50, 216, 500, 5000])
def test_always_returns_target_length(length):
    signal = np.zeros(length, dtype=np.float32)
    assert len(fixed_length_signal(signal, 216)) == 216
