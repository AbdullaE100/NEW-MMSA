"""
Pure, dependency-light audio feature helpers.

Kept separate from the model code so the deterministic length-normalisation
logic can be unit-tested without loading TensorFlow, librosa, or model weights.
"""

import numpy as np


def fixed_length_signal(signal, target_length):
    """
    Normalise a 1-D signal to exactly ``target_length`` samples.

    Longer signals are center-cropped; shorter signals are center-padded with
    zeros. The operation is fully DETERMINISTIC — the same input always yields
    the same output. (An earlier implementation used ``np.random.randint`` for
    the crop/pad offset, which made inference non-reproducible.)

    Args:
        signal (np.ndarray): 1-D input signal.
        target_length (int): Desired output length in samples.

    Returns:
        np.ndarray: A 1-D array of exactly ``target_length`` samples.
    """
    signal = np.asarray(signal)
    length = len(signal)

    if length == target_length:
        return signal
    if length > target_length:
        offset = (length - target_length) // 2
        return signal[offset:offset + target_length]

    total_pad = target_length - length
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return np.pad(signal, (left_pad, right_pad), "constant")
