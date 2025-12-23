"""
Utility functions for anomaly detection.
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_reconstruction_error(original, reconstructed):
    """
    Compute mean squared error for reconstruction.

    Args:
        original (np.ndarray): Original frames.
        reconstructed (np.ndarray): Reconstructed frames.

    Returns:
        np.ndarray: Reconstruction errors.
    """
    return np.mean((original - reconstructed) ** 2, axis=(1, 2, 3))

def set_anomaly_threshold(errors, percentile=95):
    """
    Set anomaly threshold based on training errors.

    Args:
        errors (np.ndarray): Reconstruction errors from training.
        percentile (float): Percentile for threshold.

    Returns:
        float: Threshold value.
    """
    return np.percentile(errors, percentile)

def is_anomaly(error, threshold):
    """
    Check if error indicates an anomaly.

    Args:
        error (float): Reconstruction error.
        threshold (float): Anomaly threshold.

    Returns:
        bool: True if anomaly.
    """
    return error > threshold

def plot_training_loss(history, save_path='training_loss.png'):
    """
    Plot training and validation loss.

    Args:
        history: Keras history object.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def plot_reconstruction_errors(errors, threshold, save_path='reconstruction_errors.png'):
    """
    Plot reconstruction errors.

    Args:
        errors (np.ndarray): Errors.
        threshold (float): Threshold.
        save_path (str): Path to save.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label='Reconstruction Error')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Reconstruction Errors')
    plt.xlabel('Frame')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(save_path)
    plt.show()