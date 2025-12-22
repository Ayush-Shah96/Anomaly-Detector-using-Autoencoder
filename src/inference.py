"""Inference utilities for anomaly detection.

Calculates reconstruction cost (Euclidean distance) between input and
reconstructed sequences and produces a regularity score in [0,1]. Also
provides plotting code to visualize scores over time and highlight anomalies.
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocess import video_to_sequences, extract_frames, frames_folder_to_sequences


def reconstruction_cost(x: np.ndarray, x_rec: np.ndarray) -> float:
    """Compute Euclidean reconstruction cost between sequence pairs.

    x, x_rec shapes: (seq_len, H, W, C)
    Returns scalar cost for the sequence (root mean squared error over all pixels/time).
    """
    diff = x - x_rec
    mse = np.mean(np.square(diff))
    return float(np.sqrt(mse))


def compute_scores_for_video(model: tf.keras.Model, input_path: str, seq_len: int = 10, stride: int = 1, input_type: str = 'video') -> Tuple[List[float], List[int]]:
    """Process a video file or a folder of frames and compute reconstruction costs.

    Args:
        model: loaded keras model
        input_path: path to video file or frames folder
        seq_len: sequence length
        stride: stride between sequences
        input_type: 'video' or 'frames'

    Returns:
        costs: list of reconstruction costs (one per sequence)
        center_frame_indices: list of integers (center frame index in the original clip for each sequence)
    """
    if input_type == 'video':
        sequences = list(video_to_sequences(input_path, seq_len=seq_len, stride=stride))
        # number of original frames can be retrieved by extracting frames
        frames = extract_frames(input_path)
    elif input_type == 'frames':
        sequences = list(frames_folder_to_sequences(input_path, seq_len=seq_len, stride=stride))
        # get number of frames by listing files
        frames = []
        folder = Path(input_path)
        for e in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            frames.extend(sorted(folder.glob(e)))
    else:
        raise ValueError("input_type must be 'video' or 'frames'")

    if len(sequences) == 0:
        return [], []

    seqs = np.stack(sequences, axis=0)  # (num_seq, seq_len, H, W, 1)
    preds = model.predict(seqs, batch_size=4)

    costs = []
    center_indices = []
    for i in range(seqs.shape[0]):
        cost = reconstruction_cost(seqs[i], preds[i])
        costs.append(cost)
        center_idx = i + seq_len // 2
        center_indices.append(center_idx)

    return costs, center_indices


def regularity_scores_from_costs(costs: List[float], eps: float = 1e-8) -> List[float]:
    """Convert raw reconstruction costs into regularity scores in [0,1].

    A simple min-max normalization is used, then inverted so lower cost -> higher regularity.
    """
    if len(costs) == 0:
        return []
    c = np.array(costs)
    cmin, cmax = c.min(), c.max()
    if cmax - cmin < eps:
        return [1.0] * len(c)
    norm = (c - cmin) / (cmax - cmin + eps)
    scores = 1.0 - norm
    return scores.tolist()


def plot_scores(center_indices: List[int], scores: List[float], threshold: float = 0.5, out_path: str = 'regularity_plot.png') -> None:
    """Plot regularity scores over frame index and highlight anomalies.

    Frames with score < threshold are highlighted in red.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(center_indices, scores, label='regularity_score')
    anomalies_x = [x for x, s in zip(center_indices, scores) if s < threshold]
    anomalies_y = [s for s in scores if s < threshold]
    if anomalies_x:
        plt.scatter(anomalies_x, [scores[center_indices.index(x)] for x in anomalies_x], color='red', label='anomaly')
    plt.axhline(threshold, color='orange', linestyle='--', label=f'threshold={threshold}')
    plt.xlabel('Frame index (center of sequence)')
    plt.ylabel('Regularity score (1 = normal)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved model (.h5)')
    parser.add_argument('--video', required=True, help='Path to test video')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--input_type', type=str, default='video', choices=['video','frames'], help='Type of input: video file or folder of frames')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    costs, centers = compute_scores_for_video(model, args.video, seq_len=args.seq_len, input_type=args.input_type)
    scores = regularity_scores_from_costs(costs)
    plot_scores(centers, scores, threshold=args.threshold, out_path='regularity_plot.png')
    print('Saved regularity_plot.png')
