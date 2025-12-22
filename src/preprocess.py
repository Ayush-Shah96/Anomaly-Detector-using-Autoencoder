"""Video preprocessing utilities.

This module reads video files, converts frames to grayscale, resizes to 227x227,
groups frames into temporal sequences (default length 10), and normalizes pixel
values to [0, 1]. It exposes a `dataset_from_dir` helper which returns a
`tf.data.Dataset` suitable for training a spatiotemporal autoencoder.
"""
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Generator


DEFAULT_HEIGHT = 227
DEFAULT_WIDTH = 227


def extract_frames(video_path: str, resize=(DEFAULT_WIDTH, DEFAULT_HEIGHT)) -> List[np.ndarray]:
    """Read video and return list of grayscale, resized frames normalized to [0,1].

    Args:
        video_path: Path to video file.
        resize: (width, height) tuple.

    Returns:
        frames: list of np.ndarray with shape (H, W), dtype float32, values in [0,1].
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, resize, interpolation=cv2.INTER_AREA)
        # Normalize to [0,1]
        normalized = resized.astype(np.float32) / 255.0
        frames.append(normalized)

    cap.release()
    return frames


def frames_to_sequences(frames: List[np.ndarray], seq_len: int = 10, stride: int = 1) -> Generator[np.ndarray, None, None]:
    """Yield overlapping sequences from frame list.

    Each yielded sequence has shape (seq_len, H, W, 1).
    """
    n = len(frames)
    for start in range(0, n - seq_len + 1, stride):
        seq = frames[start:start + seq_len]
        arr = np.stack(seq, axis=0)  # (seq_len, H, W)
        arr = arr[..., np.newaxis]  # (seq_len, H, W, 1)
        yield arr.astype(np.float32)


def video_to_sequences(video_path: str, seq_len: int = 10, stride: int = 1, resize=(DEFAULT_WIDTH, DEFAULT_HEIGHT)) -> List[np.ndarray]:
    """Read a video and return list of sequences ready for model input.

    Returns list of np.ndarrays of shape (seq_len, H, W, 1).
    """
    frames = extract_frames(video_path, resize=resize)
    return list(frames_to_sequences(frames, seq_len=seq_len, stride=stride))


def frames_folder_to_sequences(folder_path: str, seq_len: int = 10, stride: int = 1, resize=(DEFAULT_WIDTH, DEFAULT_HEIGHT)) -> List[np.ndarray]:
    """Read a folder of image frames and return list of sequences.

    Supports common image extensions and expects frames to be named so that
    sorting them lexicographically yields temporal order (e.g. frame0001.png).
    Returns list of np.ndarrays of shape (seq_len, H, W, 1).
    """
    folder = Path(folder_path)
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    img_paths = []
    for e in exts:
        img_paths.extend(sorted(folder.glob(e)))
    img_paths = sorted(img_paths)
    frames = []
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        frames.append(normalized)

    return list(frames_to_sequences(frames, seq_len=seq_len, stride=stride))


def dataset_from_dir(videos_dir: str, seq_len: int = 10, batch_size: int = 4, shuffle: bool = True, resize=(DEFAULT_WIDTH, DEFAULT_HEIGHT), data_type: str = 'video') -> tf.data.Dataset:
    """Create a tf.data.Dataset yielding batches of sequences.

    The output shape is (batch_size, seq_len, H, W, 1) with dtype float32.
    """
    videos_dir = Path(videos_dir)

    def generator_video():
        video_paths = sorted(videos_dir.glob("**/*.mp4")) + sorted(videos_dir.glob("**/*.avi"))
        for vp in video_paths:
            for seq in video_to_sequences(str(vp), seq_len=seq_len, resize=resize):
                yield seq

    def generator_frames():
        # each subdirectory is considered a clip containing frames
        for sub in sorted(videos_dir.iterdir()):
            if not sub.is_dir():
                continue
            for seq in frames_folder_to_sequences(str(sub), seq_len=seq_len, stride=1, resize=resize):
                yield seq

    if data_type == 'video':
        gen = generator_video
    elif data_type == 'frames':
        gen = generator_frames
    else:
        raise ValueError("data_type must be 'video' or 'frames'")

    sample_seq = next(gen(), None)
    if sample_seq is None:
        raise FileNotFoundError(f"No sequences found in {videos_dir} for data_type={data_type}")

    output_signature = tf.TensorSpec(shape=sample_seq.shape, dtype=tf.float32)

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=512)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=str, required=True, help="Directory with video files")
    parser.add_argument("--seq_len", type=int, default=10)
    args = parser.parse_args()

    ds = dataset_from_dir(args.videos, seq_len=args.seq_len, batch_size=2)
    for batch in ds.take(1):
        print("Batch shape:", batch.shape)
