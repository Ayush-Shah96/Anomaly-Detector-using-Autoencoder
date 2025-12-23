"""
Data utilities for loading and preprocessing video frames.
"""

import cv2
import numpy as np
from tqdm import tqdm

def load_video_frames(video_path, frame_skip=1, resize=(64, 64), grayscale=True):
    """
    Load frames from a video file.

    Args:
        video_path (str): Path to the video file.
        frame_skip (int): Skip every nth frame to reduce data.
        resize (tuple): Target size for resizing frames.
        grayscale (bool): Convert to grayscale if True.

    Returns:
        np.ndarray: Array of preprocessed frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames // frame_skip, desc="Loading frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_skip == 0:
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, resize)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
                pbar.update(1)
            count += 1

    cap.release()
    return np.array(frames)

def preprocess_frame(frame, resize=(64, 64), grayscale=True):
    """
    Preprocess a single frame.

    Args:
        frame: Input frame.
        resize (tuple): Target size.
        grayscale (bool): Convert to grayscale.

    Returns:
        np.ndarray: Preprocessed frame.
    """
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, resize)
    frame = frame.astype(np.float32) / 255.0
    return frame