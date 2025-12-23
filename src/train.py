"""
Training script for the autoencoder on normal video data.
"""

import os
import glob
import numpy as np
from src.data_utils import load_video_frames
from src.model import build_autoencoder
from src.utils import compute_reconstruction_error, set_anomaly_threshold, plot_training_loss
from tensorflow.keras.models import save_model

def main():
    # Configuration
    data_folder = 'data/'
    frame_skip = 5  # Skip frames to reduce data
    resize = (64, 64)
    grayscale = True
    epochs = 50
    batch_size = 32
    validation_split = 0.1

    # Find all video files in data folder
    video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv']  # Add more if needed
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(data_folder, ext)))

    if not video_paths:
        print(f"No video files found in {data_folder}")
        return

    print(f"Found {len(video_paths)} video files: {video_paths}")

    # Load frames from all videos
    all_frames = []
    for path in video_paths:
        print(f"Loading frames from {path}")
        frames = load_video_frames(path, frame_skip=frame_skip, resize=resize, grayscale=grayscale)
        all_frames.extend(frames)

    frames = np.array(all_frames)
    print(f"Total frames loaded: {len(frames)}")

    # Reshape for model input
    if grayscale:
        frames = frames.reshape(frames.shape + (1,))
    else:
        # If RGB, no need to reshape if already 3 channels
        pass

    input_shape = frames.shape[1:]
    print(f"Input shape: {input_shape}")

    # Build model
    autoencoder = build_autoencoder(input_shape)

    # Train model
    print("Training autoencoder...")
    history = autoencoder.fit(
        frames, frames,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # Save model
    os.makedirs('models', exist_ok=True)
    save_model(autoencoder, 'models/autoencoder.h5')
    print("Model saved to models/autoencoder.h5")

    # Compute reconstruction errors on training data
    reconstructed = autoencoder.predict(frames)
    errors = compute_reconstruction_error(frames, reconstructed)

    # Set threshold
    threshold = set_anomaly_threshold(errors)
    print(f"Anomaly threshold: {threshold}")

    # Save threshold
    np.save('models/threshold.npy', threshold)

    # Plot training loss
    plot_training_loss(history)

if __name__ == "__main__":
    main()