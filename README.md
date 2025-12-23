# Anomaly Detector Using Autoencoder

Lightweight project for detecting anomalies in videos using an image-based autoencoder.

Demo Video : https://www.loom.com/share/8b0ac1f46fd64d1f9d8507288b42ae09


## Overview

This repository trains an autoencoder on "normal" video frames so that high reconstruction error indicates anomalies. It includes training and detection scripts, utilities for preprocessing video frames, a Streamlit UI (`app.py`) for uploading videos, and saved model artifacts in the `models/` directory.

## Features

- Train an autoencoder on normal video frames (`src/train.py`).
- Real-time detection using webcam or video files (`src/detect.py`).
- Streamlit-based web UI for uploading videos and visualizing detected anomalous frames (`app.py`).
- Utilities for preprocessing, thresholding, and plotting training/detection metrics.

## Repository Structure

- `app.py` — Streamlit UI for uploading and analyzing videos.
- `models/` — Saved model and threshold (autoencoder.h5, threshold.npy).
- `src/` — Source code modules:
  - `data_utils.py` — Video/frame loading and preprocessing helpers.
  - `model.py` — Autoencoder architecture builder.
  - `train.py` — Training script to build and save the model and threshold.
  - `detect.py` — Real-time detection script (webcam or video file).
  - `utils.py` — Helper functions (errors, plotting, thresholding).
- `requirements.txt` — Python dependencies.
- `LICENSE` — Project license.

## Installation

1. Create a Python 3.8+ virtual environment (recommended).

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

```
opencv-python-headless
tensorflow
numpy
matplotlib
scikit-learn
tqdm
streamlit
```

## Usage

Prepare data:

- Place training videos containing only "normal" behavior in a `data/` folder at the repository root. Example formats: `.mp4`, `.avi`, `.mov`.

Train the autoencoder:

```bash
python src/train.py
```

This will:

- Load frames from `data/` (configurable in `src/train.py`).
- Train the autoencoder and save the model to `models/autoencoder.h5`.
- Compute and save an anomaly threshold to `models/threshold.npy`.

Run real-time detection (webcam or file):

```bash
python src/detect.py
```

By default `src/detect.py` uses the webcam (`video_source = 0`). To analyze a file, update `video_source` to the file path (or modify the script to accept CLI args).

Run the Streamlit UI:

```bash
streamlit run app.py
```

The UI allows uploading a video file and will analyze it using the trained model and threshold in `models/`.

## Notes & Tips

- Ensure you have enough "normal" training data—quality and variety improve detection.
- Training uses grayscale frames resized to 64x64 by default; modify settings in `src/train.py` and `src/data_utils.py` if needed.
- If the model is not present, `app.py` will prompt you to train first.

## Contributing

Contributions are welcome. Please open issues or pull requests with improvements, bug fixes, or questions.

## License

See the `LICENSE` file for license details.

## Contact

If you need help, open an issue or contact the repository owner.
# Abnormal Event Detection (ConvLSTM Autoencoder)

Project implementing a spatiotemporal autoencoder for video anomaly detection.

Quick start:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train:

```bash
python -m src.train --videos /path/to/train_videos --epochs 30 --batch_size 4
```

If your training data is a folder of frames (each clip in its own folder), run:

```bash
python -m src.train --videos data/Avenue/Train --data_type frames --epochs 30 --batch_size 4
```

3. Inference:

```bash
python -m src.inference --model models/conv_lstm_ae.h5 --video /path/to/test_video.mp4
```

For a frames folder as input (single clip folder):

```bash
python -m src.inference --model models/conv_lstm_ae.h5 --video data/Avenue/Test/Normal/clip01 --input_type frames
```

Files:
- `src/preprocess.py`: preprocessing and tf.data pipeline
- `src/model.py`: ConvLSTM autoencoder definition (detailed comments)
- `src/train.py`: training loop and loss plotting
- `src/inference.py`: compute reconstruction cost, regularity score, and plotting
