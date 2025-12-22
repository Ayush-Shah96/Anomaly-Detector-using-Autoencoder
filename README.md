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
