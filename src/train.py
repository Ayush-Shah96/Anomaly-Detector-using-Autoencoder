"""Training script for ConvLSTM autoencoder.

Usage example:
    python -m src.train --videos /path/to/train_videos --epochs 50 --batch_size 4
"""
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocess import dataset_from_dir
from src.model import build_conv_lstm_autoencoder


def train(videos_dir: str, epochs: int = 30, batch_size: int = 4, seq_len: int = 10, model_out: str = "models/conv_lstm_ae.h5", data_type: str = 'video'):
    ds = dataset_from_dir(videos_dir, seq_len=seq_len, batch_size=batch_size, data_type=data_type)

    # Build model
    # Input shape: (batch, seq_len, H, W, 1)
    sample = next(iter(ds))
    seq_len_s, h, w, c = sample.shape[1:]

    model = build_conv_lstm_autoencoder(seq_len=seq_len_s, height=h, width=w, channels=c)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    Path(os.path.dirname(model_out)).mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_out, save_best_only=True, monitor='loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-7)
    ]

    history = model.fit(ds, epochs=epochs, callbacks=callbacks)

    # Plot loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training Loss')
    plt.tight_layout()
    plt.savefig('training_loss.png')
    print('Saved training_loss.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--model_out', type=str, default='models/conv_lstm_ae.h5')
    parser.add_argument('--data_type', type=str, default='video', choices=['video', 'frames'], help='Type of input data: video files or frame folders')
    args = parser.parse_args()
    train(args.videos, epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len, model_out=args.model_out, data_type=args.data_type)
