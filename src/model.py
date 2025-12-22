"""Spatiotemporal Autoencoder using ConvLSTM2D layers.

This file builds a ConvLSTM-based autoencoder that accepts a sequence of
grayscale frames shaped (seq_len, H, W, 1) and reconstructs the same sequence.

Detailed tensor flow comments are provided around the ConvLSTM stack to make
it clear how sequence- and spatial-dimensions are transformed.
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_conv_lstm_autoencoder(seq_len: int = 10, height: int = 227, width: int = 227, channels: int = 1) -> tf.keras.Model:
    """Builds a symmetric ConvLSTM autoencoder.

    Architecture (high level):
      Input: (batch, seq_len, H, W, C)
      Encoder: several ConvLSTM2D layers (return_sequences=True), producing
               spatiotemporal feature maps with same temporal length.
      Decoder: mirrored ConvLSTM2D layers (return_sequences=True) and a
               final TimeDistributed Conv2D to map features back to C channels.

    Note: We keep return_sequences=True throughout so the decoder reconstructs
    a full sequence. Keeping sequence length intact simplifies mapping between
    input frames and reconstructed frames.

    Returns:
        model: compiled keras Model (uncompiled - user compiles/trains).
    """
    inp = layers.Input(shape=(seq_len, height, width, channels), name="input_sequence")

    # ---------------- Encoder ----------------
    # After first ConvLSTM2D:
    #   shape = (batch, seq_len, H, W, 64)
    x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, name='enc_conv_lstm_1')(inp)
    x = layers.BatchNormalization(name='enc_bn_1')(x)

    # 2nd ConvLSTM2D compresses feature channels while keeping temporal length:
    #   shape = (batch, seq_len, H, W, 32)
    x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, name='enc_conv_lstm_2')(x)
    x = layers.BatchNormalization(name='enc_bn_2')(x)

    # 3rd ConvLSTM2D - acts as the bottleneck representation (still sequence):
    #   shape = (batch, seq_len, H, W, 16)
    x = layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True, name='enc_conv_lstm_3')(x)
    x = layers.BatchNormalization(name='enc_bn_3')(x)

    # ---------------- Decoder ----------------
    # Mirror: expand channels via ConvLSTM layers. Temporal length remains seq_len.
    #   shape -> (batch, seq_len, H, W, 32)
    x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, name='dec_conv_lstm_1')(x)
    x = layers.BatchNormalization(name='dec_bn_1')(x)

    #   shape -> (batch, seq_len, H, W, 64)
    x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, name='dec_conv_lstm_2')(x)
    x = layers.BatchNormalization(name='dec_bn_2')(x)

    # Final mapping to single-channel frame per timestep:
    # TimeDistributed Conv2D maps (H, W, 64) -> (H, W, channels)
    # Output shape: (batch, seq_len, H, W, channels)
    out = layers.TimeDistributed(layers.Conv2D(filters=channels, kernel_size=(3, 3), activation='sigmoid', padding='same'), name='reconstruction_conv')(x)

    model = models.Model(inputs=inp, outputs=out, name='conv_lstm_autoencoder')

    return model


if __name__ == "__main__":
    m = build_conv_lstm_autoencoder()
    m.summary()
