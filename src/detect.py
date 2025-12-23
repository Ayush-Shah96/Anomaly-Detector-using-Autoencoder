"""
Real-time anomaly detection in video streams.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.data_utils import preprocess_frame
from src.utils import compute_reconstruction_error, is_anomaly
import matplotlib.pyplot as plt

def main():
    # Configuration
    model_path = 'models/autoencoder.h5'
    threshold_path = 'models/threshold.npy'
    resize = (64, 64)
    grayscale = True

    # Load model and threshold
    model = load_model(model_path)
    threshold = np.load(threshold_path)
    print(f"Loaded model and threshold: {threshold}")

    # Video capture: 0 for webcam, or path to video file
    video_source = 0  # Change to 'data/test_video.mp4' for file
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    errors_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        processed = preprocess_frame(frame, resize=resize, grayscale=grayscale)
        input_frame = processed.reshape(1, *processed.shape, 1) if grayscale else processed.reshape(1, *processed.shape)

        # Reconstruct
        reconstructed = model.predict(input_frame, verbose=0)

        # Compute error
        error = compute_reconstruction_error(input_frame, reconstructed)[0]
        errors_list.append(error)

        # Check for anomaly
        anomaly = is_anomaly(error, threshold)

        # Display
        display_frame = frame.copy()
        if anomaly:
            cv2.putText(display_frame, 'ANOMALY DETECTED', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)

        # Show error
        cv2.putText(display_frame, f'Error: {error:.4f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Anomaly Detection', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot reconstruction errors after detection
    from src.utils import plot_reconstruction_errors
    plot_reconstruction_errors(np.array(errors_list), threshold)

if __name__ == "__main__":
    main()