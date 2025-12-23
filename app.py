"""
Streamlit UI for Anomaly Detection in Videos using Autoencoder.
"""

import streamlit as st
import os
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from src.data_utils import preprocess_frame
import cv2

st.title("Video Anomaly Detection with Autoencoder")

st.write("Upload a video to check if it contains anomalies.")

uploaded_video = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])

if uploaded_video is not None:
    # Save video temporarily
    temp_video_path = os.path.join(tempfile.mkdtemp(), uploaded_video.name)
    with open(temp_video_path, 'wb') as f:
        f.write(uploaded_video.getbuffer())

    if st.button("Check for Anomalies"):
        if not os.path.exists('models/autoencoder.h5'):
            st.error("Model not trained yet. Please train the model first.")
        else:
            try:
                with st.spinner("Analyzing video..."):
                    # Load model and threshold
                    model = load_model('models/autoencoder.h5', compile=False)
                    model.compile(optimizer='adam', loss='mse')
                    threshold = np.load('models/threshold.npy')

                    # Process video
                    cap = cv2.VideoCapture(temp_video_path)
                    if not cap.isOpened():
                        st.error("Could not open the video file. Please check the format.")
                        st.stop()

                    anomalies = []
                    anomalous_frames = []
                    total_frames = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        total_frames += 1

                        # Preprocess
                        processed = preprocess_frame(frame, resize=(64, 64), grayscale=True)
                        input_frame = processed.reshape(1, 64, 64, 1)

                        # Predict
                        reconstructed = model.predict(input_frame, verbose=0)
                        error = np.mean((input_frame - reconstructed) ** 2)

                        if error > threshold:
                            anomalies.append(total_frames)
                            if len(anomalous_frames) < 5:  # Collect up to 5 frames
                                anomalous_frames.append(frame.copy())

                    cap.release()

                    # Display results
                    if anomalies:
                        st.error(f"Yes, the video has anomalies! Detected at {len(anomalies)} frames out of {total_frames} total frames.")
                        st.write(f"Anomalous frames: {anomalies[:10]}{'...' if len(anomalies) > 10 else ''}")

                        # Display anomalous frames
                        st.write("Sample anomalous frames:")
                        cols = st.columns(min(len(anomalous_frames), 5))
                        for i, frame in enumerate(anomalous_frames):
                            cols[i].image(frame, channels="BGR", caption=f"Frame {anomalies[i]}")
                    else:
                        st.success(f"No anomalies detected in the video. All {total_frames} frames are normal.")

                    # Display video
                    st.video(temp_video_path)
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.write("Please try again or check the video file.")