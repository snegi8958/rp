import cv2
from fer import FER
from ultralytics import YOLO
from collections import Counter
import streamlit as st
import tempfile
import os

# Initialize the emotion detector
emotion_detector = FER()

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize the Streamlit app
st.title("Emotion Detection from Video")

# Upload video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name

    # Read the video file
    cap = cv2.VideoCapture(temp_file_path)

    # Check if video opened successfully
    if not cap.isOpened():
        st.error("Error opening video file.")
    else:
        # Video writer to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

        # Emotion tracking
        emotion_counter = Counter()
        frame_skip = 5  # Process every 5th frame

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame
            if frame_count % frame_skip == 0:
                # Run YOLO detection
                results = model(frame)

                # Process results
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        face = frame[y1:y2, x1:x2]  # Crop the face

                        # Detect emotions
                        emotions = emotion_detector.detect_emotions(face)
                        if emotions:
                            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                            emotion_text = f"{dominant_emotion}"

                            # Update emotion counter
                            emotion_counter[dominant_emotion] += 1

                            # Draw rectangle and label for emotion
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, emotion_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Calculate total detections
                total_detections = sum(emotion_counter.values())
                
                # Display emotion statistics dynamically on the video
                stats_text = "Emotion Statistics:"
                if total_detections > 0:
                    for emotion, count in emotion_counter.items():
                        percentage = (count / total_detections) * 100
                        stats_text += f" {emotion}: {percentage:.1f}%;"
                
                # Display the statistics on the frame
                cv2.putText(frame, stats_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Write the frame to the output video
                out.write(frame)

                # Display the resulting frame in Streamlit
                st.image(frame, channels="BGR", use_column_width=True)

            frame_count += 1

        # Release everything
        cap.release()
        out.release()
        os.remove(temp_file_path)  # Clean up the temporary file

        # Final emotion statistics
        st.write("Final Emotion Statistics:")
        for emotion, count in emotion_counter.items():
            percentage = (count / total_detections) * 100
            st.write(f"{emotion}: {percentage:.1f}%")
        st.write("Processed video saved as: output_video.mp4")