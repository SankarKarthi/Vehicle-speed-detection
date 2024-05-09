import cv2
import numpy as np
import streamlit as st

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe(r"C:\Studies\DPS\speedy\models\deploy.prototxt", r"C:\Studies\DPS\speedy\models\mobilenet_iter_73000.caffemodel")

# Function to calculate Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return np.sqrt(np.sum((pt1 - pt2) ** 2))

# Function to perform object detection and motion speed estimation
def detect_objects_and_speed(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    _, frame = cap.read()

    # Initialize variables for object tracking
    object_center_prev = None
    object_speed = None

    # Reference points (in pixels) for speed calculation
    ref_point1 = np.array([100, 400])  # Adjust these points according to your video
    ref_point2 = np.array([600, 400])

    # Known distance between reference points (in meters)
    distance_meters = 10  # Adjust this according to the real-world distance between the reference points

    video_results = []

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.5:
                # Extract the index of the class label from the 'detections'
                idx = int(detections[0, 0, i, 1])

                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"Object {idx}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Object tracking
                object_center_curr = np.array([(startX + endX) / 2, (startY + endY) / 2])

                if object_center_prev is not None:
                    # Calculate Euclidean distance between current and previous object centers
                    distance_pixels = euclidean_distance(object_center_curr, object_center_prev)

                    # Calculate vehicle speed
                    time_seconds = 1 / 30  # Assuming video is captured at 30 frames per second
                    object_speed = (distance_meters / distance_pixels) / time_seconds  # Speed in meters per second

                    # Convert speed to kilometers per hour (km/h)
                    object_speed_kmh = object_speed * 3.6

                    # Display speed
                    cv2.putText(frame, f"Speed: {object_speed_kmh:.2f} km/h", (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                object_center_prev = object_center_curr

        # Append frame with annotations to the result list
        video_results.append(frame)

    # Release the video capture object
    cap.release()

    # Display the video with annotations using Streamlit
    for frame in video_results:
        st.image(frame, channels="BGR")

# Streamlit app
def main():
    st.title("Object Detection and Speed Estimation")

    # Video file path
    video_file_path = r"C:/Users/sanka/Downloads/input_video.mp4"

    # Perform object detection and motion speed estimation
    detect_objects_and_speed(video_file_path)

if __name__ == "__main__":
    main()
