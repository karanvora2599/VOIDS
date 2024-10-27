import cv2
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

def counter(video_path, frame_skip=1, resize_dim=(640, 640), max_age=70, min_confidence=0.5):
    # Check if CUDA (NVIDIA GPU) is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the YOLOv8 model with GPU (if available)
    model = YOLO('yolov8n.pt')  # YOLOv8n is the nano version for speed
    
    # Set model to use the correct device (GPU or CPU)
    model.to(device)

    # Initialize DeepSORT tracker with stricter parameters to reduce overcounting
    tracker = DeepSort(
        max_age=max_age,  # Increase max age to hold onto track IDs longer
        n_init=3,         # Require 3 consistent detections to confirm a new person
        nms_max_overlap=1.0,  # Non-max suppression to avoid overlapping boxes
        max_iou_distance=0.7,  # Higher IOU distance for bounding box matching
        max_cosine_distance=0.4  # Lower cosine distance for appearance matching (use stricter appearance-based tracking)
    )

    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    unique_people_count = set()  # To track unique people by their IDs
    frame_count = 0

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to speed up processing (adjust frame_skip to control performance)
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Resize the frame to speed up detection
        frame = cv2.resize(frame, resize_dim)

        # Predict on the current frame using the selected device
        results = model(frame, classes=[0], device=device, verbose=False)  # class 0 is for 'person'

        # Extract bounding boxes and confidences for people in the frame
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
        confidences = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else []

        # Filter out low confidence detections
        filtered_detections = [
            (box, confidence, 0) for box, confidence in zip(boxes, confidences) if confidence >= min_confidence
        ]  # 0 is the class ID for person

        # Update the tracker with current frame detections
        tracked_objects = tracker.update_tracks(filtered_detections, frame=frame)

        # Update unique people count based on tracked object IDs
        for track in tracked_objects:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id  # Unique ID assigned by DeepSORT
            unique_people_count.add(track_id)

        frame_count += 1

    cap.release()

    # Return the number of unique people detected in the video
    return len(unique_people_count)