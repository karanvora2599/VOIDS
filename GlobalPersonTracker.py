import cv2
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

def counter(video_path, frame_skip=1, resize_dim=(640, 640), max_age=70, min_confidence=0.5):
    """
    Counts the number of unique people in a video chunk and maintains a global count across all chunks.

    Args:
        video_path (str): Path to the video chunk.
        frame_skip (int, optional): Number of frames to skip for processing. Defaults to 1.
        resize_dim (tuple, optional): Dimensions to resize each frame. Defaults to (640, 640).
        max_age (int, optional): Maximum age for tracker. Defaults to 70.
        min_confidence (float, optional): Minimum confidence threshold for detections. Defaults to 0.5.

    Returns:
        dict: A dictionary containing:
            - 'count': Number of unique people in the current chunk.
            - 'global_count': Total number of unique people across all processed chunks.
    """

    import json  # Importing here to ensure the function is self-contained

    # Initialize function attributes on the first call
    if not hasattr(counter, "tracker"):
        # Determine device
        counter.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load YOLOv8 model
        counter.model = YOLO('yolov8n.pt')  # Ensure the model file is in the correct path
        counter.model.to(counter.device)

        # Initialize DeepSORT tracker
        counter.tracker = DeepSort(
            max_age=max_age,            # Maximum number of missed frames before a track is deleted
            n_init=3,                   # Number of consecutive detections before a track is confirmed
            nms_max_overlap=1.0,        # Non-max suppression threshold
            max_iou_distance=0.7,        # Maximum IOU distance for matching
            max_cosine_distance=0.4      # Maximum cosine distance for appearance matching
        )

        # Initialize global set to store unique track IDs
        counter.global_people_count = set()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    unique_people_in_chunk = set()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to control processing speed
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Resize frame for faster processing
        frame = cv2.resize(frame, resize_dim)

        # Perform detection using YOLOv8
        results = counter.model(frame, classes=[0], device=counter.device, verbose=False)  # class 0 is 'person'

        # Extract bounding boxes and confidence scores
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2) format
            confidences = results[0].boxes.conf.cpu().numpy()
        else:
            boxes = []
            confidences = []

        # Filter detections based on confidence threshold
        filtered_detections = [
            (box, confidence, 0)  # 0 is the class ID for 'person'
            for box, confidence in zip(boxes, confidences)
            if confidence >= min_confidence
        ]

        # Update tracker with current frame detections
        tracked_objects = counter.tracker.update_tracks(filtered_detections, frame=frame)

        # Update unique counts
        for track in tracked_objects:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            unique_people_in_chunk.add(track_id)
            counter.global_people_count.add(track_id)

        frame_count += 1

    cap.release()

    # Prepare the result
    result = {
        "person_tracker_count": len(unique_people_in_chunk),
        "global_tracker_count": len(counter.global_people_count)
    }

    return result