# Documentation: ImprovedEmotionDetector Class

## Overview

The `ImprovedEmotionDetector` class enables real-time emotion recognition from video or image inputs using a pretrained
YOLO model. It incorporates features like face tracking, preprocessing, confidence thresholding, performance monitoring,
and statistics reporting.

---

## Class Initialization

### `__init__(model_path="best.pt")`

Initializes the detector with:

- A list of emotions and their corresponding display colors.
- Internal tracking structures to maintain per-face ID states.
- Parameters like confidence thresholds, minimum face size ratios, etc.
- OpenCV Haar cascades for optional preprocessing.
- Loads a YOLO model via `load_model()`.

Use this constructor to initialize an emotion detector with a YOLO model file path.

---

## Core Methods

### `load_model(model_path)`

Loads a YOLO model from the specified path. Verifies loading by running a dummy image through the model. If the model
file is missing or invalid, the program exits gracefully with debug information.

---

### `preprocess_face(face_crop)`

Prepares a cropped face image for model inference:

- Resizes to 224×224 pixels.
- Enhances contrast using CLAHE.
- Applies bilateral filtering to reduce noise. Returns the cleaned image for improved classification accuracy.

---

### `is_face_size_valid(bbox, frame_shape)`

Checks if the detected face’s size is significant relative to the full frame using a configurable ratio. Helps filter
out noise or far-away faces.

---

### `detect_faces_opencv(frame)`

Uses OpenCV’s Haar cascade classifier to detect faces in grayscale frames. This acts as a preprocessing filter before
running the more compute-intensive YOLO model.

---

### `calculate_iou(box1, box2)`

Computes the Intersection over Union (IoU) for two bounding boxes. Used for associating face detections with existing
tracked IDs.

---

### `update_face_tracking(detections)`

Maintains persistent face tracking:

- Updates tracks using IoU scores.
- Assigns new IDs to unmatched detections.
- Deletes stale tracks. Also records emotion history and confidence values.

---

### `get_smoothed_emotion(track_id)`

Returns the most likely emotion for a face based on a weighted history of past predictions. Accounts for confidence
scores and position in history.

---

### `detect_emotions(frame, show_debug=False)`

Master method that calls either YOLO-only detection or a hybrid OpenCV+YOLO method based on user settings.

---

### `detect_emotions_yolo(frame, show_debug=False)`

Runs the YOLO model directly on the full frame. Filters results using confidence threshold and face size. Outputs a list
of detections.

---

### `detect_emotions_with_face_preprocessing(frame, opencv_faces, show_debug=False)`

For each OpenCV-detected face, it preprocesses and runs YOLO only on the face region. Offers higher precision at the
cost of speed.

---

### `draw_enhanced_detections(frame, detected_faces, show_debug=False)`

Draws bounding boxes, labels, smoothed emotions, track IDs, and optionally debugging info on the frame.

---

### `update_statistics(detected_faces)`

Maintains running totals of detected emotions and total face counts. Called after each processed frame.

---

### `draw_statistics_overlay(frame, detected_faces, show_performance=True)`

Draws a transparent overlay with:

- Number of faces
- Active tracks
- Current detected emotions
- FPS and processing time

---

### `run_camera(show_stats=True, show_debug=False)`

Launches a live webcam interface:

- Applies detection frame-by-frame
- Supports keyboard input for toggling modes
- Displays real-time annotations

---

### `reset_all_data()`

Clears all tracking, statistics, and histories. Useful for resetting state mid-session.

---

### `print_detailed_statistics()`

Prints a full report:

- Duration, frames, faces detected
- Per-emotion counts
- Track-level summaries
- Recent emotion histories

---

### `print_session_summary()`

Shorter version of statistics: session duration, FPS, top 3 detected emotions.

---

### `process_single_image(image_path)`

Processes one image:

- Detects and annotates emotions
- Saves and shows the result

---

### `process_video_file(video_path)`

Processes a video:

- Annotates each frame
- Writes output to disk
- Reports progress and statistics

---

### `print_model_info()`

Prints YOLO model info including:

- Loaded model name
- Number of classes
- Input size
- Benchmark inference speed

---

## Usage Example

```python
from detector import ImprovedEmotionDetector

detector = ImprovedEmotionDetector("best.pt")
detector.run_camera()
```

---

## Notes

- Use OpenCV face detection for better results on frontal faces.
- Adjust `confidence_threshold` and `min_face_size_ratio` depending on camera setup.
