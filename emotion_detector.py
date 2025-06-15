import os
import time
from collections import deque

import numpy as np
import cv2
from ultralytics import YOLO


class EmotionDetector:
    def __init__(self, model_path="best.pt"):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.colors = [
            (0, 0, 255),  # Red - Angry
            (0, 128, 0),  # Green - Disgust
            (128, 0, 128),  # Purple - Fear
            (0, 255, 255),  # Yellow - Happy
            (255, 0, 0),  # Blue - Sad
            (255, 165, 0),  # Orange - Surprise
            (128, 128, 128)  # Gray - Neutral
        ]

        # Face tracking
        self.face_tracks = {}
        self.next_face_id = 0
        self.max_disappeared = 30
        self.emotion_history_length = 10
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.4
        self.use_face_detection = True  # OpenCV face tracking

        # face size filtering
        self.min_face_size_ratio = 0.01  # Minimum size of face compared to image

        # Statistics
        self.detection_stats = {emotion: 0 for emotion in self.emotions}
        self.total_faces_detected = 0
        self.frame_count = 0
        self.session_start_time = time.time()

        # Performance monitoring
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=100)

        # Face detection for preprocessing
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Load YOLO model
        self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the YOLO model for emotion detection.

        Parameters:
        model_path (str): Path to the YOLO .pt model file.

        Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: If the model fails to load or test.
        """

        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found!")
            print("Please ensure you have the trained model file.")
            print("Available models in current directory:")
            for file in os.listdir('.'):
                if file.endswith('.pt'):
                    print(f"  - {file}")
            exit(1)

        try:
            print(f"Loading YOLO emotion model from {model_path}...")
            self.model = YOLO(model_path)
            print("Model loaded successfully!")

            # Test model
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            results = self.model(dummy_img, verbose=False)
            print("Model test successful!")

        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

    def preprocess_face(self, face_crop):
        """
        Apply resizing, contrast enhancement, and noise reduction to a cropped face image.

        Parameters:
        face_crop (np.ndarray): Cropped image of the face.

        Returns:
        np.ndarray: Preprocessed face image suitable for YOLO input.
        """

        # Resize to model input size
        face_resized = cv2.resize(face_crop, (224, 224))

        # Improve contrast and brightness
        lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return enhanced

    def is_face_size_valid(self, bbox, frame_shape):
        """
        Check if the bounding box of a detected face is large enough based on a minimum ratio.

        Parameters:
        bbox (tuple): Bounding box in format (x1, y1, x2, y2).
        frame_shape (tuple): Shape of the frame as (height, width, channels).

        Returns:
        bool: True if the face area meets the minimum ratio threshold, False otherwise.
        """

        x1, y1, x2, y2 = bbox
        face_width = x2 - x1
        face_height = y2 - y1

        # Check minimum area ratio
        face_area = face_width * face_height
        frame_area = frame_shape[0] * frame_shape[1]  # height * width
        area_ratio = face_area / frame_area

        return area_ratio >= self.min_face_size_ratio

    def detect_faces_opencv(self, frame):
        """
        Detect faces using OpenCV Haar cascades.

        Parameters:
        frame (np.ndarray): Image frame in BGR format.

        Returns:
        list: List of bounding boxes for detected faces.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def calculate_iou(self, box1, box2):
        """
        Compute the Intersection over Union (IoU) between two bounding boxes.

        Parameters:
        box1 (tuple): First bounding box (x1, y1, x2, y2).
        box2 (tuple): Second bounding box (x1, y1, x2, y2).

        Returns:
        float: IoU score between 0.0 and 1.0.
        """

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update_face_tracking(self, detections):
        """
        Update the internal state of face tracking by matching detections to existing tracks.

        Parameters:
        detections (list): List of face detection results containing bbox, emotion, and confidence.
        """

        # Mark all existing tracks as potentially disappeared
        for face_id in self.face_tracks:
            self.face_tracks[face_id]['disappeared'] += 1

        # Match detections to existing tracks
        matched_tracks = set()
        for detection in detections:
            bbox = detection['bbox']
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold

            for face_id, track in self.face_tracks.items():
                if track['disappeared'] > 0:  # Only consider active tracks
                    iou = self.calculate_iou(bbox, track['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = face_id

            if best_match_id is not None:
                # Update existing track
                self.face_tracks[best_match_id]['bbox'] = bbox
                self.face_tracks[best_match_id]['emotion_history'].append(detection['emotion'])
                self.face_tracks[best_match_id]['confidence_history'].append(detection['confidence'])
                self.face_tracks[best_match_id]['disappeared'] = 0
                matched_tracks.add(best_match_id)
                detection['track_id'] = best_match_id
            else:
                # Create new track
                new_id = self.next_face_id
                self.next_face_id += 1
                self.face_tracks[new_id] = {
                    'bbox': bbox,
                    'emotion_history': deque([detection['emotion']], maxlen=self.emotion_history_length),
                    'confidence_history': deque([detection['confidence']], maxlen=self.emotion_history_length),
                    'disappeared': 0,
                    'first_seen': time.time()
                }
                detection['track_id'] = new_id

        # Remove tracks that have disappeared for too long
        to_remove = []
        for face_id, track in self.face_tracks.items():
            if track['disappeared'] > self.max_disappeared:
                to_remove.append(face_id)

        for face_id in to_remove:
            del self.face_tracks[face_id]

    def get_smoothed_emotion(self, track_id):
        """
        Estimate the most likely emotion for a tracked face using weighted history.

        Parameters:
        track_id (int): ID of the tracked face.

        Returns:
        tuple: (smoothed_emotion, confidence_score) or (None, 0.0) if unavailable.
        """

        if track_id not in self.face_tracks:
            return None, 0.0

        track = self.face_tracks[track_id]
        emotion_history = list(track['emotion_history'])
        confidence_history = list(track['confidence_history'])

        if not emotion_history:
            return None, 0.0

        # Weighted voting based on confidence
        emotion_votes = {}
        total_weight = 0

        for emotion, confidence in zip(emotion_history, confidence_history):
            weight = confidence * (1.0 + 0.1 * (len(emotion_history) - emotion_history.index(emotion)))
            emotion_votes[emotion] = emotion_votes.get(emotion, 0) + weight
            total_weight += weight

        # Get most confident emotion
        best_emotion = max(emotion_votes.items(), key=lambda x: x[1])
        smoothed_confidence = best_emotion[1] / total_weight if total_weight > 0 else 0.0

        return best_emotion[0], smoothed_confidence

    def detect_emotions(self, frame, show_debug=False):
        """
        Run emotion detection on a frame using either OpenCV + YOLO or YOLO only.

        Parameters:
        frame (np.ndarray): Video frame in BGR format.
        show_debug (bool): Flag to show debugging overlays.

        Returns:
        tuple: (processed_frame, list of detected face dictionaries)
        """

        if self.use_face_detection:
            # Use OpenCV face detection first for better accuracy
            opencv_faces = self.detect_faces_opencv(frame)

            if len(opencv_faces) == 0:
                # Fallback to YOLO direct detection
                return self.detect_emotions_yolo(frame, show_debug)
            else:
                return self.detect_emotions_with_face_preprocessing(frame, opencv_faces, show_debug)
        else:
            return self.detect_emotions_yolo(frame, show_debug)

    def detect_emotions_yolo(self, frame, show_debug=False):
        """
        Run direct emotion detection using YOLO on the full image frame.

        Parameters:
        frame (np.ndarray): Frame to process.
        show_debug (bool): Whether to include debug overlays.

        Returns:
        tuple: (frame with overlays, list of face detections)
        """

        results = self.model(frame, conf=self.confidence_threshold, iou=self.nms_threshold, verbose=False)
        detected_faces = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    if confidence > self.confidence_threshold and self.is_face_size_valid((x1, y1, x2, y2),
                                                                                          frame.shape):
                        emotion = self.emotions[class_id]

                        face_info = {
                            'bbox': (x1, y1, x2, y2),
                            'emotion': emotion,
                            'confidence': confidence,
                            'method': 'yolo_direct'
                        }
                        detected_faces.append(face_info)

        # Update tracking
        self.update_face_tracking(detected_faces)

        # Draw detections
        frame = self.draw_enhanced_detections(frame, detected_faces, show_debug)

        # Update statistics
        self.update_statistics(detected_faces)

        return frame, detected_faces

    def detect_emotions_with_face_preprocessing(self, frame, opencv_faces, show_debug=False):
        """
        Use OpenCV to detect faces, preprocess them, and classify emotions using YOLO.

        Parameters:
        frame (np.ndarray): Original frame.
        opencv_faces (list): List of OpenCV-detected face bounding boxes.
        show_debug (bool): Whether to include debug overlays.

        Returns:
        tuple: (frame with overlays, list of face detections)
        """

        detected_faces = []

        for (x, y, w, h) in opencv_faces:
            # Check face size before processing
            opencv_bbox = (x, y, x + w, y + h)
            if not self.is_face_size_valid(opencv_bbox, frame.shape):
                continue

            # Add padding around face
            padding = 0.3
            pad_w = int(w * padding)
            pad_h = int(h * padding)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)

            # Extract and preprocess face
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Preprocess face
            enhanced_face = self.preprocess_face(face_crop)

            # Run YOLO on the preprocessed face
            face_results = self.model(enhanced_face, conf=self.confidence_threshold, verbose=False)

            best_confidence = 0
            best_emotion = None

            for result in face_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_emotion = self.emotions[class_id]

            if best_emotion is not None and best_confidence > self.confidence_threshold:
                face_info = {
                    'bbox': (x1, y1, x2, y2),
                    'emotion': best_emotion,
                    'confidence': best_confidence,
                    'method': 'opencv_yolo'
                }
                detected_faces.append(face_info)

        # Update tracking
        self.update_face_tracking(detected_faces)

        # Draw detections
        frame = self.draw_enhanced_detections(frame, detected_faces, show_debug)

        # Update statistics
        self.update_statistics(detected_faces)

        return frame, detected_faces

    def draw_enhanced_detections(self, frame, detected_faces, show_debug=False):
        """
        Draw face bounding boxes, emotions, and optional debug data on the frame.

        Parameters:
        frame (np.ndarray): Frame to draw on.
        detected_faces (list): List of face detection dicts.
        show_debug (bool): Show additional debug overlays if True.

        Returns:
        np.ndarray: Annotated video frame.
        """

        for face in detected_faces:
            x1, y1, x2, y2 = face['bbox']
            emotion = face['emotion']
            confidence = face['confidence']
            track_id = face.get('track_id', -1)

            # Get color for emotion
            emotion_idx = self.emotions.index(emotion)
            color = self.colors[emotion_idx]

            # Get smoothed emotion if tracking is available
            if track_id in self.face_tracks:
                smoothed_emotion, smoothed_confidence = self.get_smoothed_emotion(track_id)
                if smoothed_emotion:
                    emotion = smoothed_emotion
                    confidence = smoothed_confidence
                    emotion_idx = self.emotions.index(emotion)
                    color = self.colors[emotion_idx]

            # Draw enhanced bounding box
            thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw track ID
            if track_id >= 0:
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw emotion label with confidence
            label = f"{emotion}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            # Background for text
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0] + 10, y1), color, -1)

            # Text
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Debug information
            if show_debug:
                method = face.get('method', 'unknown')
                cv2.putText(frame, f"Method: {method}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Show emotion history if available
                if track_id in self.face_tracks:
                    history = list(self.face_tracks[track_id]['emotion_history'])
                    if len(history) > 1:
                        history_str = " -> ".join(history[-3:])  # Last 3 emotions
                        cv2.putText(frame, f"History: {history_str}", (x1, y2 + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def update_statistics(self, detected_faces):
        """
        Update global emotion statistics from the current frame detections.

        Parameters:
        detected_faces (list): List of detected face data.
        """

        for face in detected_faces:
            self.detection_stats[face['emotion']] += 1

        self.total_faces_detected += len(detected_faces)
        self.frame_count += 1

    def draw_statistics_overlay(self, frame, detected_faces, show_performance=True):
        """
        Render overlay containing detection statistics and performance metrics.

        Parameters:
        frame (np.ndarray): Frame to annotate.
        detected_faces (list): List of current frame detections.
        show_performance (bool): Include FPS and processing time if True.
        """

        # Calculate overlay size
        overlay_height = 180 if show_performance else 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        y_pos = 30

        # Current frame info
        cv2.putText(frame, f"Faces detected: {len(detected_faces)}",
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25

        # Active tracks
        active_tracks = len([t for t in self.face_tracks.values() if t['disappeared'] == 0])
        cv2.putText(frame, f"Active tracks: {active_tracks}",
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25

        # Current emotions
        if detected_faces:
            current_emotions = {}
            for face in detected_faces:
                emotion = face['emotion']
                current_emotions[emotion] = current_emotions.get(emotion, 0) + 1

            emotion_text = ", ".join([f"{e}:{c}" for e, c in current_emotions.items()])
            cv2.putText(frame, f"Current: {emotion_text}",
                        (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20

        # Performance info
        if show_performance and self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                        (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20

            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                cv2.putText(frame, f"Proc time: {avg_time * 1000:.1f}ms",
                            (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def run_camera(self, show_stats=True, show_debug=False):
        """
        Run real-time webcam-based emotion detection with interactive controls.

        Parameters:
        show_stats (bool): Enable statistics overlay.
        show_debug (bool): Enable debugging display.
        """

        print("Starting camera..")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot access camera")
            return

        # Camera setup
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Enhanced Emotion Detection Controls:")
        print("- 'q': Quit")
        print("- 's': Save screenshot")
        print("- 'd': Toggle debug mode")
        print("- 'f': Toggle face detection preprocessing")
        print("- 'r': Reset statistics and tracking")
        print("- 't': Toggle statistics display")
        print("- '+/-': Adjust confidence threshold")
        print("- 'p': Print detailed statistics")
        print("- 'z/x': Adjust minimum face size ratio")

        frame_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Process frame
            start_time = time.time()
            frame, detected_faces = self.detect_emotions(
                frame, show_debug
            )
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - frame_time)
            frame_time = current_time
            self.fps_history.append(fps)

            # Draw overlays
            if show_stats:
                self.draw_statistics_overlay(frame, detected_faces, True)

            # Add control instructions
            instructions = [
                f"Mode: {'OpenCV+YOLO' if self.use_face_detection else 'YOLO Direct'} | "
                f"Conf: {self.confidence_threshold:.2f} | Debug: {'ON' if show_debug else 'OFF'}",
                "Controls: q=quit, s=save, d=debug, f=face_mode, r=reset, t=stats, +/-=threshold"
            ]

            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, frame.shape[0] - 40 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow('Enhanced Emotion Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = os.path.join("results", f"emotion_detection_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
            elif key == ord('f'):
                self.use_face_detection = not self.use_face_detection
                print(f"Face detection mode: {'OpenCV+YOLO' if self.use_face_detection else 'YOLO Direct'}")
            elif key == ord('t'):
                show_stats = not show_stats
                print(f"Statistics display: {'ON' if show_stats else 'OFF'}")
            elif key == ord('r'):
                self.reset_all_data()
                print("All data reset")
            elif key == ord('p'):
                self.print_detailed_statistics()
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = max(0.9, self.confidence_threshold + 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.confidence_threshold = min(0.1, self.confidence_threshold - 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            elif key == ord('z'):
                self.min_face_size_ratio = min(0.001, self.min_face_size_ratio - 0.001)
                print(f"Min face to image ratio: {self.confidence_threshold:.2f}")
            elif key == ord('z'):
                self.min_face_size_ratio = max(1.0, self.min_face_size_ratio + 0.001)
                print(f"Min face to image ratio: {self.confidence_threshold:.2f}")

        cap.release()
        cv2.destroyAllWindows()
        self.print_session_summary()

    def reset_all_data(self):
        """
        Clear all tracking, history, and statistics data.
        """

        self.face_tracks.clear()
        self.next_face_id = 0
        self.detection_stats = {emotion: 0 for emotion in self.emotions}
        self.total_faces_detected = 0
        self.frame_count = 0
        self.session_start_time = time.time()
        self.fps_history.clear()
        self.processing_times.clear()

    def print_detailed_statistics(self):
        """
        Print an in-depth summary of emotion detections, tracks, and performance data.
        """

        print("\n" + "=" * 60)
        print("DETAILED EMOTION DETECTION STATISTICS")
        print("=" * 60)

        session_time = time.time() - self.session_start_time
        print(f"Session duration: {session_time:.1f} seconds")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total faces detected: {self.total_faces_detected}")
        print(f"Active face tracks: {len([t for t in self.face_tracks.values() if t['disappeared'] == 0])}")
        print(f"Total face tracks created: {self.next_face_id}")

        if self.frame_count > 0:
            print(f"Average faces per frame: {self.total_faces_detected / self.frame_count:.2f}")
            print(f"Average FPS: {self.frame_count / session_time:.1f}")

        if self.processing_times:
            avg_proc_time = sum(self.processing_times) / len(self.processing_times)
            print(f"Average processing time: {avg_proc_time * 1000:.1f}ms")

        print("\nEmotion Distribution:")
        total_detections = sum(self.detection_stats.values())
        if total_detections > 0:
            for emotion, count in sorted(self.detection_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100
                bar = "█" * int(percentage / 2)
                print(f"  {emotion:>8}: {count:>5} ({percentage:>5.1f}%) {bar}")

        print("\nActive Face Tracks:")
        for face_id, track in self.face_tracks.items():
            if track['disappeared'] == 0:
                duration = time.time() - track['first_seen']
                recent_emotions = list(track['emotion_history'])[-3:]
                print(f"  ID {face_id}: {duration:.1f}s, Recent: {' -> '.join(recent_emotions)}")

    def print_session_summary(self):
        """
        Print a summary of the detection session including top emotions and FPS.
        """

        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)

        session_time = time.time() - self.session_start_time
        print(f"Total session time: {session_time:.1f} seconds")
        print(f"Frames processed: {self.frame_count}")
        print(f"Total face detections: {self.total_faces_detected}")
        print(f"Unique faces tracked: {self.next_face_id}")

        if self.frame_count > 0:
            print(f"Average processing rate: {self.frame_count / session_time:.1f} FPS")

        # Top emotions
        if sum(self.detection_stats.values()) > 0:
            top_emotions = sorted(self.detection_stats.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Most detected emotions:")
            for i, (emotion, count) in enumerate(top_emotions, 1):
                print(f"  {i}. {emotion}: {count} times")

    def process_single_image(self, image_path):
        """
        Process a single static image for emotion detection and display/save results.

        Parameters:
        image_path (str): Path to the image file.

        Returns:
        list: List of detected faces with emotions.
        """

        print(f"Processing image: {image_path}")

        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print("Failed to load image")
            return

        # Process image
        result_frame, detected_faces = self.detect_emotions(frame, show_debug=True)

        # Print results
        print(f"Detected {len(detected_faces)} faces:")
        for i, face in enumerate(detected_faces, 1):
            print(f"  Face {i}: {face['emotion']} (confidence: {face['confidence']:.3f})")

        # Save result inside the results folder
        output_path = os.path.join("results", f"result_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, result_frame)
        print(f"Result saved: {output_path}")

        # Display result
        cv2.imshow('Emotion Detection Result', result_frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return detected_faces

    def process_video_file(self, video_path):
        """
        Process a full video file frame-by-frame for emotion detection.

        Parameters:
        video_path (str): Path to input video file.
        """

        print(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Setup output
        output_path = os.path.join("results", f"result_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset detection data
        self.reset_all_data()

        frame_count = 0
        start_time = time.time()

        print("Processing frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result_frame, detected_faces = self.detect_emotions(frame)

            # Write frame
            out.write(result_frame)

            frame_count += 1
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_count) * (total_frames - frame_count)
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - ETA: {eta:.1f}s")

        # Cleanup
        cap.release()
        out.release()

        print(f"Video processing complete: {output_path}")
        self.print_detailed_statistics()

    def print_model_info(self):
        """
        Display YOLO model configuration and test inference performance.
        """

        print("\n" + "=" * 50)
        print("MODEL INFORMATION")
        print("=" * 50)

        try:
            # Get model info
            model_info = self.model.info()
            print(f"Model: {self.model.model}")
            print(f"Input size: {self.img_size}x{self.img_size}")
            print(f"Classes: {len(self.emotions)}")
            print(f"Emotions: {', '.join(self.emotions)}")
            print(f"Current confidence threshold: {self.confidence_threshold}")
            print(f"NMS threshold: {self.nms_threshold}")

            # Test inference speed
            print("\nTesting inference speed...")
            dummy_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            times = []
            for _ in range(10):
                start = time.time()
                _ = self.model(dummy_img, verbose=False)
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            print(f"Average inference time: {avg_time * 1000:.1f}ms")
            print(f"Theoretical max FPS: {1 / avg_time:.1f}")

        except Exception as e:
            print(f"Error getting model info: {e}")


def main():
    """
    Command-line interface for emotion detection tool with options for
    webcam, single image, and video processing.
    """

    print("Enhanced Multi-Face Emotion Detection")
    print("=" * 50)

    # Initialize detector
    model_path = input("Model path (default: best.pt): ").strip() or "best.pt"
    detector = EmotionDetector(model_path)

    while True:
        print("\nChoose detection mode:")
        print("1. Real-time camera (recommended)")
        print("2. Process single image")
        print("3. Process video file")
        print("4. View model info")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            print("\nCamera Settings:")
            confidence = float(input("Confidence threshold (0.1-0.9, default 0.5): ") or "0.5")
            use_face_detection = input("Use OpenCV face preprocessing? (y/n, default y): ").lower() != 'n'
            show_debug = input("Show debug info? (y/n, default n): ").lower() == 'y'

            detector.run_camera(
                show_stats=True,
                show_debug=show_debug
            )

        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                detector.process_single_image(image_path)
            else:
                print("Image not found!")

        elif choice == '3':
            video_path = input("Enter video path: ").strip()
            if os.path.exists(video_path):
                detector.process_video_file(video_path)
            else:
                print("Video not found!")

        elif choice == '4':
            detector.print_model_info()

        elif choice == '5':
            print("Goodbye!")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
