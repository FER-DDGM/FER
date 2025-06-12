import cv2
from ultralytics import YOLO
import os
import time
import numpy as np


class MultiEmotionDetectionApp:
    def __init__(self, model_path="best.pt"):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.colors = [
            (0, 0, 255),  # red - angry
            (0, 128, 0),  # green - disgust
            (128, 0, 128),  # purple - fear
            (0, 255, 255),  # yellow - happy
            (255, 0, 0),  # blue - sad
            (255, 165, 0),  # orange - surprise
            (128, 128, 128)  # gray - neutral
        ]

        # Statistics tracking
        self.detection_stats = {emotion: 0 for emotion in self.emotions}
        self.total_faces_detected = 0
        self.frame_count = 0

        # Load model
        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found!")
            print("Please ensure you have the trained model file in the current directory.")
            print("Train the model using the Google Colab script first.")
            exit(1)

        print("Loading YOLO emotion model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")

    def detect_emotions(self, frame, confidence_threshold=0.25, show_debug=False):
        """Detect emotions using Haar + YOLO with expanded face box"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_faces = []
        face_count = 0
        h_img, w_img = frame.shape[:2]

        for (x, y, w, h) in faces:
            face_count += 1

            # Expand bounding box by 20%
            padding = 0.3
            x_pad = int(w * padding)
            y_pad = int(h * padding)

            x1 = max(0, x - x_pad)
            y1 = max(0, y - y_pad)
            x2 = min(w_img, x + w + x_pad)
            y2 = min(h_img, y + h + y_pad)

            face_crop = frame[y1:y2, x1:x2]
            resized_face = cv2.resize(face_crop, (128, 128))

            # Run emotion detection on resized face
            results = self.model(resized_face, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        if confidence > confidence_threshold:
                            emotion = self.emotions[class_id]
                            color = self.colors[class_id]

                            # Update stats
                            self.detection_stats[emotion] += 1

                            # Store face info
                            face_info = {
                                'bbox': (x1, y1, x2, y2),
                                'emotion': emotion,
                                'confidence': confidence,
                                'face_id': face_count
                            }
                            detected_faces.append(face_info)

                            # Draw bounding box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            face_id_label = f"#{face_count}"
                            cv2.putText(frame, face_id_label, (x1, y1 - 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            label = f"{emotion}: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                          (x1 + label_size[0], y1), color, -1)
                            cv2.putText(frame, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                            # Optional debug info
                            if show_debug:
                                area_label = f"Area: {(x2 - x1) * (y2 - y1)}"
                                cv2.putText(frame, area_label, (x1, y2 + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                if confidence < 0.4:
                                    cv2.putText(frame, "(Low Conf)", (x1, y2 + 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        self.total_faces_detected += face_count
        self.frame_count += 1

        return frame, detected_faces

    def draw_statistics_overlay(self, frame, detected_faces):
        """Draw statistics overlay on frame"""
        # Background for statistics
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Current frame statistics
        cv2.putText(frame, f"Faces detected: {len(detected_faces)}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show detected emotions in current frame
        if detected_faces:
            current_emotions = [face['emotion'] for face in detected_faces]
            emotion_summary = {}
            for emotion in current_emotions:
                emotion_summary[emotion] = emotion_summary.get(emotion, 0) + 1

            y_pos = 55
            for emotion, count in emotion_summary.items():
                text = f"{emotion}: {count}"
                cv2.putText(frame, text, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20

    def run_camera(self, show_stats=True, show_debug=False, confidence_threshold=0.25):
        """Run real-time emotion detection with enhanced multi-face support"""
        print("Starting camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot access camera")
            return

        print("Camera ready! Controls:")
        print("- 'q': Quit")
        print("- 's': Save screenshot")
        print("- 'd': Toggle debug mode")
        print("- 'r': Reset statistics")
        print("- '+': Increase confidence threshold")
        print("- '-': Decrease confidence threshold")

        # Performance tracking
        fps_counter = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect emotions
            frame, detected_faces = self.detect_emotions(frame, confidence_threshold, show_debug)

            # Draw statistics overlay if enabled
            if show_stats:
                self.draw_statistics_overlay(frame, detected_faces)

            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed
                start_time = time.time()
                print(f"FPS: {fps:.1f} | Faces in frame: {len(detected_faces)} | Threshold: {confidence_threshold:.2f}")

            # Add instructions
            instructions = [
                "Controls: 'q'=quit, 's'=save, 'd'=debug, 'r'=reset, '+/-'=threshold",
                f"Confidence: {confidence_threshold:.2f} | Debug: {'ON' if show_debug else 'OFF'}"
            ]

            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, frame.shape[0] - 40 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow('Multi-Face Emotion Detection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"multi_emotion_detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename} (Faces: {len(detected_faces)})")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
            elif key == ord('r'):
                self.reset_statistics()
                print("Statistics reset")
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.9, confidence_threshold + 0.05)
                print(f"Confidence threshold: {confidence_threshold:.2f}")
            elif key == ord('-'):
                confidence_threshold = max(0.1, confidence_threshold - 0.05)
                print(f"Confidence threshold: {confidence_threshold:.2f}")

        cap.release()
        cv2.destroyAllWindows()
        self.print_session_statistics()
        print("Camera stopped")

    def process_image(self, image_path, confidence_threshold=0.25, show_debug=False):
        """Process single image with multi-face detection"""
        if not os.path.exists(image_path):
            print(f"Image '{image_path}' not found!")
            return

        print(f"Processing image: {image_path}")

        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print("Failed to load image")
            return

        # Detect emotions
        result_frame, detected_faces = self.detect_emotions(frame, confidence_threshold, show_debug)

        # Draw statistics overlay
        self.draw_statistics_overlay(result_frame, detected_faces)

        # Print detection results
        print(f"Detected {len(detected_faces)} faces:")
        for i, face in enumerate(detected_faces, 1):
            print(f"  Face {i}: {face['emotion']} (confidence: {face['confidence']:.2f})")

        # Save result
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_multi_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, frame)
        print(f"Result saved: {output_path}")

        # Display result
        cv2.imshow('Multi-Face Emotion Detection Result', result_frame)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return detected_faces

    def process_video(self, video_path, confidence_threshold=0.25, show_debug=False):
        """Process video file with multi-face detection"""
        if not os.path.exists(video_path):
            print(f"Video '{video_path}' not found!")
            return

        print(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Setup video writer in 'results' folder
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_multi_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_faces_in_video = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result_frame, detected_faces = self.detect_emotions(frame, confidence_threshold, show_debug)

            # Add frame statistics
            total_faces_in_video += len(detected_faces)

            # Draw statistics overlay
            self.draw_statistics_overlay(result_frame, detected_faces)

            # Write frame
            out.write(result_frame)

            # Progress update
            frame_count += 1
            if frame_count % 100 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                avg_faces_per_frame = total_faces_in_video / frame_count
                print(
                    f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Avg faces/frame: {avg_faces_per_frame:.1f}")

        cap.release()
        out.release()

        print(f"Video processing complete: {output_path}")
        print(f"Total faces detected in video: {total_faces_in_video}")
        print(f"Average faces per frame: {total_faces_in_video / frame_count:.2f}")

    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_stats = {emotion: 0 for emotion in self.emotions}
        self.total_faces_detected = 0
        self.frame_count = 0

    def print_session_statistics(self):
        """Print session statistics"""
        print("\n" + "=" * 50)
        print("SESSION STATISTICS")
        print("=" * 50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total faces detected: {self.total_faces_detected}")
        if self.frame_count > 0:
            print(f"Average faces per frame: {self.total_faces_detected / self.frame_count:.2f}")

        print("\nEmotion distribution:")
        total_detections = sum(self.detection_stats.values())
        if total_detections > 0:
            for emotion, count in self.detection_stats.items():
                percentage = (count / total_detections) * 100
                print(f"  {emotion}: {count} ({percentage:.1f}%)")
        else:
            print("  No emotions detected")


def main():
    print("Multi-Face YOLO Emotion Detection Application")
    print("=" * 50)

    # Initialize app
    app = MultiEmotionDetectionApp()

    while True:
        print("\nChoose an option:")
        print("1. Real-time camera detection")
        print("2. Process single image")
        print("3. Process video file")
        print("4. View current statistics")
        print("5. Reset statistics")
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == '1':
            print("\nCamera Detection Settings:")
            confidence = float(input("Confidence threshold (0.1-0.9, default 0.25): ") or "0.25")
            confidence = max(0.1, min(0.9, confidence))

            show_stats = input("Show statistics overlay? (y/n, default y): ").lower() != 'n'
            show_debug = input("Show debug info? (y/n, default n): ").lower() == 'y'

            app.run_camera(show_stats=show_stats, show_debug=show_debug,
                           confidence_threshold=confidence)

        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            confidence = float(input("Confidence threshold (0.1-0.9, default 0.25): ") or "0.25")
            confidence = max(0.1, min(0.9, confidence))
            show_debug = input("Show debug info? (y/n, default n): ").lower() == 'y'

            app.process_image(image_path, confidence_threshold=confidence,
                              show_debug=show_debug)

        elif choice == '3':
            video_path = input("Enter video path: ").strip()
            confidence = float(input("Confidence threshold (0.1-0.9, default 0.25): ") or "0.25")
            confidence = max(0.1, min(0.9, confidence))
            show_debug = input("Show debug info? (y/n, default n): ").lower() == 'y'

            app.process_video(video_path, confidence_threshold=confidence,
                              show_debug=show_debug)

        elif choice == '4':
            app.print_session_statistics()

        elif choice == '5':
            app.reset_statistics()
            print("Statistics reset successfully!")

        elif choice == '6':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
