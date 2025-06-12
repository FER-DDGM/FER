import cv2
from ultralytics import YOLO
import os
import time


class EmotionDetectionApp:
    def __init__(self, model_path="yolo_emotion_model.pt"):
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

        # load model
        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found!")
            print("Please ensure you have the trained model file in the current directory.")
            print("Train the model using the Google Colab script first.")
            exit(1)

        print("Loading YOLO emotion model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")

    def detect_emotions(self, frame):
        """detect emotions in frame"""
        results = self.model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # get detection info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    if confidence > 0.25:  # Reduced threshold
                        # Get emotion and color
                        emotion = self.emotions[class_id]
                        color = self.colors[class_id]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw emotion label with confidence
                        label = f"{emotion}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        
                        # Background for text
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        
                        # Text
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Add debug info for low confidence detections
                        if confidence < 0.4:
                            debug_label = f"(Low Conf)"
                            cv2.putText(frame, debug_label, (x1, y2 + 15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame

    def run_camera(self):
        """run real-time emotion detection"""
        print("Starting camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot access camera")
            return

        print("Camera ready! Press 'q' to quit, 's' to save screenshot")

        # performance tracking
        fps_counter = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # flip for mirror effect
            frame = cv2.flip(frame, 1)

            # detect emotions
            frame = self.detect_emotions(frame)

            # calculate fps
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed
                start_time = time.time()
                print(f"FPS: {fps:.1f}")

            # add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # display frame
            cv2.imshow('YOLO Emotion Detection', frame)

            # handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"emotion_detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")

        cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped")

    def process_image(self, image_path):
        """process single image"""
        if not os.path.exists(image_path):
            print(f"Image '{image_path}' not found!")
            return

        print(f"Processing image: {image_path}")

        # load image
        frame = cv2.imread(image_path)
        if frame is None:
            print("Failed to load image")
            return

        # detect emotions
        result_frame = self.detect_emotions(frame)

        # save result
        output_path = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result_frame)
        print(f"Result saved: {output_path}")

        # display result
        cv2.imshow('Emotion Detection Result', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path):
        """process video file"""
        if not os.path.exists(video_path):
            print(f"Video '{video_path}' not found!")
            return

        print(f"Processing video: {video_path}")

        # open video
        cap = cv2.VideoCapture(video_path)

        # get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # setup video writer
        output_path = f"result_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # process frame
            result_frame = self.detect_emotions(frame)

            # write frame
            out.write(result_frame)

            # progress update
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

        cap.release()
        out.release()
        print(f"Video processing complete: {output_path}")


def main():
    print("YOLO Emotion Detection Application")
    print("=" * 40)

    # initialize app
    app = EmotionDetectionApp()

    while True:
        print("\nChoose an option:")
        print("1. Real-time camera detection")
        print("2. Process single image")
        print("3. Process video file")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            app.run_camera()

        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            app.process_image(image_path)

        elif choice == '3':
            video_path = input("Enter video path: ").strip()
            app.process_video(video_path)

        elif choice == '4':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
