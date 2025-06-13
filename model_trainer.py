import os
import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
import random
import albumentations as A
import requests
import zipfile
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

class ImprovedEmotionTrainer:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.data_dir = "improved_emotion_data"
        
        self.ferplus_path = ""

        self.img_size = 224
        self.face_padding = 0.3
        
        self.samples_per_emotion = {
            'train': 2000,
            'val': 300,
            'test': 200
        }
        
        self.train_augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
        ])
        
        self.val_augment = A.Compose([
            A.CLAHE(clip_limit=2.0, p=0.3),
        ])
        
        # Face detection for preprocessing
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # FER+ emotion folder mapping
        self.ferplus_emotion_mapping = {
            'angry': 0,      # angry
            'disgust': 1,    # disgust  
            'fear': 2,       # fear
            'happy': 3,      # happy
            'sad': 4,        # sad
            'surprise': 5,   # surprise
            'neutral': 6     # neutral
        }
    
    def download_ferplus_dataset(self):        
        import kagglehub

        self.ferplus_path = kagglehub.dataset_download("msambare/fer2013")

        print("Path to dataset files:", self.ferplus_path)

        if os.path.exists(self.ferplus_path):
            required_splits = ['train', 'test']
            required_emotions = list(self.ferplus_emotion_mapping.keys())
            
            for split in required_splits:
                split_path = os.path.join(self.ferplus_path, split)
                if not os.path.exists(split_path):
                    print(f"Missing {split} folder!")
                    return False
                
                for emotion in required_emotions:
                    emotion_path = os.path.join(split_path, emotion)
                    if not os.path.exists(emotion_path):
                        print(f"Missing {emotion} folder in {split}!")
                        return False
            
            print("FER+ dataset structure verified!")
            return True
        else:
            print(f"FER+ dataset not found at {self.ferplus_path}")
            return False
    
    def load_ferplus_images(self, split):
        """Load images from FER+ folder structure"""
        images_data = []
        
        split_path = os.path.join(self.ferplus_path, split)
        if not os.path.exists(split_path):
            print(f"Split {split} not found in {self.ferplus_path}")
            return images_data
        
        for emotion_name, emotion_id in self.ferplus_emotion_mapping.items():
            emotion_path = os.path.join(split_path, emotion_name)
            if not os.path.exists(emotion_path):
                print(f"Emotion folder {emotion_name} not found in {split}")
                continue
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(emotion_path).glob(f'*{ext}'))
                image_files.extend(Path(emotion_path).glob(f'*{ext.upper()}'))
            
            print(f"Found {len(image_files)} images for {emotion_name} in {split}")
            
            for img_path in image_files:
                images_data.append({
                    'path': str(img_path),
                    'emotion': emotion_id,
                    'emotion_name': emotion_name,
                    'split': split
                })
        
        return images_data
    
    def detect_and_crop_faces(self, image, padding=0.3):
        """Detect faces and crop with padding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        cropped_faces = []
        for (x, y, w, h) in faces:
            # Add padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + w + pad_w)
            y2 = min(image.shape[0], y + h + pad_h)
            
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size > 0:
                cropped_faces.append(face_crop)
        
        return cropped_faces
    
    def prepare_ferplus_dataset(self):
        """Prepare FER+ dataset with folder structure"""
        print("Preparing FER+ dataset...")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            os.makedirs(f"{self.data_dir}/{split}/images", exist_ok=True)
            os.makedirs(f"{self.data_dir}/{split}/labels", exist_ok=True)
        
        # Load train and test data from FER+
        print("Loading FER+ train data...")
        train_data = self.load_ferplus_images('train')
        
        print("Loading FER+ test data...")
        test_data = self.load_ferplus_images('test')
        
        if not train_data and not test_data:
            print("No data loaded from FER+ dataset!")
            return False
        
        # Split train data into train/val
        train_images, val_images = train_test_split(
            train_data, 
            test_size=0.2, 
            random_state=42, 
            stratify=[item['emotion'] for item in train_data]
        )
        
        # Combine all data
        all_data = {
            'train': train_images,
            'val': val_images, 
            'test': test_data
        }
        
        # Process each split
        processed_count = 0
        emotion_counts = {split: Counter() for split in ['train', 'val', 'test']}
        
        for split, split_data in all_data.items():
            print(f"\nProcessing {split} split...")
            
            # Shuffle data
            random.shuffle(split_data)
            
            for item in split_data:
                emotion_id = item['emotion']
                
                # Check if we have enough samples for this emotion in this split
                max_samples = self.samples_per_emotion[split]
                if emotion_counts[split][emotion_id] >= max_samples:
                    continue
                
                # Load image
                try:
                    image = cv2.imread(item['path'])
                    if image is None:
                        print(f"Failed to load image: {item['path']}")
                        continue
                    
                    # Try face detection first
                    faces = self.detect_and_crop_faces(image, self.face_padding)
                    
                    if faces:
                        # Use the largest detected face
                        face = max(faces, key=lambda x: x.shape[0] * x.shape[1])
                    else:
                        # If no face detected, use the center crop
                        h, w = image.shape[:2]
                        size = min(h, w)
                        start_h = (h - size) // 2
                        start_w = (w - size) // 2
                        face = image[start_h:start_h+size, start_w:start_w+size]
                    
                    # Create multiple variations for training
                    variations = 3 if split == 'train' else 1
                    
                    for var in range(variations):
                        # Resize to target size
                        resized_face = cv2.resize(face, (self.img_size, self.img_size))
                        
                        # Apply augmentation
                        if split == 'train' and var > 0:
                            augmented = self.train_augment(image=resized_face)
                            resized_face = augmented['image']
                        elif split == 'val':
                            augmented = self.val_augment(image=resized_face)
                            resized_face = augmented['image']
                        
                        # Save image
                        img_name = f"img_{processed_count:06d}_e{emotion_id}_v{var}.jpg"
                        img_path = f"{self.data_dir}/{split}/images/{img_name}"
                        cv2.imwrite(img_path, resized_face)
                        
                        # Create YOLO label
                        label_name = f"img_{processed_count:06d}_e{emotion_id}_v{var}.txt"
                        label_path = f"{self.data_dir}/{split}/labels/{label_name}"
                        
                        # Face image with padding
                        center_x, center_y = 0.5, 0.5
                        width, height = 0.9, 0.9  # padding
                        
                        with open(label_path, 'w') as f:
                            f.write(f"{emotion_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                        
                        emotion_counts[split][emotion_id] += 1
                        
                        # Stop if we have enough samples
                        if emotion_counts[split][emotion_id] >= max_samples:
                            break
                
                except Exception as e:
                    print(f"Error processing {item['path']}: {e}")
                    continue
                
                processed_count += 1
                
                if processed_count % 500 == 0:
                    print(f"Processed {processed_count} samples")
        
        print("Dataset preparation complete!")
        
        # Print final statistics
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()} set distribution:")
            total = sum(emotion_counts[split].values())
            for emotion_id in range(7):
                count = emotion_counts[split][emotion_id]
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  {self.emotions[emotion_id]}: {count} ({percentage:.1f}%)")
        
        return True
    
    def create_config(self):
        """Create YOLO configuration file"""
        config = {
            'path': os.path.abspath(self.data_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 7,
            'names': self.emotions
        }
        
        config_path = f"{self.data_dir}/config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"Config file created: {config_path}")
        return config_path
    
    def train_model(self, model_size='s', epochs=100, batch_size=16, resume=False):
        """Train improved YOLO model"""
        print(f"Training YOLOv8{model_size} model...")
        
        # Create config
        config_path = self.create_config()
        
        # Initialize model with appropriate size
        model_weights = f'yolov8{model_size}.pt'
        model = YOLO(model_weights)
        
        # Enhanced training parameters
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=self.img_size,
            batch=batch_size,
            device='0' if self.is_gpu_available() else 'cpu',
            name=f'emotion_yolo_v8{model_size}_improved',
            project='runs/detect',
            resume=resume,
            
            # Optimizer
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.1,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Regularization
            dropout=0.0,
            label_smoothing=0.1,
            
            # Data augmentation (Splush splash those images to hell)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.8,
            mixup=0.1,
            copy_paste=0.0,
            
            # Training settings
            patience=20,
            save=True,
            save_period=10,
            plots=True,
            verbose=True,
            
            # Validation settings
            val=True,
            split=0.2,
            
            # Other settings
            exist_ok=True,
            pretrained=True,
            seed=42,
        )
        
        best_model_path = f'runs/detect/emotion_yolo_v8{model_size}_improved/weights/best.pt'
        return best_model_path, results
    
    @staticmethod
    def is_gpu_available():
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def validate_model(self, model_path):
        """Validate the trained model"""
        print("Validating model...")
        
        model = YOLO(model_path)
        
        # Run validation
        config_path = f"{self.data_dir}/config.yaml"
        results = model.val(
            data=config_path,
            imgsz=self.img_size,
            batch=8,
            verbose=True,
            plots=True,
            save_json=True
        )
        
        return results
    
    def export_model(self, model_path, formats=['onnx', 'torchscript']):
        """Export model to different formats"""
        print("Exporting model...")
        
        model = YOLO(model_path)
        
        for format_type in formats:
            try:
                model.export(format=format_type, imgsz=self.img_size)
                print(f"Model exported to {format_type}")
            except Exception as e:
                print(f"Failed to export to {format_type}: {e}")

def main():
    print("Improved YOLO Emotion Detection Training (FER+ Dataset)")
    print("=" * 60)
    
    trainer = ImprovedEmotionTrainer()
    
    # Check dataset availability
    print("Checking FER+ dataset...")
    if not trainer.download_ferplus_dataset():
        print("Please set up the FER+ dataset according to the instructions above.")
        return
    
    # Prepare dataset
    print("\nPreparing dataset for YOLO training...")
    if not trainer.prepare_ferplus_dataset():
        print("Failed to prepare dataset.")
        return
    
    # Step 3: Configure training
    print("\nTraining configuration...")
    model_size = input("Model size (n/s/m/l/x, default 's'): ").strip() or 's'
    epochs = int(input("Number of epochs (default 100): ") or "100")
    batch_size = int(input("Batch size (default 16): ") or "16")
    
    # Step 4: Train model
    print(f"\nTraining YOLOv8{model_size} model...")
    best_model_path, results = trainer.train_model(
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size
    )
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: {best_model_path}")
    
    # Step 5: Validate model
    print("\nValidating model...")
    validation_results = trainer.validate_model(best_model_path)
    
    # Step 6: Export model
    export_choice = input("\nExport model to ONNX/TorchScript? (y/n, default n): ").strip().lower()
    if export_choice == 'y':
        trainer.export_model(best_model_path)
    
    print("\nTraining pipeline complete!")
    print(f"Copy {best_model_path} to your main directory and rename it to 'best.pt'")

if __name__ == "__main__":
    main()