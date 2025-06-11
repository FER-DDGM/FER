import os
import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
import argparse

class YOLOTrainerLocal:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.data_dir = "./emotion_data"

    def setup_environment(self):
        """Setup local environment"""
        print("Setting up local environment...")
        os.makedirs(self.data_dir, exist_ok=True)
        print("Environment ready!")

    def prepare_dataset(self, fer_csv_path):
        """Convert fer2013 to YOLO format"""
        print("Converting dataset to YOLO format...")

        for split in ['train', 'val', 'test']:
            os.makedirs(f"{self.data_dir}/{split}/images", exist_ok=True)
            os.makedirs(f"{self.data_dir}/{split}/labels", exist_ok=True)

        df = pd.read_csv(fer_csv_path)
        print(f"Total samples: {len(df)}")

        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(df)} samples")

            # if idx > 1000:
            #     break  # Limit samples for stability during testing

            emotion = int(row['emotion'])
            pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
            usage = row['Usage'].lower()

            img = cv2.resize(pixels, (128, 128))  # Use smaller image size for memory efficiency
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            split = 'train' if usage == 'training' else ('val' if usage == 'publictest' else 'test')

            img_path = f"{self.data_dir}/{split}/images/img_{idx:05d}.jpg"
            cv2.imwrite(img_path, img)

            label_path = f"{self.data_dir}/{split}/labels/img_{idx:05d}.txt"
            with open(label_path, 'w') as f:
                f.write(f"{emotion} 0.5 0.5 0.8 0.8\n")

        print("Dataset conversion complete!")

        for split in ['train', 'val', 'test']:
            count = len(os.listdir(f"{self.data_dir}/{split}/images"))
            print(f"{split}: {count} samples")

    def create_config(self):
        config = {
            'path': self.data_dir,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 7,
            'names': self.emotions
        }

        config_path = f"{self.data_dir}/config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        print(f"Config saved: {config_path}")
        return config_path

    def train_model(self, epochs=10, batch_size=8):
        print("Starting training...")

        import torch
        print(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        config_path = self.create_config()
        model = YOLO('yolov8n.pt')

        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=128,
            batch=batch_size,
            device=0 if torch.cuda.is_available() else 'cpu',
            name='emotion_yolo',
            project='./runs/detect',
            save=True,
            plots=True,
            verbose=True
        )

        best_model_path = './runs/detect/emotion_yolo/weights/best.pt'
        print(f"Training complete! Best model: {best_model_path}")

        return best_model_path, results

# main training function

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help="Path to fer2013.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    print("YOLO Emotion Detection Training (Local)")
    print("=" * 50)

    trainer = YOLOTrainerLocal()
    trainer.setup_environment()

    print("\nStep 1: Prepare Dataset")
    trainer.prepare_dataset(args.csv)

    print("\nStep 2: Train Model")
    best_model, results = trainer.train_model()

    print("\nTraining Pipeline Complete!")
    print(f"Best model saved to: {best_model}")

if __name__ == "__main__":
    main()
