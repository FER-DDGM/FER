import os
import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
import random
import albumentations as A


class YOLOEmotionTrainer:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.data_dir = "emotion_data"

        # Image dimensions and sizing
        self.img_size = 128
        self.face_size_min = 20
        self.face_size_max = 118
        self.margin = 10

        # Dataset limits
        self.train_limit = 3000
        self.val_limit = 500
        self.test_limit = 500
        self.train_variations = 3
        self.val_variations = 1

        # Background settings
        self.bg_color_min = 20
        self.bg_color_max = 235
        self.noise_std = 15
        self.face_alpha = 0.95
        self.bg_objects_prob = 0.7
        self.bg_objects_count_min = 1
        self.bg_objects_count_max = 3
        self.bg_object_size = 8
        self.bg_object_radius = 4

        # Augmentation settings
        self.aug_prob = 0.3

        self.augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.CLAHE(p=0.2),
            A.RandomGamma(p=0.2),
        ])

    def create_realistic_images(self, face_img, emotion_id, img_index, split):
        images_created = []

        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)

        num_variations = 3 if split == 'train' else 1

        for var in range(num_variations):
            bg_color = np.random.randint(20, 235, 3)
            img = np.full((self.img_size, self.img_size, 3), bg_color, dtype=np.uint8)

            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            face_size = random.randint(self.face_size_min, self.face_size_max)
            margin = 50
            center_x = random.randint(margin + face_size // 2, self.img_size - margin - face_size // 2)
            center_y = random.randint(margin + face_size // 2, self.img_size - margin - face_size // 2)

            resized_face = cv2.resize(face_img, (face_size, face_size))

            if split == 'train' and random.random() > self.aug_prob:
                augmented = self.augment(image=resized_face)
                resized_face = augmented['image']

            x1 = center_x - face_size // 2
            y1 = center_y - face_size // 2
            x2 = x1 + face_size
            y2 = y1 + face_size

            img[y1:y2, x1:x2] = (self.face_alpha * resized_face + (1 - self.face_alpha) * img[y1:y2, x1:x2]).astype(
                np.uint8)

            if random.random() > self.bg_objects_prob:
                self.add_background_objects(img, (x1, y1, x2, y2))

            img_name = f"img_{img_index:05d}_v{var}.jpg"
            img_path = f"{self.data_dir}/{split}/images/{img_name}"
            cv2.imwrite(img_path, img)

            bbox_x = center_x / self.img_size
            bbox_y = center_y / self.img_size
            bbox_w = face_size / self.img_size
            bbox_h = face_size / self.img_size

            label_name = f"img_{img_index:05d}_v{var}.txt"
            label_path = f"{self.data_dir}/{split}/labels/{label_name}"
            with open(label_path, 'w') as f:
                f.write(f"{emotion_id} {bbox_x:.6f} {bbox_y:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

            images_created.append(img_path)

        return images_created

    def add_background_objects(self, img, face_bbox):
        x1, y1, x2, y2 = face_bbox

        for _ in range(random.randint(self.bg_objects_count_min, self.bg_objects_count_max)):
            obj_size = self.bg_object_size * 2 - 1  # Account for radius
            obj_x = random.randint(0, self.img_size - obj_size)
            obj_y = random.randint(0, self.img_size - obj_size)

            if (obj_x < x2 and obj_x + obj_size > x1 and obj_y < y2 and obj_y + obj_size > y1):
                continue

            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            if random.random() > 0.5:
                cv2.rectangle(img, (obj_x, obj_y), (obj_x + self.bg_object_size, obj_y + self.bg_object_size), color,
                              -1)
            else:
                cv2.circle(img, (obj_x + self.bg_object_radius, obj_y + self.bg_object_radius), self.bg_object_radius,
                           color, -1)

    def prepare_dataset(self, fer_csv_path="fer2013.csv"):
        print("Converting dataset...")

        for split in ['train', 'val', 'test']:
            os.makedirs(f"{self.data_dir}/{split}/images", exist_ok=True)
            os.makedirs(f"{self.data_dir}/{split}/labels", exist_ok=True)

        df = pd.read_csv(fer_csv_path)
        print(f"Total samples: {len(df)}")

        train_count = val_count = test_count = 0
        df = df.sample(frac=1).reset_index(drop=True)

        for idx, row in df.iterrows():
            emotion = int(row['emotion'])
            pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
            usage = row['Usage'].lower()

            if usage == 'training' and train_count < self.train_limit:
                split = 'train'
                train_count += 1
            elif usage == 'publictest' and val_count < self.val_limit:
                split = 'val'
                val_count += 1
            elif usage == 'privatetest' and test_count < self.test_limit:
                split = 'test'
                test_count += 1
            else:
                continue

            self.create_realistic_images(pixels, emotion, idx, split)

            if idx % 500 == 0:
                print(f"Processed {idx} samples - Train: {train_count}, Val: {val_count}, Test: {test_count}")

        print("Dataset conversion complete!")
        print(f"Final counts - Train: {train_count * self.train_variations}, Val: {val_count}, Test: {test_count}")

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

        return config_path

    def train_model(self, epochs=50, batch_size=16):
        print("Training model...")

        config_path = self.create_config()
        model = YOLO('yolov8n.pt')

        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=self.img_size,
            batch=batch_size,
            device=0,
            name='emotion_yolo_v2',
            project='runs/detect',

            patience=15,
            dropout=0.2,
            weight_decay=0.001,
            lr0=0.001,
            warmup_epochs=5,

            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.2,
            shear=5,
            perspective=0.001,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.8,
            mixup=0.1,

            save=True,
            plots=True,
            verbose=True
        )

        best_model_path = 'runs/detect/emotion_yolo_v2/weights/best.pt'
        return best_model_path, results


def main():
    print("YOLO Emotion Detection Training")
    print("=" * 40)

    trainer = YOLOEmotionTrainer()

    print("Preparing dataset...")
    trainer.prepare_dataset()

    epochs = int(input("Epochs (default 40): ") or "40")

    print("Training...")
    best_model, results = trainer.train_model(epochs=epochs, batch_size=16)

    print(f"Training complete! Best model: {best_model}")


if __name__ == "__main__":
    main()
