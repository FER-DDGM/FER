# Documentation: ImprovedEmotionTrainer Class

## Overview

The `ImprovedEmotionTrainer` class is a high-level utility for preparing, training, validating, and exporting
YOLOv8-based emotion classification models using the FER+ dataset. It automates:

- Dataset download and verification (via KaggleHub)
- Preprocessing and face detection
- Data augmentation and balancing
- YOLOv8 training and evaluation
- Configuration and export of the final model

---

## Class: `ImprovedEmotionTrainer`

### Initialization

```python
trainer = ImprovedEmotionTrainer()
```

Sets up:

- Emotion labels
- Folder structure for training data
- Augmentation strategies for training and validation
- OpenCV Haar cascade for optional face cropping
- FER+ emotion ID mapping

---

### Method: `download_ferplus_dataset()`

Downloads and verifies the FER+ dataset structure using `kagglehub`.
Steps performed:

1) Initiates download using kagglehub.dataset_download("msambare/fer2013").
2) Prints the dataset path.
3) Verifies existence of all required emotion folders in both train/ and test/.
4) Logs any missing folders or errors.

**Returns**: `True` if dataset is valid, else `False`.

---

### Method: `load_ferplus_images(split)`

Loads image metadata for a given split (`train` or `test`).

**Args**:

- `split`: One of `'train'` or `'test'`

**Returns**: A list of dictionaries with `path`, `emotion_id`, `emotion_name`, and `split`.

---

### Method: `detect_and_crop_faces(image, padding=0.3)`

Uses OpenCV’s Haar cascade to detect and crop faces from an image with optional padding.

**Args**:

- `image`: Numpy image array
- `padding`: Proportional padding to add around detected faces

**Returns**: List of cropped face image arrays.

---

### Method: `prepare_ferplus_dataset()`

Main dataset preprocessing pipeline. It:

- Loads images by emotion class
- Applies augmentation (via Albumentations)
- Crops or center-crops images
- Creates YOLO-compatible `.jpg` and `.txt` files for each image
- Balances number of samples per class using user-defined limits

**Returns**: `True` if data prep is successful, else `False`

---

### Method: `create_config()`

Generates a YOLOv8-style `config.yaml` file that defines dataset paths and emotion class names.

**Returns**: Path to the YAML file

---

### Method: `train_model(model_size='s', epochs=100, batch_size=16, resume=False)`

Launches training of a YOLOv8 model using Ultralytics interface.

**Args**:

- `model_size`: YOLO size variant (`n`, `s`, `m`, `l`, `x`)
- `epochs`: Number of training epochs
- `batch_size`: Images per batch
- `resume`: Resume from latest checkpoint

**Returns**: Tuple of `(best_model_path, results_object)`

---

### Method: `is_gpu_available()`

Static utility method to check for CUDA GPU availability using PyTorch.

**Returns**: `True` if CUDA GPU is found, else `False`

---

### Method: `validate_model(model_path)`

Performs validation on the trained YOLO model against the test set.

**Args**:

- `model_path`: Path to `.pt` file

**Returns**: YOLO validation results object

---

### Method: `export_model(model_path, formats=['onnx', 'torchscript'])`

Exports the trained model to specified formats for deployment.

**Args**:

- `model_path`: Path to `.pt` file
- `formats`: List of export types (`onnx`, `torchscript`, etc.)

**Output**: Exported files saved locally in appropriate format

---

## CLI Interface: `main()`

The `main()` function provides a guided CLI to:

1. Download and validate FER+ dataset
2. Preprocess and prepare data
3. Collect user-defined training parameters
4. Train the model and show progress
5. Optionally validate and export the trained model

---

## Example Usage

```python
trainer = ImprovedEmotionTrainer()
if trainer.download_ferplus_dataset():
    if trainer.prepare_ferplus_dataset():
        path, results = trainer.train_model(model_size='s', epochs=100)
        trainer.validate_model(path)
        trainer.export_model(path)
```

---

## Note

- Ensure you have GPU enabled for optimal training speed.

