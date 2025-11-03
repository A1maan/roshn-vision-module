# Roshn Vision Module - Construction Site Safety Detection

A YOLO12-based computer vision system for real-time detection of personal protective equipment (PPE) and safety violations on construction sites.

## 📋 Overview

This module uses a fine-tuned YOLOv12 model trained on construction site safety data to detect:

- **PPE Equipment**: Hardhat, Safety Vest, Gloves, Goggles, Mask, Safety Cone
- **Violations**: Missing equipment (NO-Hardhat, NO-Vest, NO-Gloves, etc.)
- **Safety Events**: Fall detection
- **People**: Person detection for safety monitoring

## 🎯 Model Details

- **Architecture**: YOLOv12 Nano (lightweight, real-time inference)
- **Input Size**: 768x768 pixels
- **Classes**: 14 (detection + violation classes)
- **Best Model**: `yolo12_run_3/yolo_runs/yolo12_run_3/weights/best.pt`
- **Framework**: Ultralytics YOLOv8/v12 API

### Detected Classes:
1. Fall-Detected
2. Gloves
3. Goggles
4. Hardhat
5. Ladder
6. Mask
7. NO-Gloves (violation)
8. NO-Goggles (violation)
9. NO-Hardhat (violation)
10. NO-Mask (violation)
11. NO-Safety Vest (violation)
12. Person
13. Safety Cone
14. Safety Vest

## 🚀 Quick Start

### 1. Setup Environment

**Option A: Using Conda (Recommended)**
```bash
# Create conda environment
conda create -n roshn-vision python=3.10 -y
conda activate roshn-vision

# Install dependencies
pip install -r requirements.txt
```

**Option B: Direct pip installation**
```bash
pip install -r requirements.txt
```

### 2. Run Inference

#### Single Image Inference
```bash
python inference.py --source path/to/image.jpg --output results/
```

#### Batch Image Processing
```bash
python inference.py --source path/to/images_folder/ --output results/
```

#### Video Inference
```bash
python inference.py --source path/to/video.mp4 --output results/
```

#### Real-time Webcam Inference
```bash
python inference.py --source 0 --show
```

### 3. Adjust Parameters

```bash
# Custom confidence threshold (0-1, higher = stricter)
python inference.py --source image.jpg --conf 0.5

# Custom IOU threshold for NMS
python inference.py --source image.jpg --iou 0.6

# Display results in real-time
python inference.py --source image.jpg --show

# Combine options
python inference.py --source video.mp4 --output results/ --conf 0.4 --show
```

## 📂 Project Structure

```
roshn-vision-module/
├── inference.py              # Main inference pipeline
├── test_inference.py         # Quick test script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── test-images/              # Sample images for testing
├── results/                  # Output directory (auto-created)
└── yolo12_run_3/            # Trained model weights
    └── yolo_runs/
        └── yolo12_run_3/
            ├── weights/
            │   ├── best.pt   # Best model checkpoint
            │   └── last.pt   # Last checkpoint
            ├── args.yaml     # Training arguments
            └── results.csv   # Training metrics
```

## 💻 Usage Examples

### Python Script Usage

```python
from inference import SafetyDetector

# Initialize detector
detector = SafetyDetector(
    model_path='yolo12_run_3/yolo_runs/yolo12_run_3/weights/best.pt',
    conf_threshold=0.25,
    iou_threshold=0.7
)

# Single image inference
results = detector.predict_image('path/to/image.jpg', save_path='output.jpg')

# Get detailed detections with class names
detections = detector.get_detailed_detections(results)
for det in detections:
    print(f"Found {det['class_name']} with {det['confidence']:.2%} confidence")

# Print formatted results
detector.print_detections(results)

# Batch processing
all_results = detector.predict_batch('path/to/images/', 'output_dir/')
```

### Command Line Examples

```bash
# Test with provided sample image
python inference.py --source test-images/istockphoto-2117759132-612x612.jpg

# Strict detection (high confidence threshold)
python inference.py --source images/ --conf 0.7 --output results/

# Real-time webcam with display
python inference.py --source 0 --show

# Process video and save annotated output
python inference.py --source construction_video.mp4 --output results/
```

## 📊 Output Format

### Console Output
```
Model loaded with 14 classes:
  0: Fall-Detected
  1: Gloves
  ...
  13: Safety Vest

Found 4 detection(s):
----------------------------------------------------------------------
1. Safety Vest
   Confidence: 35.20%
   BBox: [357.6, 179.5, 408.7, 249.8]
2. Hardhat
   Confidence: 92.15%
   BBox: [100.2, 150.4, 250.8, 400.1]
...
----------------------------------------------------------------------

Summary:
  Safety Vest: 2
  Hardhat: 1
  Person: 1
```

### Output Files
- **Annotated images**: Saved to `results/result_*.jpg` with bounding boxes and labels
- **Videos**: Processed and saved with frame-by-frame detections

## 🔧 API Reference

### SafetyDetector Class

#### `__init__(model_path, conf_threshold=0.25, iou_threshold=0.7)`
Initialize the detector with a YOLO model.

**Parameters:**
- `model_path` (str): Path to .pt model file
- `conf_threshold` (float): Confidence threshold (0-1)
- `iou_threshold` (float): IOU threshold for NMS (0-1)

#### `predict_image(image_path, save_path=None, show=False)`
Run inference on a single image.

**Returns:** List of Results objects

#### `predict_video(video_path, output_path=None, show=False)`
Run inference on a video (streaming).

**Returns:** Generator of Results objects

#### `predict_batch(input_dir, output_dir=None)`
Process all images in a directory.

**Returns:** List of Results objects

#### `get_detailed_detections(results)`
Extract detailed information from detections.

**Returns:** List of dicts with keys: `class_id`, `class_name`, `confidence`, `bbox`

#### `get_detection_summary(results)`
Get count of each class detected.

**Returns:** Dict with class names as keys and counts as values

#### `print_detections(results)`
Print formatted detection information.

## 📦 Dependencies

See `requirements.txt`:
- `ultralytics>=8.0.0` - YOLO framework
- `opencv-python>=4.8.0` - Image processing
- `torch>=2.0.0` - PyTorch backend
- `torchvision>=0.15.0` - Vision utilities
- `numpy>=1.24.0` - Numerical computing
- `pillow>=10.0.0` - Image library

## ⚙️ Configuration

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf_threshold` | 0.2 | Confidence threshold for detections |
| `iou_threshold` | 0.7 | IOU threshold for NMS |
| `imgsz` | 768 | Input image size |

Adjust these based on your use case:
- **Higher confidence** = fewer false positives, may miss detections
- **Lower confidence** = more detections, higher false positive rate

## 🎓 Training Data

- **Dataset**: Construction-Site-Safety-2 (from Roboflow)
- **Training Epochs**: 80
- **Batch Size**: Auto (-1)
- **Optimizer**: AdamW
- **Learning Rate**: 0.003 (with cosine annealing)

## 📝 Notes

- Model trained on 768x768 images for better detail on construction sites
- YOLOv12 Nano provides good balance of speed and accuracy
- Best model selected via early stopping (patience: 20 epochs)
- Real-time inference (~100-200ms per image on GPU)

