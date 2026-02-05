# DRSQ-Net
Underwater object detection faces challenges due to color distortion, multi-scale targets, and occlusion. This study introduces a novel network, DRSQ-Net, integrating differentiable physical restoration and dynamic sparse queries to enhance detection accuracy.

# DRSQ-Net: Enhancing Underwater Object Detection through Differentiable Physical Restoration and Dynamic Feature Enhancement
Official PyTorch implementation of DRSQ-Net, a novel underwater object detection framework that integrates Differentiable Physical Restoration (DPR), Local-Global Collaborative Transformer (LGC-Former), and Dynamic Sparse Query detection head (DSQ-Head). This repository contains code, pre-trained models, and training scripts for the paper:
> Enhancing Underwater Object Detection through Differentiable Physical Restoration and Dynamic Feature Enhancement
Achieves state-of-the-art 92.1% mAP@0.5 on URPC2020 with only 4.2M parameters and >70 FPS inference speed.

## ✨ Key Features

- 🧩 Differentiable Physical Restoration (DPR): Retinex-based physical model that decouples and restores illumination/reflectance components in an end-to-end differentiable manner
- 🌐 Local-Global Collaborative Transformer (LGC-Former): Dual-branch architecture combining dynamic convolution for local detail enhancement and Vision Mamba for global context modeling
- 🎯 Dynamic Sparse Query Head (DSQ-Head): Anchor-free, NMS-free detection head with task-aware dynamic routing and learnable object queries
- ⚡ Lightweight & Efficient: Only 4.2M parameters and 11.5 GFLOPs with real-time inference capability
- 📊 State-of-the-Art Performance: Superior results on multiple underwater datasets (URPC2020, RUOD, UTDAC2020)

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Zoo](#-model-zoo)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

## 🛠️ Installation

### Prerequisites
- Linux or macOS (Windows with WSL2 recommended)
- Python 3.8 or higher
- CUDA 11.3 or higher (for GPU acceleration)
- 8GB+ RAM, 20GB+ free disk space

### Step-by-Step Setup

bash
# 1. Clone the repository
git clone https://github.com/yourusername/DRSQ-Net.git
cd DRSQ-Net

# 2. Create and activate conda environment (recommended)
conda create -n drsqnet python=3.8 -y
conda activate drsqnet

# 3. Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for the correct command for your system
# Example for CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Install DRSQ-Net as a package (optional, for development)
pip install -e .

### Verify Installation
bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python scripts/verify_installation.py

## 🚀 Quick Start

### Inference with Pre-trained Model

bash
# Download a pre-trained model (URPC2020, 92.1% mAP)
python scripts/download_weights.py --model drsqnet_small

# Run inference on a sample image
python tools/inference.py \
    --image assets/sample.jpg \
    --checkpoint weights/drsqnet_small_urpc2020.pth \
    --output results/detection.jpg

# Run inference on a video
python tools/inference.py \
    --video assets/sample_video.mp4 \
    --checkpoint weights/drsqnet_small_urpc2020.pth \
    --output results/detection_video.avi \
    --show

### Training on Custom Data

bash
# 1. Prepare your dataset in COCO format
python tools/prepare_data.py \
    --input /path/to/your/images \
    --annotations /path/to/annotations.json \
    --output data/custom_dataset

# 2. Train DRSQ-Net
python tools/train.py \
    --config configs/drsqnet_small.yaml \
    --data data/custom_dataset \
    --name custom_experiment \
    --epochs 300 \
    --batch-size 16

## Training

### Training on URPC2020 Dataset

bash
# 1. Download and prepare URPC2020 dataset
python scripts/download_datasets.py --dataset urpc2020
python tools/preprocess_dataset.py --dataset urpc2020 --img-size 640

# 2. Train DRSQ-Net-Small
python tools/train.py \
    --config configs/drsqnet_small.yaml \
    --data data/URPC2020 \
    --name drsqnet_small_urpc2020 \
    --epochs 300 \
    --batch-size 16 \
    --workers 8 \
    --device 0  # Use GPU 0

# 3. Monitor training (optional)
tensorboard --logdir runs/drsqnet_small_urpc2020

### Multi-GPU Training

bash
# Distributed Data Parallel (DDP) training
torchrun --nproc_per_node=4 tools/train.py \
    --config configs/drsqnet_large.yaml \
    --data data/URPC2020 \
    --batch-size 64 \
    --sync-bn \
    --amp  # Automatic Mixed Precisio

### Training with Custom Configuration

python
# train_custom.py
import yaml
from models import DRSQNet
from tools.trainer import Trainer

# Load and modify config
with open('configs/drsqnet_small.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['model']['num_classes'] = 10  # Your number of classes
config['data']['img_size'] = 800
config['training']['lr'] = 0.001

# Initialize model
model = DRSQNet(
    num_classes=config['model']['num_classes'],
    img_size=config['data']['img_size'],
    version='small'
)

# Train
trainer = Trainer(model, config)
trainer.fit()

### Advanced Training Options

bash
# Resume from checkpoint
python tools/train.py \
    --resume checkpoints/latest.pth \
    --config configs/drsqnet_small.yaml

# Train with validation only
python tools/train.py \
    --validate-only \
    --checkpoint checkpoints/best.pth

# Hyperparameter search
python tools/hpo.py \
    --config configs/hpo_drsqnet.yaml \
    --trials 50 \
    --gpus 2

## 📊 Evaluation

### Evaluate on Test Set

bash
# Evaluate DRSQ-Net on URPC2020 test set
python tools/test.py \
    --config configs/drsqnet_small.yaml \
    --checkpoint weights/drsqnet_small_urpc2020.pth \
    --data data/URPC2020 \
    --task test \
    --save-json \
    --save-txt


### Compare with Other Models

bash
# Benchmark DRSQ-Net against other detectors
python tools/benchmark.py \
    --models yolov8 yolov11 drsqnet_small \
    --data data/URPC2020 \
    --batch-sizes 1 8 16 \
    --img-sizes 320 640 1280

### Export Metrics
bash
# Export detailed metrics to CSV/JSON
python tools/export_metrics.py \
    --results runs/drsqnet_small_urpc2020/results.json \
    --format csv json html

# Generate comparison plots
python tools/visualize_results.py \
    --results-dir runs/ \
    --output-dir plots/ \
    --metrics map recall fps params

## 🎯 Inference

### Basic Usage

python
# inference_demo.py
import cv2
import torch
from models.drsqnet import DRSQNet
from utils.visualization import visualize_detections

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DRSQNet(num_classes=4, version='small').to(device)
model.load_state_dict(torch.load('weights/drsqnet_small_urpc2020.pth'))
model.eval()

# Load and preprocess image
image = cv2.imread('assets/sample.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Inference
with torch.no_grad():
    detections = model(image_rgb)

# Visualize results
result_image = visualize_detections(
    image, 
    detections,
    class_names=['echinus', 'starfish', 'holothurian', 'scallop'],
    conf_threshold=0.5
)

cv2.imwrite('result.jpg', result_image)

### Advanced Inference Options

```bash
# Real-time webcam detection
python tools/inference.py \
    --source 0 \  # Webcam
    --checkpoint weights/drsqnet_small_urpc2020.pth \
    --conf-thres 0.5 \
    --iou-thres 0.45 \
    --show \
    --save

# Process multiple images
python tools/inference.py \
    --source data/images/*.jpg \
    --checkpoint weights/drsqnet_small_urpc2020.pth \
    --output results/batch_detections/ \
    --save-txt \
    --save-conf

# Video processing with tracking
python tools/inference.py \
    --video assets/underwater_video.mp4 \
    --checkpoint weights/drsqnet_small_urpc2020.pth \
    --track \  # Enable object tracking
    --output results/tracked_video.avi
```

### Model Export for Deployment

bash
# Export to ONNX
python tools/export.py \
    --checkpoint weights/drsqnet_small_urpc2020.pth \
    --format onnx \
    --opset 12 \
    --dynamic \
    --simplify

# Export to TensorRT
python tools/export.py \
    --checkpoint weights/drsqnet_small_urpc2020.pth \
    --format tensorrt \
    --workspace 8 \
    --fp16

# Export to TorchScript
python tools/export.py \
    --checkpoint weights/drsqnet_small_urpc2020.pth \
    --format torchscript \
    --optimize

# Test exported model
python tools/test_exported.py \
    --model exported/drsqnet_small.onnx \
    --data data/URPC2020 \
    --format onnx

## 📁 Datasets

### Supported Datasets

| Dataset | Images | Classes | Description | Download |
|---------|--------|---------|-------------|----------|
| URPC2020 | 6,300 | 4 | Underwater Robot Professional Contest 2020 | [Official](http://www.urpc.org.cn) |
| RUOD | 14,000 | 10 | Large-scale underwater benchmark | [GitHub](https://github.com/dlut-dimt/RUOD) |
| UTDAC2020 | 6,461 | 4 | Underwater Target Detection Algorithm Competition | [Request](mailto:utdac@hrbeu.edu.cn) |

### Dataset Preparation

bash
# Download all supported datasets
python scripts/download_datasets.py --all

# Or download specific datasets
python scripts/download_datasets.py --dataset urpc2020 ruod

# Convert custom dataset to COCO format
python tools/convert_dataset.py \
    --input-format yolo \  # or voc, coco, labelme
    --input-dir /path/to/your/data \
    --output-dir data/custom_coco \
    --split-ratio 0.8 0.1 0.1



## 🗂️ Project Structure

DRSQ-Net/
├── configs/                    # Configuration files
│   ├── drsqnet_small.yaml     # Small model config (4.2M params)
│   ├── drsqnet_medium.yaml    # Medium model config (8.7M params)
│   ├── drsqnet_large.yaml     # Large model config (16.3M params)
│   └── drsqnet_tiny.yaml      # Tiny model config (2.1M params)
├── data/                       # Data loading and processing
│   ├── datasets.py            # Dataset classes
│   ├── augmentations.py       # Data augmentation
│   ├── transforms.py          # Image transformations
│   └── collate.py             # Batch collation
├── docs/                       # Documentation
│   ├── architecture.md        # Architecture details
│   ├── training_guide.md      # Training guide
│   └── api_reference.md       # API reference
├── models/                     # Model definitions
│   ├── drsqnet.py             # Main DRSQ-Net model
│   ├── dpr_module.py          # Differentiable Physical Restoration
│   ├── lgc_former.py          # Local-Global Collaborative Transformer
│   ├── dsq_head.py            # Dynamic Sparse Query Head
│   ├── backbone.py            # Backbone networks
│   └── necks.py               # Neck networks (FPN, etc.)
├── tools/                      # Training and evaluation scripts
│   ├── train.py               # Main training script
│   ├── test.py                # Evaluation script
│   ├── inference.py           # Inference script
│   ├── export.py              # Model export
│   └── benchmark.py           # Performance benchmarking
├── utils/                      # Utility functions
│   ├── losses.py              # Loss functions
│   ├── metrics.py             # Evaluation metrics
│   ├── visualizer.py          # Visualization utilities
│   ├── logger.py              # Logging utilities
│   └── checkpoint.py          # Checkpoint management
├── scripts/                    # Helper scripts
│   ├── download_weights.py    # Download pre-trained models
│   ├── download_datasets.py   # Download datasets
│   ├── setup_environment.sh   # Environment setup
│   └── verify_installation.py # Verify installation
├── weights/                    # Pre-trained models
├── runs/                      # Training runs and experiments
├── assets/                    # Sample images and videos
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
└── README.md                  # This file
