import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

class UnderwaterDataset(Dataset):
    """Underwater object detection dataset for URPC2020, RUOD, UTDAC2020"""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 640,
        augment: bool = False,
        mosaic: bool = True,
        mixup: float = 0.5,
        cache_images: bool = False,
    ):
        """
        Args:
            root_dir: Dataset root directory
            split: 'train', 'val', or 'test'
            img_size: Input image size
            augment: Whether to apply data augmentation
            mosaic: Whether to use mosaic augmentation
            mixup: Mixup probability
            cache_images: Cache images in memory for faster training
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"
        self.mosaic = mosaic and self.augment
        self.mixup = mixup
        self.cache_images = cache_images
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Class names for URPC2020
        self.class_names = ["echinus", "starfish", "holothurian", "scallop"]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # Image cache
        self.imgs = [None] * len(self.annotations)
        self.img_shapes = [None] * len(self.annotations)
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        print(f"Dataset loaded: {len(self)} images, {self.num_classes} classes")
    
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations"""
        annotations = []
        
        # Check for different annotation formats
        annotation_files = [
            self.root_dir / f"{self.split}.json",  # COCO format
            self.root_dir / f"{self.split}" / "labels",  # YOLO format
            self.root_dir / "annotations.json",  # Single file
        ]
        
        for ann_file in annotation_files:
            if ann_file.exists():
                if str(ann_file).endswith('.json'):
                    return self._load_coco_annotations(ann_file)
                elif ann_file.is_dir():
                    return self._load_yolo_annotations(ann_file)
        
        # If no annotation file found, create dummy annotations for image files
        img_dir = self.root_dir / self.split / "images" if (self.root_dir / self.split / "images").exists() else self.root_dir / self.split
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        
        for img_file in img_files:
            annotations.append({
                "image_id": img_file.stem,
                "file_name": str(img_file),
                "image_size": (640, 640, 3),  # Default size
                "bboxes": [],
                "labels": [],
                "areas": [],
                "iscrowd": [],
            })
        
        return annotations
    
    def _load_coco_annotations(self, ann_file: Path) -> List[Dict]:
        """Load COCO format annotations"""
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Create image id to info mapping
        img_info = {img['id']: img for img in data['images']}
        
        # Group annotations by image
        img_annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = {
                    "bboxes": [],
                    "labels": [],
                    "areas": [],
                    "iscrowd": [],
                }
            
            # Convert COCO bbox [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            bbox = [x, y, x + w, y + h]
            
            img_annotations[img_id]["bboxes"].append(bbox)
            img_annotations[img_id]["labels"].append(ann['category_id'])
            img_annotations[img_id]["areas"].append(ann['area'])
            img_annotations[img_id]["iscrowd"].append(ann.get('iscrowd', 0))
        
        # Create final annotations list
        annotations = []
        for img_id, img in img_info.items():
            anns = img_annotations.get(img_id, {
                "bboxes": [], "labels": [], "areas": [], "iscrowd": []
            })
            
            annotations.append({
                "image_id": img_id,
                "file_name": str(self.root_dir / img['file_name']),
                "image_size": (img['height'], img['width'], 3),
                "bboxes": anns["bboxes"],
                "labels": anns["labels"],
                "areas": anns["areas"],
                "iscrowd": anns["iscrowd"],
            })
        
        return annotations
    
    def _load_yolo_annotations(self, label_dir: Path) -> List[Dict]:
        """Load YOLO format annotations"""
        annotations = []
        img_dir = label_dir.parent / "images"
        
        for label_file in label_dir.glob("*.txt"):
            img_file = img_dir / f"{label_file.stem}.jpg"
            
            if not img_file.exists():
                continue
            
            # Read image to get size
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Parse YOLO annotations
            bboxes = []
            labels = []
            areas = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        box_w = float(parts[3]) * w
                        box_h = float(parts[4]) * h
                        
                        # Convert to [x1, y1, x2, y2]
                        x1 = x_center - box_w / 2
                        y1 = y_center - box_h / 2
                        x2 = x_center + box_w / 2
                        y2 = y_center + box_h / 2
                        
                        bboxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
                        areas.append(box_w * box_h)
            
            annotations.append({
                "image_id": label_file.stem,
                "file_name": str(img_file),
                "image_size": (h, w, 3),
                "bboxes": bboxes,
                "labels": labels,
                "areas": areas,
                "iscrowd": [0] * len(bboxes),
            })
        
        return annotations
    
    def _get_transforms(self):
        """Get data augmentation transforms"""
        if self.augment:
            return A.Compose([
                A.Mosaic(img_size=self.img_size, p=0.5 if self.mosaic else 0.0),
                A.MixUp(p=self.mixup),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.0),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.Blur(blur_limit=3, p=0.1),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.ToGray(p=0.01),
                A.ISONoise(p=0.2),
                A.GaussNoise(p=0.2),
                A.RandomResizedCrop(
                    height=self.img_size,
                    width=self.img_size,
                    scale=(0.5, 1.0),
                    p=0.5
                ),
                A.Resize(self.img_size, self.img_size, always_apply=True),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size, always_apply=True),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3
            ))
    
    def _load_image(self, idx: int) -> np.ndarray:
        """Load image from disk or cache"""
        if self.cache_images and self.imgs[idx] is not None:
            return self.imgs[idx].copy()
        
        img_path = self.annotations[idx]["file_name"]
        img = cv2.imread(img_path)
        
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.cache_images:
            self.imgs[idx] = img
            self.img_shapes[idx] = img.shape
        
        return img
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get image and annotations"""
        # Load image
        img = self._load_image(idx)
        ann = self.annotations[idx]
        
        # Get bounding boxes and labels
        bboxes = ann["bboxes"]
        labels = ann["labels"]
        
        if len(bboxes) == 0:
            # Create dummy annotation for images without objects
            bboxes = [[0, 0, 1, 1]]
            labels = [0]
        
        # Apply transforms
        transformed = self.transform(
            image=img,
            bboxes=bboxes,
            labels=labels
        )
        
        image = transformed["image"]
        bboxes = transformed["bboxes"]
        labels = transformed["labels"]
        
        # Convert to tensors
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            areas = torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in bboxes], dtype=torch.float32)
            iscrowd = torch.tensor([0] * len(bboxes), dtype=torch.uint8)
        else:
            # Create dummy tensors
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.uint8)
        
        target = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor(img.shape[:2]),
        }
        
        return image, target
    
    @property
    def num_classes(self) -> int:
        return len(self.class_names)
    
    def get_dataloader(
        self,
        batch_size: int = 16,
        shuffle: bool = None,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Get DataLoader for this dataset"""
        if shuffle is None:
            shuffle = self.split == "train"
        
        collate_fn = CollateFunction()
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
        )


class CollateFunction:
    """Custom collate function for batches with variable number of objects"""
    
    def __call__(self, batch):
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        return images, targets