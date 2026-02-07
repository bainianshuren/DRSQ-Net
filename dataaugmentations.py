"""
Advanced data augmentations for underwater object detection
Includes Mosaic and MixUp as described in the paper
"""
import random
import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from typing import List, Tuple, Dict, Any

class Mosaic(DualTransform):
    """
    Mosaic augmentation: combine 4 images into one
    """
    def __init__(
        self,
        img_size: int = 640,
        p: float = 0.5,
        always_apply: bool = False,
    ):
        super().__init__(always_apply, p)
        self.img_size = img_size
    
    def apply(
        self,
        img: np.ndarray,
        selected_imgs: List[np.ndarray],
        selected_bboxes: List[List],
        selected_labels: List[List],
        **params
    ) -> Tuple[np.ndarray, List, List]:
        """Apply mosaic augmentation"""
        if len(selected_imgs) < 4:
            return img, params.get('bboxes', []), params.get('labels', [])
        
        # Output image
        output_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        output_bboxes = []
        output_labels = []
        
        # Randomly choose split points
        split_x = random.randint(int(self.img_size * 0.3), int(self.img_size * 0.7))
        split_y = random.randint(int(self.img_size * 0.3), int(self.img_size * 0.7))
        
        # Define 4 quadrants
        quadrants = [
            (0, 0, split_x, split_y),  # Top-left
            (split_x, 0, self.img_size, split_y),  # Top-right
            (0, split_y, split_x, self.img_size),  # Bottom-left
            (split_x, split_y, self.img_size, self.img_size),  # Bottom-right
        ]
        
        for idx, (x1, y1, x2, y2) in enumerate(quadrants):
            img_idx = idx % len(selected_imgs)
            img_part = selected_imgs[img_idx]
            bboxes_part = selected_bboxes[img_idx]
            labels_part = selected_labels[img_idx]
            
            # Resize quadrant to fit the part
            h, w = img_part.shape[:2]
            scale_x = (x2 - x1) / w
            scale_y = (y2 - y1) / h
            scale = min(scale_x, scale_y)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized_img = cv2.resize(img_part, (new_w, new_h))
            
            # Place in quadrant (centered)
            pad_x = (x2 - x1 - new_w) // 2
            pad_y = (y2 - y1 - new_h) // 2
            
            x_start = x1 + pad_x
            y_start = y1 + pad_y
            x_end = x_start + new_w
            y_end = y_start + new_h
            
            output_img[y_start:y_end, x_start:x_end] = resized_img
            
            # Adjust bounding boxes
            for bbox, label in zip(bboxes_part, labels_part):
                x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox
                
                # Scale bbox
                x1_new = x_start + x1_bbox * scale
                y1_new = y_start + y1_bbox * scale
                x2_new = x_start + x2_bbox * scale
                y2_new = y_start + y2_bbox * scale
                
                # Check if bbox is valid
                if (x2_new - x1_new > 1 and y2_new - y1_new > 1 and
                    x1_new < self.img_size and y1_new < self.img_size and
                    x2_new > 0 and y2_new > 0):
                    
                    output_bboxes.append([x1_new, y1_new, x2_new, y2_new])
                    output_labels.append(label)
        
        return output_img, output_bboxes, output_labels
    
    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters dependent on targets"""
        # This is called by albumentations
        return params
    
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("img_size", "p")


class MixUp(DualTransform):
    """
    MixUp augmentation: blend two images
    """
    def __init__(
        self,
        alpha: float = 0.8,
        p: float = 0.5,
        always_apply: bool = False,
    ):
        super().__init__(always_apply, p)
        self.alpha = alpha
    
    def apply(
        self,
        img: np.ndarray,
        mix_img: np.ndarray,
        mix_bboxes: List[List],
        mix_labels: List[List],
        **params
    ) -> Tuple[np.ndarray, List, List]:
        """Apply mixup augmentation"""
        if mix_img is None:
            return img, params.get('bboxes', []), params.get('labels', [])
        
        # Get current image bboxes and labels
        bboxes = params.get('bboxes', [])
        labels = params.get('labels', [])
        
        # Blend images
        lam = np.random.beta(self.alpha, self.alpha)
        blended_img = (img * lam + mix_img * (1 - lam)).astype(np.uint8)
        
        # Combine bboxes and labels
        combined_bboxes = bboxes + mix_bboxes
        combined_labels = labels + mix_labels
        
        return blended_img, combined_bboxes, combined_labels
    
    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters dependent on targets"""
        # This would need access to other images in the batch
        # Simplified version: randomly select from a small pool
        return params
    
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("alpha", "p")


class UnderwaterAugmentation(DualTransform):
    """
    Special augmentations for underwater images
    Simulates common underwater degradations
    """
    def __init__(
        self,
        blur_limit: int = 5,
        haze_intensity: float = 0.1,
        color_cast_intensity: float = 0.05,
        noise_intensity: float = 0.02,
        p: float = 0.3,
        always_apply: bool = False,
    ):
        super().__init__(always_apply, p)
        self.blur_limit = blur_limit
        self.haze_intensity = haze_intensity
        self.color_cast_intensity = color_cast_intensity
        self.noise_intensity = noise_intensity
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply underwater-specific augmentations"""
        if random.random() > self.p:
            return img
        
        # Random blur (simulating turbid water)
        if random.random() < 0.3:
            ksize = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        # Add haze effect (simulating scattering)
        if random.random() < 0.4:
            haze = np.ones_like(img) * 200 * self.haze_intensity
            alpha = random.uniform(0.1, 0.3)
            img = cv2.addWeighted(img, 1 - alpha, haze.astype(np.uint8), alpha, 0)
        
        # Color cast (blue-green shift)
        if random.random() < 0.5:
            # Increase blue and green channels
            blue_shift = random.uniform(1.0, 1.0 + self.color_cast_intensity)
            green_shift = random.uniform(1.0, 1.0 + self.color_cast_intensity)
            red_shift = random.uniform(1.0 - self.color_cast_intensity * 0.5, 1.0)
            
            img = img.astype(np.float32)
            img[:, :, 0] = np.clip(img[:, :, 0] * blue_shift, 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * green_shift, 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * red_shift, 0, 255)
            img = img.astype(np.uint8)
        
        # Add noise (simulating low-light conditions)
        if random.random() < 0.3:
            noise = np.random.normal(0, self.noise_intensity * 255, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return (
            "blur_limit",
            "haze_intensity",
            "color_cast_intensity",
            "noise_intensity",
            "p",
        )


def get_underwater_augmentations(
    img_size: int = 640,
    mosaic_prob: float = 0.5,
    mixup_prob: float = 0.3,
    is_training: bool = True,
) -> A.Compose:
    """
    Get comprehensive augmentations for underwater object detection
    
    Args:
        img_size: Target image size
        mosaic_prob: Probability of mosaic augmentation
        mixup_prob: Probability of mixup augmentation
        is_training: Whether this is for training
    
    Returns:
        Albumentations Compose object
    """
    if not is_training:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # Training augmentations
    transforms = []
    
    # Geometric augmentations
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
    ])
    
    # Color augmentations
    transforms.extend([
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.CLAHE(p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RGBShift(
            r_shift_limit=20,
            g_shift_limit=20,
            b_shift_limit=20,
            p=0.3
        ),
        A.ChannelShuffle(p=0.1),
    ])
    
    # Underwater-specific augmentations
    transforms.append(UnderwaterAugmentation(p=0.3))
    
    # Noise and blur
    transforms.extend([
        A.Blur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ISONoise(p=0.2),
        A.ImageCompression(quality_lower=70, p=0.2),
    ])
    
    # Advanced augmentations
    if mosaic_prob > 0:
        transforms.append(Mosaic(img_size=img_size, p=mosaic_prob))
    if mixup_prob > 0:
        transforms.append(MixUp(alpha=0.8, p=mixup_prob))
    
    # Final transforms
    transforms.extend([
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.5, 1.0),
            p=0.5
        ),
        A.Resize(img_size, img_size, always_apply=True),
        A.CoarseDropout(
            max_holes=10,
            max_height=img_size//10,
            max_width=img_size//10,
            p=0.2
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.3,
            min_area=16  # Minimum area for small objects
        )
    )