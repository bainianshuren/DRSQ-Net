"""
Image transformations for preprocessing and postprocessing
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Union

def normalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """Normalize image with mean and std"""
    image = image.astype(np.float32) / 255.0
    if mean is not None:
        image -= mean
    if std is not None:
        image /= std
    return image


def denormalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """Denormalize image"""
    image = image.copy()
    if std is not None:
        image *= std
    if mean is not None:
        image += mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    keep_aspect_ratio: bool = True,
    padding_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with padding to maintain aspect ratio
    
    Args:
        image: Input image (H, W, C)
        target_size: Target (height, width)
        keep_aspect_ratio: Whether to keep aspect ratio
        padding_value: Padding value
    
    Returns:
        resized_image, scale, (padding_top, padding_left)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    if keep_aspect_ratio:
        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), padding_value, dtype=np.uint8)
        
        # Calculate padding
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        
        return padded, scale, (pad_top, pad_left)
    else:
        # Simple resize
        resized = cv2.resize(image, (target_w, target_h))
        return resized, 1.0, (0, 0)


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to YCbCr color space"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Using OpenCV conversion
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        # Reorder to YCbCr (OpenCV uses YCrCb)
        ycbcr[:, :, [1, 2]] = ycbcr[:, :, [2, 1]]
        return ycbcr
    elif len(image.shape) == 4:  # Batch of images
        ycbcr_batch = []
        for img in image:
            ycbcr_batch.append(rgb_to_ycbcr(img))
        return np.stack(ycbcr_batch, axis=0)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert YCbCr image to RGB color space"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Reorder to YCrCb for OpenCV
        ycrcb = image.copy()
        ycrcb[:, :, [1, 2]] = ycrcb[:, :, [2, 1]]
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        return rgb
    elif len(image.shape) == 4:  # Batch of images
        rgb_batch = []
        for img in image:
            rgb_batch.append(ycbcr_to_rgb(img))
        return np.stack(rgb_batch, axis=0)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def extract_y_channel(image: np.ndarray) -> np.ndarray:
    """Extract Y (luminance) channel from YCbCr image"""
    if len(image.shape) == 3:
        return image[:, :, 0:1]  # Keep channel dimension
    elif len(image.shape) == 4:
        return image[:, :, :, 0:1]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def merge_y_channel(y_channel: np.ndarray, cbcr_channels: np.ndarray) -> np.ndarray:
    """Merge Y channel with CbCr channels"""
    if len(y_channel.shape) == 3:
        return np.concatenate([y_channel, cbcr_channels], axis=2)
    elif len(y_channel.shape) == 4:
        return np.concatenate([y_channel, cbcr_channels], axis=3)
    else:
        raise ValueError(f"Unsupported image shape: {y_channel.shape}")


def compute_edge_map(
    image: np.ndarray,
    method: str = "canny",
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """
    Compute edge map for edge-aware loss
    
    Args:
        image: Input image (grayscale or RGB)
        method: 'canny' or 'sobel'
        low_threshold: Canny low threshold
        high_threshold: Canny high threshold
    
    Returns:
        Edge map
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image
    
    if method == "canny":
        edges = cv2.Canny(gray, low_threshold, high_threshold)
    elif method == "sobel":
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = (edges / edges.max() * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    return edges


def apply_retinex_decomposition(
    image: np.ndarray,
    sigma_list: List[float] = [15, 80, 250],
    gain: float = 128,
    offset: float = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Retinex decomposition to separate illumination and reflectance
    
    Args:
        image: Input image (RGB)
        sigma_list: Gaussian kernel sigmas for multi-scale Retinex
        gain: Gain for reflectance computation
        offset: Offset for reflectance computation
    
    Returns:
        reflectance, illumination
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image_float = image.astype(np.float32) + 1e-6
    
    # Initialize illumination and reflectance
    illumination = np.zeros_like(image_float)
    reflectance = np.zeros_like(image_float)
    
    # Multi-scale Retinex
    for sigma in sigma_list:
        # Gaussian blur for illumination estimation
        kernel_size = int(2 * np.ceil(2 * sigma) + 1)
        illumination_sigma = cv2.GaussianBlur(
            image_float,
            (kernel_size, kernel_size),
            sigma
        )
        
        # Compute reflectance for this scale
        reflectance_sigma = np.log(image_float) - np.log(illumination_sigma)
        
        # Accumulate
        illumination += illumination_sigma
        reflectance += reflectance_sigma
    
    # Average
    illumination /= len(sigma_list)
    reflectance /= len(sigma_list)
    
    # Normalize reflectance
    reflectance = gain * reflectance + offset
    reflectance = np.clip(reflectance, 0, 255).astype(np.uint8)
    
    # Normalize illumination
    illumination = np.clip(illumination, 0, 255).astype(np.uint8)
    
    return reflectance, illumination


def compute_color_histogram(
    image: np.ndarray,
    bins: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Compute color histogram for image analysis"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
        
        if normalize:
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_b = cv2.normalize(hist_b, hist_b).flatten()
        
        hist = np.concatenate([hist_r, hist_g, hist_b])
    else:
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        if normalize:
            hist = cv2.normalize(hist, hist).flatten()
    
    return hist


def random_crop_with_objects(
    image: np.ndarray,
    bboxes: List[List[float]],
    labels: List[int],
    min_object_size: float = 0.1,
    max_tries: int = 50,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Random crop that ensures objects are included
    
    Args:
        image: Input image
        bboxes: Bounding boxes in [x1, y1, x2, y2] format
        labels: Object labels
        min_object_size: Minimum fraction of object that must be visible
        max_tries: Maximum number of tries
    
    Returns:
        Cropped image, adjusted bboxes, adjusted labels
    """
    h, w = image.shape[:2]
    
    for _ in range(max_tries):
        # Random crop size
        crop_h = random.randint(int(h * 0.5), int(h * 0.9))
        crop_w = random.randint(int(w * 0.5), int(w * 0.9))
        
        # Random crop position
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # Check if enough objects are visible
        visible_bboxes = []
        visible_labels = []
        
        for bbox, label in zip(bboxes, labels):
            bx1, by1, bx2, by2 = bbox
            
            # Compute intersection
            ix1 = max(bx1, x1)
            iy1 = max(by1, y1)
            ix2 = min(bx2, x2)
            iy2 = min(by2, y2)
            
            if ix1 < ix2 and iy1 < iy2:
                # Object is at least partially visible
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                original_area = (bx2 - bx1) * (by2 - by1)
                
                if intersection_area / original_area >= min_object_size:
                    # Adjust bbox coordinates
                    adj_bx1 = ix1 - x1
                    adj_by1 = iy1 - y1
                    adj_bx2 = ix2 - x1
                    adj_by2 = iy2 - y1
                    
                    visible_bboxes.append([adj_bx1, adj_by1, adj_bx2, adj_by2])
                    visible_labels.append(label)
        
        if len(visible_bboxes) >= max(1, len(bboxes) * 0.3):
            # Good crop found
            cropped_image = image[y1:y2, x1:x2]
            return cropped_image, visible_bboxes, visible_labels
    
    # If no good crop found, return original
    return image, bboxes, labels


class ImagePreprocessor:
    """Preprocess images for DRSQ-Net"""
    
    def __init__(
        self,
        img_size: int = 640,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
        keep_aspect_ratio: bool = True,
    ):
        self.img_size = img_size
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self.keep_aspect_ratio = keep_aspect_ratio
    
    def __call__(
        self,
        image: Union[np.ndarray, List[np.ndarray]],
        return_tensor: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Preprocess image(s)
        
        Args:
            image: Input image or list of images
            return_tensor: Whether to return torch tensor
        
        Returns:
            Preprocessed image(s)
        """
        single_image = False
        if isinstance(image, np.ndarray) and len(image.shape) == 3:
            single_image = True
            image = [image]
        
        processed_images = []
        infos = []
        
        for img in image:
            # Resize with padding
            resized, scale, (pad_top, pad_left) = resize_with_padding(
                img,
                (self.img_size, self.img_size),
                self.keep_aspect_ratio,
                padding_value=114,
            )
            
            # Convert to float and normalize
            normalized = normalize_image(resized, self.normalize_mean, self.normalize_std)
            
            # Convert to tensor if requested
            if return_tensor:
                # Convert to CHW format
                normalized = normalized.transpose(2, 0, 1)
                tensor = torch.from_numpy(normalized).float()
                processed_images.append(tensor)
            else:
                processed_images.append(normalized)
            
            # Store resize info
            infos.append({
                "scale": scale,
                "pad_top": pad_top,
                "pad_left": pad_left,
                "original_size": img.shape[:2],
                "padded_size": resized.shape[:2],
            })
        
        if return_tensor:
            if single_image:
                return processed_images[0], infos[0]
            else:
                return torch.stack(processed_images, dim=0), infos
        else:
            if single_image:
                return processed_images[0], infos[0]
            else:
                return processed_images, infos


class ImagePostprocessor:
    """Postprocess model outputs"""
    
    def __init__(self, conf_threshold: float = 0.5, nms_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
    
    def rescale_boxes(
        self,
        boxes: torch.Tensor,
        image_info: Dict,
    ) -> torch.Tensor:
        """Rescale boxes from padded image to original image coordinates"""
        scale = image_info["scale"]
        pad_top = image_info["pad_top"]
        pad_left = image_info["pad_left"]
        orig_h, orig_w = image_info["original_size"]
        padded_h, padded_w = image_info["padded_size"]
        
        # Remove padding
        boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
        boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
        boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
        boxes[:, 3] = (boxes[:, 3] - pad_top) / scale
        
        # Clip to image boundaries
        boxes[:, 0] = boxes[:, 0].clamp(0, orig_w)
        boxes[:, 1] = boxes[:, 1].clamp(0, orig_h)
        boxes[:, 2] = boxes[:, 2].clamp(0, orig_w)
        boxes[:, 3] = boxes[:, 3].clamp(0, orig_h)
        
        return boxes
    
    def filter_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        image_info: Dict = None,
        apply_nms: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Filter predictions by confidence and apply NMS
        
        Args:
            predictions: Model predictions
            image_info: Image resize information
            apply_nms: Whether to apply NMS
        
        Returns:
            Filtered predictions
        """
        boxes = predictions["boxes"]
        scores = predictions["scores"]
        labels = predictions["labels"]
        
        # Filter by confidence
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        if len(boxes) == 0:
            return []
        
        # Rescale boxes if image info provided
        if image_info is not None:
            boxes = self.rescale_boxes(boxes, image_info)
        
        # Apply NMS
        if apply_nms and len(boxes) > 1:
            keep = torchvision.ops.nms(boxes, scores, self.nms_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        
        # Sort by score
        if len(scores) > 0:
            sorted_idx = torch.argsort(scores, descending=True)
            boxes = boxes[sorted_idx]
            scores = scores[sorted_idx]
            labels = labels[sorted_idx]
        
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }