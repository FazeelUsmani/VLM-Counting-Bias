"""
Image Processing Utilities for VLM Counting Research

This module provides utilities for image preprocessing, augmentation,
and analysis to support vision-language model evaluation.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import base64
import io
from typing import Tuple, List, Dict, Optional, Union, Any
import math


def preprocess_image(image: Union[Image.Image, str, bytes], 
                    target_size: Optional[Tuple[int, int]] = None,
                    normalize: bool = True,
                    quality_enhancement: bool = False) -> Image.Image:
    """Preprocess image for VLM analysis.
    
    Args:
        image: PIL Image, file path, or bytes
        target_size: Target size (width, height) for resizing
        normalize: Whether to normalize image quality
        quality_enhancement: Whether to apply quality enhancement
        
    Returns:
        Preprocessed PIL Image
    """
    # Convert input to PIL Image
    if isinstance(image, str):
        pil_image = Image.open(image)
    elif isinstance(image, bytes):
        pil_image = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil_image = image.copy()
    else:
        raise ValueError("Unsupported image type")
    
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Resize if target size specified
    if target_size:
        pil_image = resize_image_smart(pil_image, target_size)
    
    # Quality enhancement
    if quality_enhancement:
        pil_image = enhance_image_quality(pil_image)
    
    # Normalization
    if normalize:
        pil_image = normalize_image_lighting(pil_image)
    
    return pil_image


def resize_image_smart(image: Image.Image, target_size: Tuple[int, int], 
                      maintain_aspect_ratio: bool = True) -> Image.Image:
    """Intelligently resize image while maintaining quality.
    
    Args:
        image: Input PIL Image
        target_size: Target (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    if maintain_aspect_ratio:
        # Calculate scaling factor to fit within target size
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize with high-quality resampling
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste resized image centered
        final_image = Image.new('RGB', target_size, color=(255, 255, 255))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        final_image.paste(resized, (paste_x, paste_y))
        
        return final_image
    else:
        # Direct resize to target size
        return image.resize(target_size, Image.Resampling.LANCZOS)


def enhance_image_quality(image: Image.Image, 
                         enhance_contrast: bool = True,
                         enhance_sharpness: bool = True,
                         enhance_color: bool = True) -> Image.Image:
    """Enhance image quality for better VLM analysis.
    
    Args:
        image: Input PIL Image
        enhance_contrast: Whether to enhance contrast
        enhance_sharpness: Whether to enhance sharpness
        enhance_color: Whether to enhance color saturation
        
    Returns:
        Enhanced PIL Image
    """
    enhanced = image.copy()
    
    if enhance_contrast:
        # Moderate contrast enhancement
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.1)
    
    if enhance_sharpness:
        # Slight sharpness enhancement
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)
    
    if enhance_color:
        # Moderate color enhancement
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.05)
    
    return enhanced


def normalize_image_lighting(image: Image.Image, method: str = 'histogram') -> Image.Image:
    """Normalize image lighting to improve consistency.
    
    Args:
        image: Input PIL Image
        method: Normalization method ('histogram', 'adaptive', 'gamma')
        
    Returns:
        Normalized PIL Image
    """
    if method == 'histogram':
        # Histogram equalization
        return ImageOps.equalize(image)
    
    elif method == 'adaptive':
        # Adaptive histogram equalization using OpenCV
        img_array = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(normalized)
    
    elif method == 'gamma':
        # Gamma correction
        gamma = 1.2
        img_array = np.array(image)
        
        # Normalize to 0-1 range
        normalized = img_array.astype(np.float32) / 255.0
        
        # Apply gamma correction
        corrected = np.power(normalized, 1/gamma)
        
        # Convert back to 0-255 range
        result = (corrected * 255).astype(np.uint8)
        return Image.fromarray(result)
    
    else:
        return image


def image_to_base64(image: Image.Image, format: str = 'PNG', quality: int = 95) -> str:
    """Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image
        format: Image format ('PNG', 'JPEG')
        quality: JPEG quality (1-100)
        
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    
    if format.upper() == 'JPEG':
        image.save(buffer, format=format, quality=quality, optimize=True)
    else:
        image.save(buffer, format=format, optimize=True)
    
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image
    """
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


def analyze_image_properties(image: Image.Image) -> Dict[str, Any]:
    """Analyze image properties relevant to object counting.
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with image analysis results
    """
    img_array = np.array(image)
    
    # Basic properties
    height, width = img_array.shape[:2]
    aspect_ratio = width / height
    
    # Color analysis
    mean_rgb = np.mean(img_array, axis=(0, 1))
    std_rgb = np.std(img_array, axis=(0, 1))
    
    # Brightness analysis
    brightness = np.mean(img_array)
    
    # Contrast analysis (standard deviation of grayscale)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    contrast = np.std(gray)
    
    # Edge density (measure of detail/complexity)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (width * height)
    
    # Color diversity (number of unique colors)
    unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
    color_diversity = unique_colors / (width * height)
    
    # Noise estimation (high-frequency content)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_estimate = np.var(laplacian)
    
    return {
        'dimensions': {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'total_pixels': width * height
        },
        'color_properties': {
            'mean_rgb': mean_rgb.tolist(),
            'std_rgb': std_rgb.tolist(),
            'brightness': float(brightness),
            'color_diversity': float(color_diversity)
        },
        'quality_metrics': {
            'contrast': float(contrast),
            'edge_density': float(edge_density),
            'noise_estimate': float(noise_estimate)
        },
        'complexity_indicators': {
            'high_contrast': contrast > 50,
            'high_detail': edge_density > 0.1,
            'low_noise': noise_estimate < 100
        }
    }


def detect_potential_occlusion(image: Image.Image, threshold: float = 0.1) -> Dict[str, Any]:
    """Detect potential occlusion patterns in image.
    
    Args:
        image: PIL Image
        threshold: Threshold for dark region detection
        
    Returns:
        Dictionary with occlusion analysis
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect very dark regions (potential occlusion)
    dark_mask = gray < (threshold * 255)
    dark_ratio = np.sum(dark_mask) / dark_mask.size
    
    # Find connected components of dark regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_mask.astype(np.uint8), connectivity=8
    )
    
    # Analyze dark regions
    dark_regions = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_TOP+2], stats[i, cv2.CC_STAT_WIDTH:cv2.CC_STAT_HEIGHT+1]
        
        dark_regions.append({
            'area': int(area),
            'bbox': (int(x), int(y), int(w), int(h)),
            'aspect_ratio': w / h if h > 0 else 0,
            'center': (float(centroids[i][0]), float(centroids[i][1]))
        })
    
    # Sort by area (largest first)
    dark_regions.sort(key=lambda x: x['area'], reverse=True)
    
    return {
        'dark_ratio': float(dark_ratio),
        'num_dark_regions': len(dark_regions),
        'largest_dark_regions': dark_regions[:5],  # Top 5
        'potential_occlusion': dark_ratio > 0.05,
        'occlusion_severity': 'high' if dark_ratio > 0.3 else 'medium' if dark_ratio > 0.1 else 'low'
    }


def segment_image_regions(image: Image.Image, method: str = 'kmeans', n_clusters: int = 5) -> Dict[str, Any]:
    """Segment image into regions for analysis.
    
    Args:
        image: PIL Image
        method: Segmentation method ('kmeans', 'watershed', 'threshold')
        n_clusters: Number of clusters for k-means
        
    Returns:
        Dictionary with segmentation results
    """
    img_array = np.array(image)
    
    if method == 'kmeans':
        # K-means color clustering
        data = img_array.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to image
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(img_array.shape)
        
        # Analyze clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_info = []
        
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            cluster_info.append({
                'cluster_id': int(label),
                'pixel_count': int(count),
                'percentage': float(count / len(labels) * 100),
                'center_color': centers[label].tolist()
            })
        
        return {
            'method': 'kmeans',
            'segmented_image': Image.fromarray(segmented_image),
            'n_clusters': n_clusters,
            'cluster_info': cluster_info
        }
    
    elif method == 'threshold':
        # Simple threshold-based segmentation
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Otsu's thresholding
        thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate region statistics
        foreground_pixels = np.sum(binary == 255)
        background_pixels = np.sum(binary == 0)
        total_pixels = binary.size
        
        return {
            'method': 'threshold',
            'segmented_image': Image.fromarray(binary),
            'threshold_value': float(thresh_val),
            'foreground_ratio': float(foreground_pixels / total_pixels),
            'background_ratio': float(background_pixels / total_pixels)
        }
    
    else:
        raise ValueError(f"Unsupported segmentation method: {method}")


def calculate_image_similarity(image1: Image.Image, image2: Image.Image, 
                             method: str = 'ssim') -> float:
    """Calculate similarity between two images.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        method: Similarity method ('ssim', 'mse', 'histogram')
        
    Returns:
        Similarity score (higher = more similar)
    """
    # Ensure images are same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
    
    img1_array = np.array(image1)
    img2_array = np.array(image2)
    
    if method == 'ssim':
        # Structural Similarity Index
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)
        
        similarity = ssim(gray1, gray2)
        return float(similarity)
    
    elif method == 'mse':
        # Mean Squared Error (converted to similarity)
        mse = np.mean((img1_array - img2_array) ** 2)
        # Convert MSE to similarity (0-1 scale)
        max_mse = 255 ** 2  # Maximum possible MSE for 8-bit images
        similarity = 1.0 - (mse / max_mse)
        return float(similarity)
    
    elif method == 'histogram':
        # Histogram correlation
        hist1 = cv2.calcHist([img1_array], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_array], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return float(correlation)
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


def create_image_grid(images: List[Image.Image], 
                     grid_size: Optional[Tuple[int, int]] = None,
                     image_size: Tuple[int, int] = (200, 200),
                     padding: int = 10,
                     background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Create a grid of images for visualization.
    
    Args:
        images: List of PIL Images
        grid_size: Grid dimensions (cols, rows). If None, auto-calculate
        image_size: Size to resize each image
        padding: Padding between images
        background_color: Background color
        
    Returns:
        Grid image as PIL Image
    """
    if not images:
        raise ValueError("No images provided")
    
    n_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = math.ceil(math.sqrt(n_images))
        rows = math.ceil(n_images / cols)
    else:
        cols, rows = grid_size
    
    # Calculate grid dimensions
    grid_width = cols * image_size[0] + (cols + 1) * padding
    grid_height = rows * image_size[1] + (rows + 1) * padding
    
    # Create grid image
    grid_image = Image.new('RGB', (grid_width, grid_height), background_color)
    
    # Place images
    for idx, img in enumerate(images[:cols * rows]):
        row = idx // cols
        col = idx % cols
        
        # Resize image
        resized_img = img.resize(image_size, Image.Resampling.LANCZOS)
        
        # Calculate position
        x = padding + col * (image_size[0] + padding)
        y = padding + row * (image_size[1] + padding)
        
        # Paste image
        grid_image.paste(resized_img, (x, y))
    
    return grid_image


if __name__ == "__main__":
    # Example usage and testing
    print("Testing image processing utilities...")
    
    # Create a test image
    test_image = Image.new('RGB', (400, 300), color='lightblue')
    
    # Test preprocessing
    processed = preprocess_image(test_image, target_size=(512, 512))
    print(f"Original size: {test_image.size}, Processed size: {processed.size}")
    
    # Test image analysis
    properties = analyze_image_properties(processed)
    print(f"Image properties: {properties['dimensions']}")
    
    # Test base64 conversion
    base64_str = image_to_base64(processed)
    print(f"Base64 length: {len(base64_str)} characters")
    
    # Test conversion back
    restored = base64_to_image(base64_str)
    print(f"Restored image size: {restored.size}")
    
    print("Image processing utilities test completed!")
