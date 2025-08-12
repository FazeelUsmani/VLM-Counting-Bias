"""
Synthetic Data Generator for VLM Counting Bias Research

This module generates synthetic images with controlled object counts and occlusion patterns
for systematic evaluation of vision-language model counting capabilities.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from typing import List, Dict, Tuple, Optional, Any
import math
import colorsys
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ObjectConfig:
    """Configuration for synthetic objects."""
    object_type: str
    min_size: int
    max_size: int
    color_range: List[str]
    shape: str  # 'circle', 'rectangle', 'polygon'
    can_overlap: bool = True


@dataclass
class SyntheticImage:
    """Container for generated synthetic image and metadata."""
    image: Image.Image
    metadata: Dict[str, Any]
    ground_truth: Dict[str, int]
    occlusion_level: float
    objects: List[Dict[str, Any]]


class SyntheticDataGenerator:
    """Generate synthetic images with controlled object counts and occlusion."""
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512), seed: Optional[int] = None):
        """Initialize the synthetic data generator.
        
        Args:
            image_size: Size of generated images (width, height)
            seed: Random seed for reproducibility
        """
        self.image_size = image_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Predefined color palettes
        self.color_palettes = {
            'bright': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta'],
            'pastel': ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightcoral', 'lightsalmon'],
            'dark': ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'darkviolet', 'darkcyan'],
            'natural': ['brown', 'olive', 'tan', 'sienna', 'maroon', 'forestgreen']
        }
        
        # Color name to RGB mapping
        self.color_map = {
            'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
            'yellow': (255, 255, 0), 'purple': (128, 0, 128), 'orange': (255, 165, 0),
            'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'black': (0, 0, 0),
            'white': (255, 255, 255), 'gray': (128, 128, 128),
            'lightblue': (173, 216, 230), 'lightgreen': (144, 238, 144),
            'lightyellow': (255, 255, 224), 'lightpink': (255, 182, 193),
            'lightcoral': (240, 128, 128), 'lightsalmon': (255, 160, 122),
            'darkred': (139, 0, 0), 'darkblue': (0, 0, 139), 'darkgreen': (0, 100, 0),
            'darkorange': (255, 140, 0), 'darkviolet': (148, 0, 211), 'darkcyan': (0, 139, 139),
            'brown': (165, 42, 42), 'olive': (128, 128, 0), 'tan': (210, 180, 140),
            'sienna': (160, 82, 45), 'maroon': (128, 0, 0), 'forestgreen': (34, 139, 34)
        }
        
        # Default object configurations
        self.default_configs = {
            'circles': ObjectConfig(
                object_type='circle',
                min_size=20, max_size=60,
                color_range=self.color_palettes['bright'],
                shape='circle'
            ),
            'rectangles': ObjectConfig(
                object_type='rectangle', 
                min_size=25, max_size=70,
                color_range=self.color_palettes['bright'],
                shape='rectangle'
            ),
            'triangles': ObjectConfig(
                object_type='triangle',
                min_size=25, max_size=65,
                color_range=self.color_palettes['bright'],
                shape='polygon'
            )
        }
    
    def generate_background(self, bg_type: str = 'plain', bg_color: str = 'white') -> Image.Image:
        """Generate background image.
        
        Args:
            bg_type: Type of background ('plain', 'gradient', 'texture', 'noise')
            bg_color: Base background color
            
        Returns:
            PIL Image with background
        """
        width, height = self.image_size
        
        if bg_type == 'plain':
            color = self.color_map.get(bg_color, (255, 255, 255))
            image = Image.new('RGB', self.image_size, color)
            
        elif bg_type == 'gradient':
            image = Image.new('RGB', self.image_size)
            draw = ImageDraw.Draw(image)
            
            # Create vertical gradient
            base_color = self.color_map.get(bg_color, (255, 255, 255))
            for y in range(height):
                # Vary brightness
                factor = 0.7 + 0.3 * (y / height)
                color = tuple(int(c * factor) for c in base_color)
                draw.line([(0, y), (width, y)], fill=color)
                
        elif bg_type == 'texture':
            # Simple texture pattern
            image = Image.new('RGB', self.image_size, self.color_map.get(bg_color, (255, 255, 255)))
            draw = ImageDraw.Draw(image)
            
            # Add random dots for texture
            for _ in range(width * height // 100):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                brightness = random.randint(0, 50)
                base_color = self.color_map.get(bg_color, (255, 255, 255))
                color = tuple(max(0, min(255, c + brightness - 25)) for c in base_color)
                draw.point((x, y), fill=color)
                
        elif bg_type == 'noise':
            # Generate noise background
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            base_color = np.array(self.color_map.get(bg_color, (255, 255, 255)))
            noise_image = noise + base_color
            noise_image = np.clip(noise_image, 0, 255)
            image = Image.fromarray(noise_image.astype(np.uint8))
            
        else:
            # Default to plain
            image = Image.new('RGB', self.image_size, self.color_map.get(bg_color, (255, 255, 255)))
        
        return image
    
    def generate_object_positions(self, num_objects: int, object_sizes: List[Tuple[int, int]], 
                                 avoid_overlap: bool = False) -> List[Tuple[int, int]]:
        """Generate non-overlapping positions for objects.
        
        Args:
            num_objects: Number of objects to position
            object_sizes: List of (width, height) for each object
            avoid_overlap: Whether to avoid overlaps
            
        Returns:
            List of (x, y) positions
        """
        width, height = self.image_size
        positions = []
        
        for i in range(num_objects):
            obj_w, obj_h = object_sizes[i]
            max_attempts = 100
            
            # Ensure object fits within image bounds
            min_x = max(obj_w // 2, 1)
            max_x = max(width - obj_w // 2, min_x + 1)
            min_y = max(obj_h // 2, 1)
            max_y = max(height - obj_h // 2, min_y + 1)
            
            # If object is too large for the image, resize it
            if min_x >= max_x or min_y >= max_y:
                # Scale down object size to fit
                scale_x = min(0.8, (width * 0.6) / obj_w)
                scale_y = min(0.8, (height * 0.6) / obj_h)
                scale = min(scale_x, scale_y)
                obj_w = max(5, int(obj_w * scale))
                obj_h = max(5, int(obj_h * scale))
                
                min_x = max(obj_w // 2, 1)
                max_x = max(width - obj_w // 2, min_x + 1)
                min_y = max(obj_h // 2, 1)
                max_y = max(height - obj_h // 2, min_y + 1)
            
            for attempt in range(max_attempts):
                x = random.randint(min_x, max_x - 1)
                y = random.randint(min_y, max_y - 1)
                
                if not avoid_overlap:
                    positions.append((x, y))
                    break
                
                # Check for overlaps with existing objects
                overlap = False
                for j, (prev_x, prev_y) in enumerate(positions):
                    prev_w, prev_h = object_sizes[j]
                    
                    if (abs(x - prev_x) < (obj_w + prev_w) // 2 and 
                        abs(y - prev_y) < (obj_h + prev_h) // 2):
                        overlap = True
                        break
                
                if not overlap:
                    positions.append((x, y))
                    break
            else:
                # If we can't find a non-overlapping position, place randomly
                x = random.randint(min_x, max_x - 1)
                y = random.randint(min_y, max_y - 1)
                positions.append((x, y))
        
        return positions
    
    def draw_circle(self, draw: ImageDraw.Draw, x: int, y: int, radius: int, color: Tuple[int, int, int]) -> Dict[str, Any]:
        """Draw a circle and return its metadata."""
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        
        return {
            'shape': 'circle',
            'center': (x, y),
            'radius': radius,
            'color': color,
            'bbox': (x - radius, y - radius, x + radius, y + radius)
        }
    
    def draw_rectangle(self, draw: ImageDraw.Draw, x: int, y: int, width: int, height: int, 
                      color: Tuple[int, int, int]) -> Dict[str, Any]:
        """Draw a rectangle and return its metadata."""
        x1, y1 = x - width // 2, y - height // 2
        x2, y2 = x + width // 2, y + height // 2
        draw.rectangle([x1, y1, x2, y2], fill=color)
        
        return {
            'shape': 'rectangle',
            'center': (x, y),
            'width': width,
            'height': height,
            'color': color,
            'bbox': (x1, y1, x2, y2)
        }
    
    def draw_triangle(self, draw: ImageDraw.Draw, x: int, y: int, size: int, 
                     color: Tuple[int, int, int]) -> Dict[str, Any]:
        """Draw a triangle and return its metadata."""
        # Equilateral triangle
        height = int(size * math.sqrt(3) / 2)
        points = [
            (x, y - height // 2),  # Top
            (x - size // 2, y + height // 2),  # Bottom left
            (x + size // 2, y + height // 2)   # Bottom right
        ]
        draw.polygon(points, fill=color)
        
        return {
            'shape': 'triangle',
            'center': (x, y),
            'size': size,
            'color': color,
            'points': points,
            'bbox': (x - size // 2, y - height // 2, x + size // 2, y + height // 2)
        }
    
    def generate_objects(self, image: Image.Image, object_config: ObjectConfig, 
                        num_objects: int) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Generate objects on the image.
        
        Args:
            image: Base image to draw on
            object_config: Configuration for objects
            num_objects: Number of objects to generate
            
        Returns:
            Tuple of (modified image, list of object metadata)
        """
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        objects = []
        
        # Generate object sizes
        object_sizes = []
        for _ in range(num_objects):
            if object_config.shape == 'circle':
                radius = random.randint(object_config.min_size, object_config.max_size)
                object_sizes.append((radius * 2, radius * 2))
            elif object_config.shape == 'rectangle':
                width = random.randint(object_config.min_size, object_config.max_size)
                height = random.randint(object_config.min_size, object_config.max_size)
                object_sizes.append((width, height))
            else:  # triangle
                size = random.randint(object_config.min_size, object_config.max_size)
                object_sizes.append((size, size))
        
        # Generate positions
        positions = self.generate_object_positions(num_objects, object_sizes, 
                                                  not object_config.can_overlap)
        
        # Draw objects
        for i, (x, y) in enumerate(positions):
            color_name = random.choice(object_config.color_range)
            color = self.color_map.get(color_name, (255, 0, 0))
            
            if object_config.shape == 'circle':
                radius = object_sizes[i][0] // 2
                obj_metadata = self.draw_circle(draw, x, y, radius, color)
            elif object_config.shape == 'rectangle':
                width, height = object_sizes[i]
                obj_metadata = self.draw_rectangle(draw, x, y, width, height, color)
            else:  # triangle
                size = object_sizes[i][0]
                obj_metadata = self.draw_triangle(draw, x, y, size, color)
            
            obj_metadata.update({
                'id': i,
                'color_name': color_name,
                'position': (x, y),
                'visible': True
            })
            objects.append(obj_metadata)
        
        return image_copy, objects
    
    def apply_occlusion(self, image: Image.Image, objects: List[Dict[str, Any]], 
                       occlusion_level: float, occlusion_type: str = 'random_patches') -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Apply occlusion to the image.
        
        Args:
            image: Image with objects
            objects: List of object metadata
            occlusion_level: Fraction of image to occlude (0.0 to 1.0)
            occlusion_type: Type of occlusion ('random_patches', 'strips', 'blocks')
            
        Returns:
            Tuple of (occluded image, updated object metadata)
        """
        if occlusion_level <= 0:
            return image, objects
        
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        width, height = self.image_size
        
        total_area = width * height
        target_occlusion_area = total_area * occlusion_level
        current_occlusion_area = 0
        
        updated_objects = [obj.copy() for obj in objects]
        
        if occlusion_type == 'random_patches':
            # Generate random rectangular patches
            while current_occlusion_area < target_occlusion_area:
                patch_w = random.randint(30, min(150, width // 3))
                patch_h = random.randint(30, min(150, height // 3))
                patch_x = random.randint(0, width - patch_w)
                patch_y = random.randint(0, height - patch_h)
                
                # Draw black occlusion patch
                draw.rectangle([patch_x, patch_y, patch_x + patch_w, patch_y + patch_h], 
                              fill=(0, 0, 0))
                
                current_occlusion_area += patch_w * patch_h
                
                # Update object visibility
                patch_bbox = (patch_x, patch_y, patch_x + patch_w, patch_y + patch_h)
                for obj in updated_objects:
                    if self._bbox_overlap(obj['bbox'], patch_bbox):
                        overlap_area = self._bbox_intersection_area(obj['bbox'], patch_bbox)
                        obj_area = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
                        
                        # Mark as partially occluded if significant overlap
                        if overlap_area > obj_area * 0.3:
                            obj['visible'] = False
                        elif overlap_area > obj_area * 0.1:
                            obj['partially_occluded'] = True
        
        elif occlusion_type == 'strips':
            # Horizontal or vertical strips
            strip_horizontal = random.choice([True, False])
            strip_width = int(target_occlusion_area / (height if strip_horizontal else width))
            
            if strip_horizontal:
                # Horizontal strips
                num_strips = max(1, int(occlusion_level * 10))
                strip_height = strip_width // num_strips
                
                for _ in range(num_strips):
                    y = random.randint(0, height - strip_height)
                    draw.rectangle([0, y, width, y + strip_height], fill=(0, 0, 0))
                    current_occlusion_area += width * strip_height
                    
                    # Update visibility
                    strip_bbox = (0, y, width, y + strip_height)
                    for obj in updated_objects:
                        if self._bbox_overlap(obj['bbox'], strip_bbox):
                            obj['visible'] = False
            else:
                # Vertical strips
                num_strips = max(1, int(occlusion_level * 10))
                strip_width_actual = strip_width // num_strips
                
                for _ in range(num_strips):
                    x = random.randint(0, width - strip_width_actual)
                    draw.rectangle([x, 0, x + strip_width_actual, height], fill=(0, 0, 0))
                    current_occlusion_area += strip_width_actual * height
                    
                    # Update visibility
                    strip_bbox = (x, 0, x + strip_width_actual, height)
                    for obj in updated_objects:
                        if self._bbox_overlap(obj['bbox'], strip_bbox):
                            obj['visible'] = False
        
        elif occlusion_type == 'blocks':
            # Large rectangular blocks
            num_blocks = max(1, int(occlusion_level * 5))
            block_area = target_occlusion_area / num_blocks
            
            for _ in range(num_blocks):
                block_size = int(math.sqrt(block_area))
                block_w = random.randint(block_size // 2, min(block_size * 2, width // 2))
                block_h = int(block_area / block_w)
                
                if block_h > height // 2:
                    block_h = height // 2
                    block_w = int(block_area / block_h)
                
                block_x = random.randint(0, width - block_w)
                block_y = random.randint(0, height - block_h)
                
                draw.rectangle([block_x, block_y, block_x + block_w, block_y + block_h], 
                              fill=(0, 0, 0))
                
                current_occlusion_area += block_w * block_h
                
                # Update visibility
                block_bbox = (block_x, block_y, block_x + block_w, block_y + block_h)
                for obj in updated_objects:
                    if self._bbox_overlap(obj['bbox'], block_bbox):
                        overlap_area = self._bbox_intersection_area(obj['bbox'], block_bbox)
                        obj_area = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
                        
                        if overlap_area > obj_area * 0.5:
                            obj['visible'] = False
                        elif overlap_area > obj_area * 0.2:
                            obj['partially_occluded'] = True
        
        return image_copy, updated_objects
    
    def _bbox_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)
    
    def _bbox_intersection_area(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> int:
        """Calculate intersection area of two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_min >= x_max or y_min >= y_max:
            return 0
        
        return (x_max - x_min) * (y_max - y_min)
    
    def generate_dataset(self, 
                        object_configs: List[ObjectConfig],
                        object_counts: List[int],
                        occlusion_levels: List[float],
                        num_images_per_condition: int,
                        background_types: List[str] = None,
                        occlusion_types: List[str] = None) -> List[SyntheticImage]:
        """Generate a complete synthetic dataset.
        
        Args:
            object_configs: List of object configurations to use
            object_counts: List of object counts to test
            occlusion_levels: List of occlusion levels (0.0 to 1.0)
            num_images_per_condition: Number of images per condition
            background_types: Types of backgrounds to use
            occlusion_types: Types of occlusion to apply
            
        Returns:
            List of SyntheticImage objects
        """
        if background_types is None:
            background_types = ['plain']
        if occlusion_types is None:
            occlusion_types = ['random_patches']
        
        dataset = []
        image_id = 0
        
        total_combinations = (len(object_configs) * len(object_counts) * 
                            len(occlusion_levels) * len(background_types) * 
                            len(occlusion_types) * num_images_per_condition)
        
        print(f"Generating {total_combinations} synthetic images...")
        
        for obj_config in object_configs:
            for count in object_counts:
                for occlusion_level in occlusion_levels:
                    for bg_type in background_types:
                        for occ_type in occlusion_types:
                            for img_idx in range(num_images_per_condition):
                                # Generate background
                                bg_color = 'white' if bg_type == 'plain' else random.choice(['lightgray', 'beige', 'lightblue'])
                                background = self.generate_background(bg_type, bg_color)
                                
                                # Generate objects
                                image_with_objects, objects = self.generate_objects(
                                    background, obj_config, count
                                )
                                
                                # Apply occlusion
                                final_image, final_objects = self.apply_occlusion(
                                    image_with_objects, objects, occlusion_level, occ_type
                                )
                                
                                # Count visible objects
                                visible_count = sum(1 for obj in final_objects if obj.get('visible', True))
                                
                                # Create metadata
                                metadata = {
                                    'image_id': f"synthetic_{image_id:06d}",
                                    'object_type': obj_config.object_type,
                                    'total_objects': count,
                                    'visible_objects': visible_count,
                                    'occlusion_level': occlusion_level,
                                    'occlusion_type': occ_type,
                                    'background_type': bg_type,
                                    'background_color': bg_color,
                                    'image_size': self.image_size,
                                    'generation_params': {
                                        'min_size': obj_config.min_size,
                                        'max_size': obj_config.max_size,
                                        'color_range': obj_config.color_range,
                                        'shape': obj_config.shape
                                    }
                                }
                                
                                ground_truth = {
                                    'total_count': count,
                                    'visible_count': visible_count,
                                    'occluded_count': count - visible_count
                                }
                                
                                synthetic_image = SyntheticImage(
                                    image=final_image,
                                    metadata=metadata,
                                    ground_truth=ground_truth,
                                    occlusion_level=occlusion_level,
                                    objects=final_objects
                                )
                                
                                dataset.append(synthetic_image)
                                image_id += 1
                                
                                if image_id % 100 == 0:
                                    print(f"Generated {image_id}/{total_combinations} images...")
        
        print(f"Synthetic dataset generation complete: {len(dataset)} images")
        return dataset
    
    def save_dataset(self, dataset: List[SyntheticImage], output_dir: str):
        """Save dataset to disk.
        
        Args:
            dataset: List of synthetic images
            output_dir: Directory to save images and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        metadata_list = []
        
        for synthetic_img in dataset:
            # Save image
            image_filename = f"{synthetic_img.metadata['image_id']}.png"
            image_path = images_dir / image_filename
            synthetic_img.image.save(image_path)
            
            # Collect metadata
            metadata_entry = {
                'filename': image_filename,
                'filepath': str(image_path),
                **synthetic_img.metadata,
                'ground_truth': synthetic_img.ground_truth,
                'num_objects': len(synthetic_img.objects),
                'objects': synthetic_img.objects
            }
            metadata_list.append(metadata_entry)
        
        # Save metadata
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'total_images': len(dataset),
            'object_types': list(set(img.metadata['object_type'] for img in dataset)),
            'occlusion_levels': sorted(list(set(img.occlusion_level for img in dataset))),
            'object_counts': sorted(list(set(img.ground_truth['total_count'] for img in dataset))),
            'background_types': list(set(img.metadata['background_type'] for img in dataset)),
            'occlusion_types': list(set(img.metadata['occlusion_type'] for img in dataset)),
            'image_size': dataset[0].metadata['image_size'] if dataset else None
        }
        
        summary_path = output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Dataset saved to {output_path}")
        print(f"- Images: {images_dir}")
        print(f"- Metadata: {metadata_path}")
        print(f"- Summary: {summary_path}")


def create_default_dataset(output_dir: str = "data/synthetic", 
                          image_size: Tuple[int, int] = (512, 512),
                          seed: int = 42) -> List[SyntheticImage]:
    """Create a default synthetic dataset for VLM counting research.
    
    Args:
        output_dir: Directory to save the dataset
        image_size: Size of generated images
        seed: Random seed for reproducibility
        
    Returns:
        List of generated synthetic images
    """
    generator = SyntheticDataGenerator(image_size=image_size, seed=seed)
    
    # Use default configurations
    object_configs = [
        generator.default_configs['circles'],
        generator.default_configs['rectangles']
    ]
    
    dataset = generator.generate_dataset(
        object_configs=object_configs,
        object_counts=[3, 5, 7, 10],
        occlusion_levels=[0.0, 0.25, 0.50, 0.75],
        num_images_per_condition=5,  # Reduced for faster generation
        background_types=['plain', 'gradient'],
        occlusion_types=['random_patches', 'blocks']
    )
    
    # Save dataset
    generator.save_dataset(dataset, output_dir)
    
    return dataset


if __name__ == "__main__":
    # Example usage
    print("Creating default synthetic dataset...")
    dataset = create_default_dataset()
    print(f"Generated {len(dataset)} synthetic images")
    
    # Display some statistics
    if dataset:
        total_counts = [img.ground_truth['total_count'] for img in dataset]
        visible_counts = [img.ground_truth['visible_count'] for img in dataset]
        occlusion_levels = [img.occlusion_level for img in dataset]
        
        print(f"\nDataset Statistics:")
        print(f"- Object counts: {sorted(set(total_counts))}")
        print(f"- Visible counts range: {min(visible_counts)}-{max(visible_counts)}")
        print(f"- Occlusion levels: {sorted(set(occlusion_levels))}")
        print(f"- Object types: {set(img.metadata['object_type'] for img in dataset)}")
