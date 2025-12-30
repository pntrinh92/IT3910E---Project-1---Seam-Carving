

import os
import sys
import cv2  
import numpy as np
import argparse
from pathlib import Path
import time
from typing import List, Tuple, Dict
from seam_carving import seam_carve


INPUT_DIR = "input"
OUTPUT_DIR = "output"
MASK_DIR = "masks"

# Speed up processing by downsizing large images
MAX_IMAGE_WIDTH = 500  # Images wider than this will be resized for faster processing
ENABLE_DOWNSIZE = True  # Set to False to process at full resolution

RESIZE_CONFIGS = [
    {"name": "small", "width_ratio": 0.5, "height_ratio": 0.5},
    {"name": "medium", "width_ratio": 0.7, "height_ratio": 0.7},
    {"name": "large", "width_ratio": 0.8, "height_ratio": 0.8},
]

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']



def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/standard", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/backward_energy", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/with_mask", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/comparison", exist_ok=True)
    print("Output directories created")

def get_all_images(directory: str) -> List[Path]:
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(Path(directory).glob(f"*{ext}"))
        images.extend(Path(directory).glob(f"*{ext.upper()}"))
    return sorted(images)

def get_mask_path(image_path: Path, mask_type: str = "protect") -> Path:
    mask_filename = f"{image_path.stem}_{mask_type}_mask{image_path.suffix}"
    return Path(MASK_DIR) / mask_filename

def print_section(title: str):
    print(f"  {title}")

def print_progress(current: int, total: int, message: str = ""):
    percent = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r[{bar}] {percent:.1f}% {message}", end="", flush=True)
    if current == total:
        print()  # New line when complete

def downsize_if_needed(image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Downsize image if it's too large to speed up processing."""
    h, w = image.shape[:2]
    
    if not ENABLE_DOWNSIZE or w <= MAX_IMAGE_WIDTH:
        return image, mask, 1.0
    
    # Calculate scale factor
    scale = MAX_IMAGE_WIDTH / w
    new_w = MAX_IMAGE_WIDTH
    new_h = int(h * scale)
    
    # Resize image
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Resize mask if present
    resized_mask = None
    if mask is not None:
        resized_mask = cv2.resize(mask, (new_w, new_h))
    
    return resized_image, resized_mask, scale

def apply_seam_carving(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    mask: np.ndarray = None,
    use_forward: bool = True
) -> np.ndarray:
    h, w = image.shape[:2]
    dy = target_height - h
    dx = target_width - w
    
    image_float = image.astype(np.float64)
    mask_float = mask.astype(np.float64) if mask is not None else None
    
    result = seam_carve(image_float, dy, dx, mask_float, vis=False, use_forward=use_forward)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def standard_resize(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

def create_comparison_image(original: np.ndarray, seam_carved: np.ndarray, standard: np.ndarray) -> np.ndarray:
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 255, 0)  # Green
    
    # Original
    orig_labeled = original.copy()
    h, w = orig_labeled.shape[:2]
    cv2.putText(orig_labeled, f'ORIGINAL {w}x{h}', (10, 40), 
                font, font_scale, color, thickness)
    
    # Seam carved
    seam_labeled = seam_carved.copy()
    h, w = seam_labeled.shape[:2]
    cv2.putText(seam_labeled, f'SEAM CARVED {w}x{h}', (10, 40), 
                font, font_scale, (255, 0, 0), thickness)
    
    # Standard resize
    std_labeled = standard.copy()
    h, w = std_labeled.shape[:2]
    cv2.putText(std_labeled, f'STANDARD RESIZE {w}x{h}', (10, 40), 
                font, font_scale, (0, 0, 255), thickness)
    
    # Stack horizontally
    max_height = max(orig_labeled.shape[0], seam_labeled.shape[0], std_labeled.shape[0])
    
    def pad_to_height(img, target_h):
        h, w = img.shape[:2]
        if h < target_h:
            padding = target_h - h
            return np.pad(img, ((0, padding), (0, 0), (0, 0)), mode='constant', constant_values=255)
        return img
    
    orig_labeled = pad_to_height(orig_labeled, max_height)
    seam_labeled = pad_to_height(seam_labeled, max_height)
    std_labeled = pad_to_height(std_labeled, max_height)
    
    comparison = np.hstack([orig_labeled, seam_labeled, std_labeled])
    
    return comparison

def process_single_image(
    image_path: Path,
    config: Dict,
    process_type: str = "standard"
) -> Dict:
    result = {
        'success': False,
        'input_path': str(image_path),
        'output_path': None,
        'processing_time': 0,
        'error': None
    }
    
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            result['error'] = "Could not read image"
            return result
        
        orig_h, orig_w = image.shape[:2]
        
        # Downsize if needed for faster processing
        mask = None
        if process_type == "with_mask":
            mask_path = get_mask_path(image_path)
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), 0)
            else:
                result['error'] = f"Mask not found"
                return result
        
        image, mask, scale = downsize_if_needed(image, mask)
        h, w = image.shape[:2]
        
        if scale < 1.0:
            print(f"\n    Downsized from {orig_w}x{orig_h} to {w}x{h} for faster processing", end="")
        
        target_w = int(w * config['width_ratio'])
        target_h = int(h * config['height_ratio'])
        
        if target_w >= w or target_h >= h:
            result['error'] = "Target dimensions must be smaller"
            return result
        
        use_forward = True
        output_subdir = "standard"
        suffix = ""
        
        if process_type == "backward_energy":
            use_forward = False
            output_subdir = "backward_energy"
            suffix = "_backward"
        elif process_type == "with_mask":
            output_subdir = "with_mask"
            suffix = "_masked"
        
        output_filename = f"{image_path.stem}_{config['name']}{suffix}{image_path.suffix}"
        output_path = Path(OUTPUT_DIR) / output_subdir / output_filename
        
        start_time = time.time()
        result_image = apply_seam_carving(image, target_w, target_h, mask, use_forward)
        processing_time = time.time() - start_time
        
        cv2.imwrite(str(output_path), result_image)
        
        result['success'] = True
        result['output_path'] = str(output_path)
        result['processing_time'] = processing_time
        result['original_size'] = (orig_w, orig_h)
        result['result_size'] = (target_w, target_h)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def process_with_comparison(image_path: Path, config: Dict) -> Dict:
    result = {
        'success': False,
        'input_path': str(image_path),
        'comparison_path': None,
        'error': None
    }
    
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            result['error'] = "Could not read image"
            return result
        
        # Downsize for faster processing
        image, _, scale = downsize_if_needed(image)
        
        h, w = image.shape[:2]
        target_w = int(w * config['width_ratio'])
        target_h = int(h * config['height_ratio'])

        seam_carved = apply_seam_carving(image, target_w, target_h, use_forward=True)
        
        # Apply standard resize
        standard = standard_resize(image, target_w, target_h)
        
        # Create comparison
        comparison = create_comparison_image(image, seam_carved, standard)
        
        # Save comparison
        comparison_filename = f"{image_path.stem}_{config['name']}_comparison{image_path.suffix}"
        comparison_path = Path(OUTPUT_DIR) / "comparison" / comparison_filename
        cv2.imwrite(str(comparison_path), comparison)
        
        result['success'] = True
        result['comparison_path'] = str(comparison_path)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def batch_process_all(
    resize_config: str = "medium",
    process_types: List[str] = None,
    create_comparisons: bool = True
):
    setup_directories()
    
    if resize_config == "all":
        configs = RESIZE_CONFIGS
    else:
        configs = [c for c in RESIZE_CONFIGS if c['name'] == resize_config]
        if not configs:
            print(f"Invalid resize config: {resize_config}")
            return
    
    # Default process types
    if process_types is None:
        process_types = ["standard", "backward_energy", "with_mask"]
    
    # Get all images
    images = get_all_images(INPUT_DIR)
    
    if not images:
        print(f"No images found in {INPUT_DIR}/")
        return
    
    print_section("Batch processing seam carving")
    print(f"Found {len(images)} images in {INPUT_DIR}/")
    print(f"Configurations: {[c['name'] for c in configs]}")
    print(f"Process types: {process_types}")
    print(f"Create comparisons: {create_comparisons}")
    
    # Statistics
    total_tasks = len(images) * len(configs) * len(process_types)
    if create_comparisons:
        total_tasks += len(images) * len(configs)
    
    completed = 0
    successful = 0
    failed = 0
    results_log = []
    
    # Process each image
    total_time = 0
    for img_idx, image_path in enumerate(images, 1):
        print_section(f"Processing Image {img_idx}/{len(images)}: {image_path.name}")
        
        # Read image to show size
        img_test = cv2.imread(str(image_path))
        if img_test is not None:
            h, w = img_test.shape[:2]
            print(f"  Original size: {w}x{h} pixels")
        
        for config in configs:
            print(f"  Configuration: {config['name']} ({config['width_ratio']*100:.0f}%)")
            
            # Process with different methods
            for process_type in process_types:
                completed += 1
                task_start = time.time()
                
                print(f"    [{process_type}] ", end="", flush=True)
                
                result = process_single_image(image_path, config, process_type)
                
                task_time = time.time() - task_start
                total_time += task_time
                
                if result['success']:
                    successful += 1
                    print(f"✓ ({task_time:.1f}s)")
                    results_log.append(f"{result['output_path']}")
                else:
                    failed += 1
                    print(f"✗ {result['error']}")
                    if result['error']:
                        results_log.append(f"{image_path.name} ({process_type}): {result['error']}")
                
                # Show progress
                avg_time = total_time / completed if completed > 0 else 0
                remaining = total_tasks - completed
                est_time = avg_time * remaining
                print(f"    Progress: {completed}/{total_tasks} | Avg: {avg_time:.1f}s/task | Est. remaining: {est_time/60:.1f}min")
            
            # Create comparison if requested
            if create_comparisons:
                completed += 1
                task_start = time.time()
                
                print(f"    [comparison] ", end="", flush=True)
                
                comp_result = process_with_comparison(image_path, config)
                
                task_time = time.time() - task_start
                total_time += task_time
                
                if comp_result['success']:
                    successful += 1
                    print(f"✓ ({task_time:.1f}s)")
                    results_log.append(f"{comp_result['comparison_path']}")
                else:
                    failed += 1
                    print(f"✗ {comp_result['error']}")
                    if comp_result['error']:
                        results_log.append(f"Comparison failed: {comp_result['error']}")
    
    # Summary
    print_section("Processing complete")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    
    # Show directory structure
    print("\nOutput structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── standard/          (Forward energy, no mask)")
    print(f"    ├── backward_energy/   (Backward energy method)")
    print(f"    ├── with_mask/         (With protection masks)")
    print(f"    └── comparison/        (Side-by-side comparisons)")
    
    # Show some results
    if results_log:
        print("\nProcessing log (last 10 entries):")
        for log in results_log[-10:]:
            print(f"  {log}")

def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process images with seam carving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images with medium size (70%)
  python main.py
  
  # Process with small size (50%)
  python main.py --size small
  
  # Process with all sizes
  python main.py --size all
  
  # Process only with standard method (no comparisons)
  python main.py --types standard --no-comparison
  
  # Process only with backward energy
  python main.py --types backward_energy
        """
    )
    
    parser.add_argument(
        '--size',
        choices=['small', 'medium', 'large', 'all'],
        default='medium',
        help='Resize configuration to use (default: medium)'
    )
    
    parser.add_argument(
        '--types',
        nargs='+',
        choices=['standard', 'backward_energy', 'with_mask'],
        default=None,
        help='Processing types to apply (default: all)'
    )
    
    parser.add_argument(
        '--no-comparison',
        action='store_true',
        help='Skip creating comparison images'
    )
    
    parser.add_argument(
        '--input',
        default="input",
        help='Input directory (default: input)'
    )
    
    parser.add_argument(
        '--output',
        default="output",
        help='Output directory (default: output)'
    )
    
    args = parser.parse_args()
    
    # Use command-line arguments for directories
    input_dir = args.input
    output_dir = args.output
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        print(f"Please create it and add some images.")
        sys.exit(1)
    
    # Update global directories for this run
    global INPUT_DIR, OUTPUT_DIR
    INPUT_DIR = input_dir
    OUTPUT_DIR = output_dir
    
    # Run batch processing
    try:
        batch_process_all(
            resize_config=args.size,
            process_types=args.types,
            create_comparisons=not args.no_comparison
        )
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
