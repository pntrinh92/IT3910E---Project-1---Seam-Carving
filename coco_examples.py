#!/usr/bin/env python3

import os
import cv2
from seam_carving import ContentAwareImageResizer


def ensure_directories():
    """Create necessary directories."""
    os.makedirs('images/coco', exist_ok=True)
    os.makedirs('output/coco', exist_ok=True)
    os.makedirs('images/masks', exist_ok=True)


def process_single_coco_image(input_path, output_path, target_h, target_w):
    """
    Process a single COCO image.
    """
    print(f"\nProcessing: {input_path}")
    
    # Load image to get dimensions
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not load image")
        return False
    
    orig_h, orig_w = img.shape[:2]
    print(f"  Original size: {orig_w}x{orig_h}")
    print(f"  Target size: {target_w}x{target_h}")
    
    # Validate target size 
    if target_w >= orig_w or target_h >= orig_h:
        print(f"Target size must be smaller than original")
        return False
    
    try:
        # Create resizer and process
        resizer = ContentAwareImageResizer(input_path, target_h, target_w)
        resizer.export_result(output_path)
        
        # Verify result
        result = cv2.imread(output_path)
        result_h, result_w = result.shape[:2]
        print(f"  ✓ Success! Result: {result_w}x{result_h}")
        print(f"  Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def batch_process_coco_images(coco_dir='images/coco/val2017', output_dir='output/coco'):
    print("Batch Processing COCO Images")
    image_files = [f for f in os.listdir(coco_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"\nNo images found in {coco_dir}")
        print(f"  Please add COCO images to this directory first.")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    success_count = 0
    for i, filename in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")
        
        input_path = os.path.join(coco_dir, filename)
        output_path = os.path.join(output_dir, f"resized_{filename}")


        if process_single_coco_image(input_path, output_path, 300, 400):
            success_count += 1
    
    print(f"complete: {success_count}/{len(image_files)} successful")


def demo_coco_scenarios():
    print("Coco Dataset - Seam Carving")
    
    ensure_directories()
    coco_images = [f for f in os.listdir('images/coco/val2017') 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not coco_images:
        print("\nNo COCO images found in images/coco/val2017")
        return
    
    print(f"\nFound {len(coco_images)} COCO images")
    
    # Use first image for demonstration
    test_image = os.path.join('images/coco', coco_images[0])
    
    print("1. Width Reduction")
    img = cv2.imread(test_image)
    h, w = img.shape[:2]
    new_w = int(w * 0.7) 
    process_single_coco_image(test_image, 'output/coco/demo_width_reduced.jpg', h, new_w)
    
    print("2. Height Reduction")
    new_h = int(h * 0.7) 
    process_single_coco_image(test_image, 'output/coco/demo_height_reduced.jpg', new_h, w)
    
    print("\n" + "-" * 70)
    print("3. Both Dimensions Reduced")
    print("-" * 70)
    new_h = int(h * 0.75)
    new_w = int(w * 0.75)
    process_single_coco_image(test_image, 'output/coco/demo_both_reduced.jpg', new_h, new_w)
    
    print("complete")


def create_simple_mask_demo():
    print("creating example mask")
    
    coco_images = [f for f in os.listdir('images/coco/val2017') 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not coco_images:
        print("No COCO images found")
        return
    img_path = os.path.join('images/coco', coco_images[0])
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    import numpy as np
    mask = np.zeros((h, w), dtype=np.uint8)
    
    center_h, center_w = h // 2, w // 2
    top = h // 4
    bottom = 3 * h // 4
    left = w // 4
    right = 3 * w // 4
    
    mask[top:bottom, left:right] = 255
    
    mask_path = 'images/masks/center_protection.jpg'
    cv2.imwrite(mask_path, mask)
    
    print(f"Created protection mask: {mask_path}")
    print(f"  Image size: {w}x{h}")
    print(f"  Protected region: center 50%")
    print("\nYou can now use this mask with:")
    print(f"  python main.py -i {img_path} -o output/protected_result.jpg \\")
    print(f"                 -H 400 -W 600 -p {mask_path}")


if __name__ == "__main__":
    import sys
    
    ensure_directories()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == '--batch':
            batch_process_coco_images()
        elif command == '--demo':
            demo_coco_scenarios()
        elif command == '--mask':
            create_simple_mask_demo()   
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  --batch   : Process all images in images/coco/")
            print("  --demo    : Run demonstration scenarios")
            print("  --mask    : Create example protection mask")
    else:
        print("No command provided. Use --batch, --demo, or --mask.")
