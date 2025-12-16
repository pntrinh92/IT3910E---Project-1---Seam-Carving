#!/usr/bin/env python3
"""
Simple test script for seam carving implementation.
Creates a test image and demonstrates basic functionality.
"""

import numpy as np
import cv2
import os


def create_test_image():
    """Create a simple test image with colored stripes."""
    # Create 600x800 image
    height, width = 600, 800
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add colored vertical stripes
    stripe_width = width // 5
    colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
    ]
    
    for i, color in enumerate(colors):
        start_x = i * stripe_width
        end_x = (i + 1) * stripe_width if i < 4 else width
        img[:, start_x:end_x] = color
    
    # Add a white circle in the center (important object)
    center_x, center_y = width // 2, height // 2
    cv2.circle(img, (center_x, center_y), 100, (255, 255, 255), -1)
    cv2.putText(img, 'IMPORTANT', (center_x - 80, center_y + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img


def create_protection_mask(img_shape):
    """Create a protection mask for the center circle."""
    height, width = img_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Protect the center circle
    center_x, center_y = width // 2, height // 2
    cv2.circle(mask, (center_x, center_y), 120, 255, -1)
    
    return mask


def create_removal_mask(img_shape):
    """Create a removal mask for the center circle."""
    height, width = img_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Mark center circle for removal
    center_x, center_y = width // 2, height // 2
    cv2.circle(mask, (center_x, center_y), 110, 255, -1)
    
    return mask


def test_basic_functionality():
    """Test basic seam carving without actual processing."""
    print("=" * 70)
    print("SEAM CARVING - BASIC FUNCTIONALITY TEST")
    print("=" * 70)
    
    # Create directories
    os.makedirs('images', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Create test image
    print("\n[1/4] Creating test image...")
    test_img = create_test_image()
    cv2.imwrite('images/test_image.jpg', test_img)
    print(f"✓ Created test image: images/test_image.jpg ({test_img.shape[1]}x{test_img.shape[0]})")
    
    # Create protection mask
    print("\n[2/4] Creating protection mask...")
    protect_mask = create_protection_mask(test_img.shape)
    cv2.imwrite('images/test_mask_protect.jpg', protect_mask)
    print("✓ Created protection mask: images/test_mask_protect.jpg")
    
    # Create removal mask
    print("\n[3/4] Creating removal mask...")
    remove_mask = create_removal_mask(test_img.shape)
    cv2.imwrite('images/test_mask_remove.jpg', remove_mask)
    print("✓ Created removal mask: images/test_mask_remove.jpg")
    
    # Test if seam carving module loads
    print("\n[4/4] Testing seam carving module import...")
    try:
        from seam_carving import ContentAwareImageResizer
        print("✓ Successfully imported ContentAwareImageResizer")
        
        # Display usage instructions
        print("\n" + "=" * 70)
        print("READY TO TEST!")
        print("=" * 70)
        print("\nNow you can run:")
        print("\n1. Basic resize:")
        print("   python main.py -i images/test_image.jpg -o output/test_resized.jpg -H 400 -W 600")
        
        print("\n2. Resize with protection:")
        print("   python main.py -i images/test_image.jpg -o output/test_protected.jpg \\")
        print("                  -H 400 -W 600 -p images/test_mask_protect.jpg")
        
        print("\n3. Object removal:")
        print("   python main.py -i images/test_image.jpg -o output/test_removed.jpg \\")
        print("                  -m remove -r images/test_mask_remove.jpg")
        
        print("\n" + "=" * 70)
        
    except ImportError as e:
        print(f"✗ Error importing seam carving module: {e}")
        print("\nMake sure seam_carving.py is in the same directory!")
        return False
    
    return True


def quick_resize_test():
    """Run a quick actual resize test."""
    print("\n" + "=" * 70)
    print("RUNNING QUICK RESIZE TEST (This may take a minute...)")
    print("=" * 70)
    
    try:
        from seam_carving import ContentAwareImageResizer
        
        if not os.path.exists('images/test_image.jpg'):
            print("✗ Test image not found. Run basic functionality test first.")
            return
        
        print("\nResizing test_image.jpg from 800x600 to 600x400...")
        resizer = ContentAwareImageResizer('images/test_image.jpg', 400, 600)
        resizer.export_result('output/quick_test.jpg')
        print("✓ Success! Result saved to output/quick_test.jpg")
        
        # Show file info
        result = cv2.imread('output/quick_test.jpg')
        print(f"  Original size: 800x600")
        print(f"  Result size: {result.shape[1]}x{result.shape[0]}")
        
    except Exception as e:
        print(f"✗ Error during resize: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Run quick test with actual processing
        if test_basic_functionality():
            quick_resize_test()
    else:
        # Just setup test files
        test_basic_functionality()
        print("\nTo run actual processing test, use: python test.py --quick")
