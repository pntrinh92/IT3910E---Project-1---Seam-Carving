from seam_carving import ContentAwareImageResizer
import os
import argparse
import cv2


def resize_image_basic(input_path, output_path, target_height, target_width):
    #Basic image resizing without any mask protection.
    print(f"Resizing {input_path}")
    print(f"Target size: {target_width}x{target_height}")
    
    resizer = ContentAwareImageResizer(input_path, target_height, target_width)
    resizer.export_result(output_path)
    
    print(f"saved result to {output_path}")


def resize_with_protection(input_path, output_path, target_height, target_width, mask_path):
    #Resize image while protecting important regions marked by mask.
    print(f"Resizing {input_path} with protection mask")
    print(f"Target size: {target_width}x{target_height}")
    
    resizer = ContentAwareImageResizer(input_path, target_height, target_width, 
                                      protection_mask=mask_path)
    resizer.export_result(output_path)
    
    print(f"saved result to {output_path}")


def remove_object(input_path, output_path, mask_path):
    print(f"Removing object from {input_path}")
    
    # Get original dimensions
    img = cv2.imread(input_path)
    original_height, original_width = img.shape[:2]
    
    resizer = ContentAwareImageResizer(input_path, original_height, original_width,
                                      removal_mask=mask_path)
    resizer.export_result(output_path)
    
    print(f"saved result to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Seam Carving - Content-Aware Image Resizing')
    
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', required=True, help='Output image path')
    parser.add_argument('--height', '-H', type=int, help='Target height')
    parser.add_argument('--width', '-W', type=int, help='Target width')
    parser.add_argument('--protect-mask', '-p', help='Protection mask path (optional)')
    parser.add_argument('--remove-mask', '-r', help='Object removal mask path (optional)')
    parser.add_argument('--mode', '-m', choices=['resize', 'remove'], default='resize',
                       help='Mode: resize or remove object')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Execute based on mode
    if args.mode == 'remove':
        if not args.remove_mask:
            print("Error: --remove-mask required for object removal mode!")
            return
        remove_object(args.input, args.output, args.remove_mask)
        
    else:  # resize mode
        if args.height is None or args.width is None:
            print("Error: --height and --width required for resize mode!")
            return
            
        if args.protect_mask:
            resize_with_protection(args.input, args.output, args.height, args.width, 
                                 args.protect_mask)
        else:
            resize_image_basic(args.input, args.output, args.height, args.width)


def run_examples():
    if not os.path.exists('output'):
        os.makedirs('output')
    
    print("Seam carving")
    
    # basic resize
    print("\nbasic image resize")
    print("-" * 60)
    if os.path.exists('images/input.jpg'):
        resize_image_basic('images/input.jpg', 'output/resized.jpg', 400, 300)
    else:
        print("⚠ Skipped: images/input.jpg not found")
    
    # resize with protection mask
    print("\nresize with protection mask")
    print("-" * 60)
    if os.path.exists('images/input.jpg') and os.path.exists('images/mask_protect.jpg'):
        resize_with_protection('images/input.jpg', 'output/protected_resize.jpg', 
                              400, 300, 'images/mask_protect.jpg')
    else:
        print("skip: required files not found")
    #object removal
    print("\nobject removal")
    print("-" * 60)
    if os.path.exists('images/input.jpg') and os.path.exists('images/mask_remove.jpg'):
        remove_object('images/input.jpg', 'output/object_removed.jpg', 
                     'images/mask_remove.jpg')
    else:
        print("skip: required files not found")
    
    print("complete! check the 'output' folder.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("No arguments provided. Running examples...\n")
        print("  python main.py --input <image> --output <output> --height <h> --width <w>")
        print("  python main.py --input <image> --output <output> --mode remove --remove-mask <mask>")
        print("\n help: python main.py --help")
        print("\n" + "=" * 60 + "\n")
        run_examples()
    else:
        main()