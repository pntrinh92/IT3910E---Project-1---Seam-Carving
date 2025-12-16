import numpy as np
import cv2
import argparse
import sys


class ContentAwareImageResizer:
    def __init__(self, input_file, target_height, target_width, 
                 protection_mask='', removal_mask=''):
        #initialize the image resizer with parameters
        self.input_path = input_file
        self.target_h = target_height
        self.target_w = target_width
        
        self.source_img = cv2.imread(input_file).astype(np.float64)
        self.original_h, self.original_w = self.source_img.shape[:2]
        
        #initial working img 
        self.working_img = np.copy(self.source_img)
        
        # operation mode determined
        self.remove_object_mode = (removal_mask != '')
        
        if self.remove_object_mode:
            # Load object removal mask
            self.protection_map = cv2.imread(removal_mask, 0).astype(np.float64)
            self.has_protection = False
        else:
            self.has_protection = (protection_mask != '')
            if self.has_protection:
                self.protection_map = cv2.imread(protection_mask, 0).astype(np.float64)
        
        # define convolution kernels for forward energy computation
        self.horizontal_kernel = np.array([[0., 0., 0.], 
                                           [-1., 0., 1.], 
                                           [0., 0., 0.]], dtype=np.float64)
        
        self.vertical_left_kernel = np.array([[0., 0., 0.], 
                                              [0., 0., 1.], 
                                              [0., -1., 0.]], dtype=np.float64)
        
        self.vertical_right_kernel = np.array([[0., 0., 0.], 
                                               [1., 0., 0.], 
                                               [0., -1., 0.]], dtype=np.float64)
        
        # high weight value for masked regions
        self.mask_weight = 1000
        
        # execute
        self.execute_processing()
    
    
    def execute_processing(self):
        if self.remove_object_mode:
            self.perform_object_removal()
        else:
            self.perform_content_aware_resize()
    
    
    def perform_content_aware_resize(self):
        """resize image using seam carving in both directions. vertical -> horizontal"""
        height_diff = int(self.target_h - self.original_h)
        width_diff = int(self.target_w - self.original_w)
        
        # vertical
        if width_diff < 0:
            self.remove_vertical_seams(abs(width_diff))
        elif width_diff > 0:
            self.insert_vertical_seams(width_diff)
        
        # horizontal - rotate
        if height_diff < 0:
            self.working_img = self.rotate_image_ccw(self.working_img)
            if self.has_protection:
                self.protection_map = self.rotate_mask_ccw(self.protection_map)
            self.remove_vertical_seams(abs(height_diff))
            self.working_img = self.rotate_image_cw(self.working_img)
        elif height_diff > 0:
            self.working_img = self.rotate_image_ccw(self.working_img)
            if self.has_protection:
                self.protection_map = self.rotate_mask_ccw(self.protection_map)
            self.insert_vertical_seams(height_diff)
            self.working_img = self.rotate_image_cw(self.working_img)
    
    
    def perform_object_removal(self):
        """remove object marked by mask and restore original dimensions."""
        needs_rotation = False
        obj_h, obj_w = self.calculate_object_bounds()
        
        # rotate if object is wider than tall 
        if obj_h < obj_w:
            self.working_img = self.rotate_image_ccw(self.working_img)
            self.protection_map = self.rotate_mask_ccw(self.protection_map)
            needs_rotation = True
        
        # remove seams through the masked object
        while np.any(self.protection_map > 0):
            gradient_map = self.calculate_gradient_energy()
            # Apply negative weight to prioritize masked pixels
            gradient_map[self.protection_map > 0] *= -self.mask_weight
            cost_map = self.build_forward_cost_map(gradient_map)
            optimal_path = self.trace_minimum_seam(cost_map)
            self.remove_single_seam(optimal_path)
            self.remove_seam_from_mask(optimal_path)
        
        # calculate pixels 
        if not needs_rotation:
            restore_count = self.original_w - self.working_img.shape[1]
        else:
            restore_count = self.original_h - self.working_img.shape[1]
        
        # restore
        self.insert_vertical_seams(restore_count)
        
        if needs_rotation:
            self.working_img = self.rotate_image_cw(self.working_img)
    
    
    def remove_vertical_seams(self, seam_count):
        if self.has_protection:
            for _ in range(seam_count):
                gradient_map = self.calculate_gradient_energy()
                # Increase energy in protected regions
                gradient_map[self.protection_map > 0] *= self.mask_weight
                cost_map = self.build_forward_cost_map(gradient_map)
                optimal_path = self.trace_minimum_seam(cost_map)
                self.remove_single_seam(optimal_path)
                self.remove_seam_from_mask(optimal_path)
        else:
            for _ in range(seam_count):
                gradient_map = self.calculate_gradient_energy()
                cost_map = self.build_forward_cost_map(gradient_map)
                optimal_path = self.trace_minimum_seam(cost_map)
                self.remove_single_seam(optimal_path)
    
    
    def insert_vertical_seams(self, seam_count):
        if self.has_protection:
            # store original state
            backup_img = np.copy(self.working_img)
            backup_mask = np.copy(self.protection_map)
            seam_registry = []
            
            # find seams to insert
            for _ in range(seam_count):
                gradient_map = self.calculate_gradient_energy()
                gradient_map[self.protection_map > 0] *= self.mask_weight
                cost_map = self.build_backward_cost_map(gradient_map)
                optimal_path = self.trace_minimum_seam(cost_map)
                seam_registry.append(optimal_path)
                self.remove_single_seam(optimal_path)
                self.remove_seam_from_mask(optimal_path)
            
            # restore and insert seams
            self.working_img = np.copy(backup_img)
            self.protection_map = np.copy(backup_mask)
            
            for _ in range(len(seam_registry)):
                current_seam = seam_registry.pop(0)
                self.duplicate_seam(current_seam)
                self.duplicate_seam_on_mask(current_seam)
                seam_registry = self.adjust_seam_indices(seam_registry, current_seam)
        else:
            # Store original state
            backup_img = np.copy(self.working_img)
            seam_registry = []
            
            # Find seams to insert
            for _ in range(seam_count):
                gradient_map = self.calculate_gradient_energy()
                cost_map = self.build_backward_cost_map(gradient_map)
                optimal_path = self.trace_minimum_seam(cost_map)
                seam_registry.append(optimal_path)
                self.remove_single_seam(optimal_path)
            
            # Restore and insert seams
            self.working_img = np.copy(backup_img)
            
            for _ in range(len(seam_registry)):
                current_seam = seam_registry.pop(0)
                self.duplicate_seam(current_seam)
                seam_registry = self.adjust_seam_indices(seam_registry, current_seam)
    
    
    def calculate_gradient_energy(self):
        """Compute gradient-based energy map using Scharr operator."""
        blue_ch, green_ch, red_ch = cv2.split(self.working_img)
        
        blue_grad = np.absolute(cv2.Scharr(blue_ch, -1, 1, 0)) + \
                    np.absolute(cv2.Scharr(blue_ch, -1, 0, 1))
        
        green_grad = np.absolute(cv2.Scharr(green_ch, -1, 1, 0)) + \
                     np.absolute(cv2.Scharr(green_ch, -1, 0, 1))
        
        red_grad = np.absolute(cv2.Scharr(red_ch, -1, 1, 0)) + \
                   np.absolute(cv2.Scharr(red_ch, -1, 0, 1))
        
        return blue_grad + green_grad + red_grad
    
    
    def build_backward_cost_map(self, gradient_map):
        """Build cumulative cost map using backward energy (standard DP)."""
        rows, cols = gradient_map.shape
        cost_table = np.copy(gradient_map)
        
        for r in range(1, rows):
            for c in range(cols):
                left_bound = max(c - 1, 0)
                right_bound = min(c + 2, cols)
                cost_table[r, c] = gradient_map[r, c] + \
                                   np.amin(cost_table[r - 1, left_bound:right_bound])
        
        return cost_table
    
    
    def build_forward_cost_map(self, gradient_map):
        """
        Build cumulative cost map using forward energy method.
        Considers pixel differences when removing seams.
        """
        horizontal_diff = self.apply_convolution_kernel(self.horizontal_kernel)
        left_vertical_diff = self.apply_convolution_kernel(self.vertical_left_kernel)
        right_vertical_diff = self.apply_convolution_kernel(self.vertical_right_kernel)
        
        rows, cols = gradient_map.shape
        cost_table = np.copy(gradient_map)
        
        for r in range(1, rows):
            for c in range(cols):
                if c == 0:
                    # left 
                    right_cost = cost_table[r - 1, c + 1] + \
                                horizontal_diff[r - 1, c + 1] + \
                                right_vertical_diff[r - 1, c + 1]
                    up_cost = cost_table[r - 1, c] + horizontal_diff[r - 1, c]
                    cost_table[r, c] = gradient_map[r, c] + min(right_cost, up_cost)
                    
                elif c == cols - 1:
                    # right 
                    left_cost = cost_table[r - 1, c - 1] + \
                               horizontal_diff[r - 1, c - 1] + \
                               left_vertical_diff[r - 1, c - 1]
                    up_cost = cost_table[r - 1, c] + horizontal_diff[r - 1, c]
                    cost_table[r, c] = gradient_map[r, c] + min(left_cost, up_cost)
                    
                else:
                    # middle columns
                    left_cost = cost_table[r - 1, c - 1] + \
                               horizontal_diff[r - 1, c - 1] + \
                               left_vertical_diff[r - 1, c - 1]
                    right_cost = cost_table[r - 1, c + 1] + \
                                horizontal_diff[r - 1, c + 1] + \
                                right_vertical_diff[r - 1, c + 1]
                    up_cost = cost_table[r - 1, c] + horizontal_diff[r - 1, c]
                    cost_table[r, c] = gradient_map[r, c] + \
                                      min(left_cost, right_cost, up_cost)
        
        return cost_table
    
    
    def apply_convolution_kernel(self, filter_kernel):
        blue_ch, green_ch, red_ch = cv2.split(self.working_img)
        
        result = np.absolute(cv2.filter2D(blue_ch, -1, kernel=filter_kernel)) + \
                 np.absolute(cv2.filter2D(green_ch, -1, kernel=filter_kernel)) + \
                 np.absolute(cv2.filter2D(red_ch, -1, kernel=filter_kernel))
        
        return result
    
    
    def trace_minimum_seam(self, cost_map):
        rows, cols = cost_map.shape
        seam_path = np.zeros(rows, dtype=np.uint32)
        seam_path[-1] = np.argmin(cost_map[-1])
        
        # Trace back to top
        for r in range(rows - 2, -1, -1):
            previous_col = seam_path[r + 1]
            
            if previous_col == 0:
                # Left boundary
                seam_path[r] = np.argmin(cost_map[r, :2])
            else:
                # General case
                left_bound = previous_col - 1
                right_bound = min(previous_col + 2, cols)
                seam_path[r] = np.argmin(cost_map[r, left_bound:right_bound]) + left_bound
        
        return seam_path
    
    
    def remove_single_seam(self, seam_indices):
        rows, cols = self.working_img.shape[:2]
        reduced_img = np.zeros((rows, cols - 1, 3))
        
        for r in range(rows):
            target_col = seam_indices[r]
            reduced_img[r, :, 0] = np.delete(self.working_img[r, :, 0], target_col)
            reduced_img[r, :, 1] = np.delete(self.working_img[r, :, 1], target_col)
            reduced_img[r, :, 2] = np.delete(self.working_img[r, :, 2], target_col)
        
        self.working_img = np.copy(reduced_img)
    
    
    def duplicate_seam(self, seam_indices):
        rows, cols = self.working_img.shape[:2]
        expanded_img = np.zeros((rows, cols + 1, 3))
        
        for r in range(rows):
            target_col = seam_indices[r]
            
            for channel in range(3):
                if target_col == 0:
                    # Left edge
                    avg_value = np.average(self.working_img[r, target_col:target_col + 2, channel])
                    expanded_img[r, target_col, channel] = self.working_img[r, target_col, channel]
                    expanded_img[r, target_col + 1, channel] = avg_value
                    expanded_img[r, target_col + 2:, channel] = self.working_img[r, target_col + 1:, channel]
                else:
                    # General case
                    avg_value = np.average(self.working_img[r, target_col - 1:target_col + 1, channel])
                    expanded_img[r, :target_col, channel] = self.working_img[r, :target_col, channel]
                    expanded_img[r, target_col, channel] = avg_value
                    expanded_img[r, target_col + 1:, channel] = self.working_img[r, target_col:, channel]
        
        self.working_img = np.copy(expanded_img)
    
    
    def adjust_seam_indices(self, seam_list, inserted_seam):
        updated_seams = []
        
        for seam in seam_list:
            # Shift indices that are at or past the inserted seam
            seam[seam >= inserted_seam] += 2
            updated_seams.append(seam)
        
        return updated_seams
    
    
    def rotate_image_ccw(self, img):
        rows, cols, channels = img.shape
        rotated = np.zeros((cols, rows, channels))
        flipped = np.fliplr(img)
        
        for ch in range(channels):
            for r in range(rows):
                rotated[:, r, ch] = flipped[r, :, ch]
        
        return rotated
    
    
    def rotate_image_cw(self, img):
        rows, cols, channels = img.shape
        rotated = np.zeros((cols, rows, channels))
        
        for ch in range(channels):
            for r in range(rows):
                rotated[:, rows - 1 - r, ch] = img[r, :, ch]
        
        return rotated
    
    
    def rotate_mask_ccw(self, mask_array):

        rows, cols = mask_array.shape
        rotated = np.zeros((cols, rows))
        flipped = np.fliplr(mask_array)
        
        for r in range(rows):
            rotated[:, r] = flipped[r, :]
        
        return rotated
    
    
    def remove_seam_from_mask(self, seam_indices):
        """Remove seam from protection mask."""
        rows, cols = self.protection_map.shape
        reduced_mask = np.zeros((rows, cols - 1))
        
        for r in range(rows):
            target_col = seam_indices[r]
            reduced_mask[r, :] = np.delete(self.protection_map[r, :], target_col)
        
        self.protection_map = np.copy(reduced_mask)
    
    
    def duplicate_seam_on_mask(self, seam_indices):
        """Duplicate seam in protection mask."""
        rows, cols = self.protection_map.shape
        expanded_mask = np.zeros((rows, cols + 1))
        
        for r in range(rows):
            target_col = seam_indices[r]
            
            if target_col == 0:
                avg_value = np.average(self.protection_map[r, target_col:target_col + 2])
                expanded_mask[r, target_col] = self.protection_map[r, target_col]
                expanded_mask[r, target_col + 1] = avg_value
                expanded_mask[r, target_col + 2:] = self.protection_map[r, target_col + 1:]
            else:
                avg_value = np.average(self.protection_map[r, target_col - 1:target_col + 1])
                expanded_mask[r, :target_col] = self.protection_map[r, :target_col]
                expanded_mask[r, target_col] = avg_value
                expanded_mask[r, target_col + 1:] = self.protection_map[r, target_col:]
        
        self.protection_map = np.copy(expanded_mask)
    
    
    def calculate_object_bounds(self):
        """Calculate dimensions of masked object region."""
        row_coords, col_coords = np.where(self.protection_map > 0)
        obj_height = np.amax(row_coords) - np.amin(row_coords) + 1
        obj_width = np.amax(col_coords) - np.amin(col_coords) + 1
        
        return obj_height, obj_width
    
    
    def export_result(self, output_filename):
        """Save the processed image to file."""
        cv2.imwrite(output_filename, self.working_img.astype(np.uint8))



if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Seam Carving - Content-Aware Image Resizing')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-resize", action='store_true', help="Image resize mode")
    group.add_argument("-remove", action='store_true', help="Object removal mode")

    ap.add_argument("-im", help="Path to input image", required=True)
    ap.add_argument("-out", help="Output file name", required=True)
    ap.add_argument("-mask", help="Path to protective mask (optional)", default='')
    ap.add_argument("-rmask", help="Path to removal mask (for object removal)", default='')
    ap.add_argument("-dy", help="Number of rows to add/remove (negative to remove)", type=int, default=0)
    ap.add_argument("-dx", help="Number of columns to add/remove (negative to remove)", type=int, default=0)
    
    args = vars(ap.parse_args())

    # Read input image
    IM_PATH = args["im"]
    OUTPUT_NAME = args["out"]
    MASK_PATH = args["mask"]
    R_MASK_PATH = args["rmask"]
    
    # Validate input
    im = cv2.imread(IM_PATH)
    if im is None:
        print(f"Error: Could not load image from {IM_PATH}")
        sys.exit(1)
    
    h, w = im.shape[:2]
    print(f"Input image: {IM_PATH}")
    print(f"Original size: {w}x{h}")
    
    # Resize mode
    if args["resize"]:
        dy, dx = args["dy"], args["dx"]
        
        if dy == 0 and dx == 0:
            print("Error: Please specify -dy and/or -dx for resize")
            sys.exit(1)
        
        # Calculate target dimensions
        target_h = h + dy
        target_w = w + dx
        
        if target_h <= 0 or target_w <= 0:
            print(f"Error: Invalid target dimensions {target_w}x{target_h}")
            sys.exit(1)
        
        print(f"Target size: {target_w}x{target_h}")
        print(f"Change: {dx:+d} columns, {dy:+d} rows")
        
        # Check if we need to add or remove seams
        if dx > 0 or dy > 0:
            print("Warning: Seam insertion (enlarging) may produce lower quality results")
        
        print("Processing...")
        
        # Apply seam carving
        resizer = ContentAwareImageResizer(
            IM_PATH,
            target_h,
            target_w,
            protection_mask=MASK_PATH
        )
        resizer.export_result(OUTPUT_NAME)
        
        # Verify output
        result = cv2.imread(OUTPUT_NAME)
        if result is not None:
            result_h, result_w = result.shape[:2]
            print(f"✓ Success! Output saved to: {OUTPUT_NAME}")
            print(f"  Final size: {result_w}x{result_h}")
        else:
            print("Error: Failed to save output")
            sys.exit(1)
    
    # Object removal mode
    elif args["remove"]:
        if not R_MASK_PATH:
            print("Error: Object removal requires -rmask parameter")
            sys.exit(1)
        
        print(f"Object removal mask: {R_MASK_PATH}")
        print("Processing...")
        
        # Apply object removal
        resizer = ContentAwareImageResizer(
            IM_PATH,
            h,  # Keep original height
            w,  # Keep original width
            protection_mask=MASK_PATH,
            removal_mask=R_MASK_PATH
        )
        resizer.export_result(OUTPUT_NAME)
        
        print(f"success, output saved to: {OUTPUT_NAME}")



