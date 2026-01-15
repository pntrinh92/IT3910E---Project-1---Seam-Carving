import numpy as np
import cv2
from scipy import ndimage as ndi

# Try to import numba for JIT compilation (optional, for performance)
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(func=None, *args, **kwargs):
        if func is None:
            def decorator(f):
                return f
            return decorator
        else:
            return func

ENERGY_MASK_CONST = 100000.0
MASK_THRESHOLD = 10


def backward_energy(im):
    if len(im.shape) == 2:
        xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
        ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
        grad_mag = np.sqrt(xgrad**2 + ygrad**2)
    else:
        xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
        ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
        grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))
    
    return grad_mag


def forward_energy(im):
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    energy = cU
        
    return energy


@jit    
def find_greedy_seam_jit(energy, h, w):
    seam_idx = np.zeros(h, dtype=np.int32)
    
    seam_idx[0] = np.argmin(energy[0])
    
    for i in range(1, h):
        current_col = seam_idx[i-1]
        
        if current_col == 0:
            neighbors = [0, 1]
        elif current_col == w - 1:
            neighbors = [w-2, w-1]
        else:
            neighbors = [current_col - 1, current_col, current_col + 1]
        
        min_energy = float('inf')
        min_col = current_col
        
        for col in neighbors:
            if energy[i, col] < min_energy:
                min_energy = energy[i, col]
                min_col = col
        
        seam_idx[i] = min_col
    
    return seam_idx


def find_greedy_seam_python(energy, h, w):
    seam_idx = np.zeros(h, dtype=np.int32)
    
    seam_idx[0] = np.argmin(energy[0])
    
    for i in range(1, h):
        current_col = seam_idx[i-1]
        
        if current_col == 0:
            neighbor_energies = energy[i, 0:2]
            offset = 0
        elif current_col == w - 1:
            neighbor_energies = energy[i, w-2:w]
            offset = w - 2
        else:
            neighbor_energies = energy[i, current_col-1:current_col+2]
            offset = current_col - 1
        
        min_idx = np.argmin(neighbor_energies)
        seam_idx[i] = offset + min_idx
    
    return seam_idx


def get_greedy_seam(im, mask=None, use_forward=True):
    h, w = im.shape[:2]
    
    energyfn = forward_energy if use_forward else backward_energy
    energy = energyfn(im)
    
    if mask is not None:
        energy[np.where(mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST
    
    if HAS_NUMBA:
        seam_idx = find_greedy_seam_jit(energy, h, w)
    else:
        seam_idx = find_greedy_seam_python(energy, h, w)
    
    boolmask = np.ones((h, w), dtype=np.bool_)
    for i in range(h):
        boolmask[i, seam_idx[i]] = False
    
    return seam_idx, boolmask


@jit
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    output = np.zeros((h, w - 1, 3), dtype=im.dtype)
    for row in range(h):
        col_idx = 0
        for col in range(w):
            if boolmask[row, col]:
                output[row, col_idx, 0] = im[row, col, 0]
                output[row, col_idx, 1] = im[row, col, 1]
                output[row, col_idx, 2] = im[row, col, 2]
                col_idx += 1
    return output


@jit
def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    output = np.zeros((h, w - 1), dtype=im.dtype)
    for row in range(h):
        col_idx = 0
        for col in range(w):
            if boolmask[row, col]:
                output[row, col_idx] = im[row, col]
                col_idx += 1
    return output


def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)


def greedy_seams_removal(im, num_remove, mask=None, vis=False, rot=False, use_forward=True, step_callback=None):
    for i in range(num_remove):
        seam_idx, boolmask = get_greedy_seam(im, mask, use_forward)
        
        # Call the callback with current progress, image BEFORE removal, and seam indices
        if step_callback is not None:
            step_callback(i + 1, num_remove, im, seam_idx, rot)
        
        im = remove_seam(im, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    
    return im, mask


def greedy_seam_carve(im, dy, dx, mask=None, vis=False, use_forward=True, step_callback=None):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    if mask is not None:
        mask = mask.astype(np.float64)

    output = im

    if dx < 0:
        output, mask = greedy_seams_removal(output, -dx, mask, vis, use_forward=use_forward, step_callback=step_callback)

    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = greedy_seams_removal(output, -dy, mask, vis, rot=True, use_forward=use_forward, step_callback=step_callback)
        output = rotate_image(output, False)

    return output


if __name__ == '__main__':
    import argparse
    import os
    import time
    
    ap = argparse.ArgumentParser(description='Greedy Seam Carving - Fast but Suboptimal')
    ap.add_argument("-im", help="Path to image", required=True)
    ap.add_argument("-out", help="Output file name", required=True)
    ap.add_argument("-dy", help="Number of vertical seams to remove", type=int, default=0)
    ap.add_argument("-dx", help="Number of horizontal seams to remove", type=int, default=0)
    ap.add_argument("-backward_energy", help="Use backward energy", action='store_true')
    args = vars(ap.parse_args())

    IM_PATH = args["im"]
    OUTPUT_NAME = args["out"]
    
    if not os.path.exists(IM_PATH):
        print(f"Error: Image file not found: {IM_PATH}")
        exit(1)
    
    im = cv2.imread(IM_PATH)
    if im is None:
        print(f"Error: Could not read image: {IM_PATH}")
        exit(1)
    
    h, w = im.shape[:2]
    print(f"Input image: {w}x{h} pixels")
    
    USE_FORWARD = not args["backward_energy"]
    energy_type = "forward" if USE_FORWARD else "backward"
    print(f"Using {energy_type} energy")
    
    # Apply greedy seam carving
    dy, dx = args["dy"], args["dx"]
    print(f"Removing seams: dy={dy}, dx={dx}")
    
    start_time = time.time()
    output = greedy_seam_carve(im, dy, dx, mask=None, vis=False, use_forward=USE_FORWARD)
    elapsed_time = time.time() - start_time
    
    # Save result
    output = np.clip(output, 0, 255).astype(np.uint8)
    cv2.imwrite(OUTPUT_NAME, output)
    
    result_h, result_w = output.shape[:2]
    print(f"Output image: {result_w}x{result_h} pixels")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Saved to: {OUTPUT_NAME}")
    