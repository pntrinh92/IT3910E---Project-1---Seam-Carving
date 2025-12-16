# IT3910E - Project 1 - Seam Carving

The objective of this study is to utilize a seam carving algorithm to achieve content-aware image scaling and the seamless removal of specific objects. 

This approach enables image retargeting while preserving the integrity of semantic content, thereby avoiding the geometric distortion typically caused by standard scaling.

## Features

- **Content-Aware Resizing**: Resize images while preserving important features
- **Protection Masks**: Protect important regions (faces, objects) during resizing
- **Object Removal**: Seamlessly remove unwanted objects from images
- **Forward Energy**: Uses forward energy method for better quality results
- **COCO Dataset**


### Installation

```bash
pip install numpy opencv-python
```

#### Experiment

```bash
# Simple resize
python main.py -i images/input.jpg -o output/result.jpg -H 400 -W 600

# With protection mask
python main.py -i images/photo.jpg -o output/protected.jpg -H 400 -W 600 -p images/mask.jpg

# Object removal
python main.py -i images/photo.jpg -o output/cleaned.jpg -m remove -r images/object_mask.jpg
```

### Run Tests

```bash
# Create test images
python test.py

# Run processing test
python test.py --quick

# See examples
python main.py
```

## Project Structure

```
IT3910E---Project-1---Seam-Carving/
├── main.py                # main execution 
├── seam_carving.py        # seam carving algorithm details
├── test.py                # test examples
├── app.py                  
├── coco_examples.py       # COCO dataset specific examples            #
├── README.md              
├── images/                # input images 
│   ├── coco/             # COCO dataset images
│   └── masks/            # protect/removal masks
└── output/                # results
    └── coco/            
```

## Algorithm Overview

### 1. Energy Map Computation
Uses gradient-based energy calculation with Scharr operator:
- Computes gradients in X and Y directions
- Combines energy from all color channels
- Higher energy = more important features

### 2. Seam Finding (Dynamic Programming)
Finds minimum energy path through image:
- **Backward Energy**: standard DP approach
- **Forward Energy**: considers pixel differences 
- Handles both vertical and horizontal seams

### 3. Seam Removal/Insertion
- **Removal**: Deletes low-energy seams to reduce size
- **Insertion**: Duplicates seams to enlarge (averages neighbors)

### 4. Mask Support
- **Protection Masks**: Increase energy to preserve regions
- **Removal Masks**: Decrease energy to prioritize removal

## COCO Dataset 

### Setup
```bash
# Create directories
mkdir -p images/coco output/coco


### Process COCO Images

```bash
# 1 image
python main.py -i images/coco/val2017/000000000139.jpg -o output/coco/result.jpg -H 480 -W 640

# batch process
python coco_examples.py --batch

# run 
python coco_examples.py --demo

# Create example masks
python coco_examples.py --mask
```

## Advanced Usage

### Programmatic API

```python
from seam_carving import ContentAwareImageResizer

# Basic resize
resizer = ContentAwareImageResizer('input.jpg', target_height=400, target_width=600)
resizer.export_result('output.jpg')

# With protection
resizer = ContentAwareImageResizer('input.jpg', 400, 600, 
                                  protection_mask='mask.jpg')
resizer.export_result('protected_output.jpg')

# Object removal
resizer = ContentAwareImageResizer('input.jpg', original_h, original_w,
                                  removal_mask='object.jpg')
resizer.export_result('cleaned_output.jpg')
```

### Creating Masks

```python
import cv2
import numpy as np

img = cv2.imread('input.jpg')
h, w = img.shape[:2]

mask = np.zeros((h, w), dtype=np.uint8)
cv2.rectangle(mask, (100, 100), (300, 300), 255, -1)  
cv2.circle(mask, (400, 400), 50, 255, -1)            

cv2.imwrite('mask.jpg', mask)
```

### Testing

```bash
# Unit tests with sample images
python test.py

# Full processing test
python test.py --quick

# COCO demonstrations
python coco_examples.py --demo
```

#### Streamlit 
### Option 1: Use the Launcher Script
```bash
./launch_streamlit.sh
```

### Option 2: Direct Command
```bash
pip install streamlit opencv-python numpy Pillow
streamlit run app.py
```

1. **Upload Section** - Drag & drop your image
2. **Resize Options** - Choose percentage, exact size, or presets
3. **Process Button** - Click to apply seam carving
4. **Results Tabs**:
   - Seam Carved Result
   - Comparison with Standard Resize
   - Side-by-Side View
5. **Download Buttons** - Save your results


**Author**: IT3910E Course Project  
**Topic**: Seam Carving - Content-Aware Image Resizing  
**Dataset**: COCO Dataset (Common Objects in Context)

