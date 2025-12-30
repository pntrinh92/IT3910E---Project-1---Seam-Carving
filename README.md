# IT3910E - Project 1 - Seam Carving

The objective of this study is to utilize a seam carving algorithm to achieve content-aware image scaling and the seamless removal of specific objects. 

This approach enables image retargeting while preserving the integrity of semantic content, thereby avoiding the geometric distortion typically caused by standard scaling.

## Features

- **Content-Aware Resizing**: Resize images while preserving important features
- **Protection Masks**: Protect important regions (faces, objects) during resizing
- **Object Removal**: Seamlessly remove unwanted objects from images
- **Forward Energy**: Uses forward energy method for better quality results


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


## Batch Process All Images

The optimized `main.py` processes all images in the `input/` directory automatically.

### Basic Usage

```bash
# Process all images with default settings (medium size, 70%)
python main.py

# Process with small size (50% reduction)
python main.py --size small

# Process with all sizes (50%, 70%, 80%)
python main.py --size all

# Process only with standard method (faster)
python main.py --types standard --no-comparison

# Process only with backward energy
python main.py --types backward_energy
```



### Output Structure

output/
├── standard/          # Forward energy (default method)
├── backward_energy/   # Backward energy method
├── with_mask/         # With protection masks (if masks exist)
└── comparison/        # Side-by-side comparisons



## Web Interface (Streamlit)

```bash
streamlit run app.py
```

## 🖥️ Command Line (Single Image)

```bash
# Resize single image
python seam_carving.py -resize -im input/beach.jpg -out output/beach_result.jpg -dx -200 -dy 20

# With visualization
python seam_carving.py -resize -im input/beach.jpg -out output/result.jpg -dx -200 -dy 20 -vis

# Object removal
python seam_carving.py -remove -im input/image.jpg -out output/removed.jpg -rmask masks/object_mask.jpg
```

---

**Project**: IT3910E - Seam Carving  
**Optimized for**: Speed and batch processing

