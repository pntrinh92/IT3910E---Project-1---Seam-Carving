#!/usr/bin/env python3

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io
import time
from typing import Tuple, Optional

# Import seam carving functions
from seam_carving import seam_carve

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Seam Carving - Content-Aware Image Resizing",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "IT3910E Project 1 - Seam Carving Implementation"
    }
)

# ============================================================================
# STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling to the application."""
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding-top: 1rem;
        }
        
        /* Headers */
        h1 {
            color: #1f2937;
            font-weight: 700;
            padding-bottom: 1rem;
            border-bottom: 3px solid #e5e7eb;
        }
        
        h2 {
            color: #374151;
            font-weight: 600;
            margin-top: 2rem;
        }
        
        h3 {
            color: #4b5563;
            font-weight: 500;
        }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            background: #ffffff;
            color: #1f2937;
            height: 3.5em;
            border-radius: 10px;
            border: 2px solid #e5e7eb;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stButton>button:hover {
            background: #f9fafb;
            border-color: #d1d5db;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid #d1d5db;
            background-color: #f9fafb;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: 600;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f0f2f6;
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            color: #1f2937;
            border-bottom: 3px solid #6b7280;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed #d1d5db;
            border-radius: 10px;
            padding: 2rem;
            background-color: #ffffff;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #dbeafe 0%, #bfdbfe 100%);
            border-right: 1px solid #93c5fd;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #1e293b;
        }
        
        [data-testid="stSidebar"] h3 {
            color: #0f172a;
            font-weight: 600;
        }
        
        [data-testid="stSidebar"] p {
            color: #334155;
        }
        
        [data-testid="stSidebar"] label {
            color: #1e293b !important;
        }
        
        [data-testid="stSidebar"] .stRadio label {
            color: #1e293b !important;
        }
        
        [data-testid="stSidebar"] .stCheckbox label {
            color: #1e293b !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: #6b7280;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #666;
            border-top: 1px solid #e0e0e0;
            margin-top: 3rem;
        }
        </style>
    """, unsafe_allow_html=True)

def setup_directories():
    """Create necessary output directories."""
    os.makedirs('output/streamlit', exist_ok=True)

def get_image_info(image: np.ndarray) -> dict:
    """Extract image information."""
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    size_bytes = image.nbytes
    size_kb = size_bytes / 1024
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'size_kb': size_kb,
        'aspect_ratio': width / height
    }


def apply_seam_carving(
    image_array: np.ndarray, 
    target_width: int, 
    target_height: int,
    protection_mask: np.ndarray = None,
    use_forward_energy: bool = True,
    progress_callback=None
) -> np.ndarray:
    """
    Apply seam carving algorithm to resize image.
    
    Args:
        image_array: Input image as numpy array (RGB)
        target_width: Desired output width
        target_height: Desired output height
        progress_callback: Optional callback for progress updates
        
    Returns:
        Resized image as numpy array (RGB)
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Get current dimensions
    current_height, current_width = image_bgr.shape[:2]
    
    # SPEED OPTIMIZATION: Downsize large images (DISABLED for full quality)
    # Change MAX_WIDTH to control output size:
    # - 600: Fast processing, lower resolution
    # - 1200: Balanced quality and speed
    # - 2000: High quality, slower processing
    # - 99999: No downsizing, full resolution (slowest)
    MAX_WIDTH = 99999  # Set to 99999 for no downsizing (full resolution)
    scale_factor = 1.0
    
    if current_width > MAX_WIDTH:
        scale_factor = MAX_WIDTH / current_width
        new_width = MAX_WIDTH
        new_height = int(current_height * scale_factor)
        image_bgr = cv2.resize(image_bgr, (new_width, new_height))
        current_width = new_width
        current_height = new_height
        
        # Adjust target dimensions proportionally
        target_width = int(target_width * scale_factor)
        target_height = int(target_height * scale_factor)
        
        if progress_callback:
            progress_callback(0.1, f"Downsized to {new_width}x{new_height} for faster processing...")
    
    # Convert to float64
    image_bgr = image_bgr.astype(np.float64)
    
    # Process protection mask if provided
    mask_float = None
    if protection_mask is not None:
        # Resize mask to match image dimensions
        mask_resized = cv2.resize(protection_mask, (current_width, current_height))
        # Convert to grayscale if needed
        if len(mask_resized.shape) == 3:
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
        mask_float = mask_resized.astype(np.float64)
        if progress_callback:
            progress_callback(0.15, "Processing protection mask...")
    
    # Calculate seam changes needed
    dy = target_height - current_height
    dx = target_width - current_width
    
    if progress_callback:
        energy_type = "forward" if use_forward_energy else "backward"
        progress_callback(0.2, f"Computing {energy_type} energy map...")
    
    try:
        # Apply seam carving
        result_bgr = seam_carve(
            image_bgr, 
            dy, 
            dx, 
            mask=mask_float, 
            vis=False, 
            use_forward=use_forward_energy
        )
        
        if progress_callback:
            progress_callback(0.8, "Finalizing output...")
        
        # Ensure output is uint8
        result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return result_rgb
        
    except Exception as e:
        raise Exception(f"Seam carving failed: {str(e)}")

def standard_resize(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Apply standard resize for comparison."""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def render_sidebar():
    """Render sidebar with project information and settings."""
    with st.sidebar:
        st.markdown("### 🎯 Course Information ")
        
        st.markdown("""
        **Course:** IT3910E  
        **Project:** Seam Carving  
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 About Seam Carving")
        st.info("""
        Seam carving is an advanced image resizing technique that intelligently 
        removes or adds pixels while preserving important image features.
    
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Processing Methods")
        
        # Energy method selection
        energy_method = st.radio(
            "Energy Algorithm",
            ["Forward Energy", "Backward Energy"],
        )
        
        # Protection mask upload
        st.markdown("**Protection Mask (Optional)**")
        protection_mask_file = st.file_uploader(
            "Upload protection mask",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="White areas will be protected during resizing",
            key="protect_mask"
        )
        
        st.markdown("---")
        
        st.markdown("### 🛠️ Display Settings")
        show_comparison = st.checkbox("Show standard resize comparison", value=True)
        show_metrics = st.checkbox("Show detailed metrics", value=True)
        show_energy_map = st.checkbox("Show energy map", value=False)
        
        return {
            'show_comparison': show_comparison,
            'show_metrics': show_metrics,
            'show_energy_map': show_energy_map,
            'use_forward_energy': energy_method == "Forward Energy (Recommended)",
            'protection_mask': protection_mask_file
        }

def render_header():
    """Render application header."""
    st.title("🎨 Seam Carving: Content-Aware Image Resizing")
    
    st.markdown("---")

def render_upload_section():
    """Render image upload section."""
    st.header("📤 Upload Your Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to apply seam carving. Supported formats: JPG, PNG, BMP"
    )
    
    return uploaded_file

def render_resize_controls(original_width: int, original_height: int):
    """Render resize control interface."""
    st.header("🎛️ Configure Resize Settings")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Original Width", f"{original_width} px")
    with col2:
        st.metric("Original Height", f"{original_height} px")
    
    st.markdown("---")
    
    # Method selection
    method = st.radio(
        "Select Resize Method",
        ["Quick Presets", "Percentage", "Exact Dimensions"],
        horizontal=True
    )
    
    target_width, target_height = None, None
    
    if method == "Quick Presets":
        preset = st.selectbox(
            "Choose a preset",
            [
                "80% ",
                "70% ",
                "60% ",
                "50% ",
                "Custom percentage"
            ]
        )
        
        if "80%" in preset:
            factor = 0.8
        elif "70%" in preset:
            factor = 0.7
        elif "60%" in preset:
            factor = 0.6
        elif "50%" in preset:
            factor = 0.5
        else:
            factor = st.slider("Custom percentage", 10, 99, 70) / 100
        
        target_width = int(original_width * factor)
        target_height = int(original_height * factor)
    
    elif method == "Percentage":
        col1, col2 = st.columns(2)
        with col1:
            width_pct = st.slider(
                "Width (%)",
                min_value=10,
                max_value=99,
                value=70,
                help="Percentage of original width to keep"
            )
        with col2:
            height_pct = st.slider(
                "Height (%)",
                min_value=10,
                max_value=99,
                value=70,
                help="Percentage of original height to keep"
            )
        
        target_width = int(original_width * width_pct / 100)
        target_height = int(original_height * height_pct / 100)
    
    else:  # Exact Dimensions
        col1, col2 = st.columns(2)
        with col1:
            target_width = st.number_input(
                "Target Width (pixels)",
                min_value=50,
                max_value=original_width - 1,
                value=int(original_width * 0.7)
            )
        with col2:
            target_height = st.number_input(
                "Target Height (pixels)",
                min_value=50,
                max_value=original_height - 1,
                value=int(original_height * 0.7)
            )
    
    # Show preview of target dimensions
    st.markdown("---")
    st.subheader("📐 Target Dimensions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Target Width", 
            f"{target_width} px",
            delta=f"{target_width - original_width} px"
        )
    with col2:
        st.metric(
            "Target Height", 
            f"{target_height} px",
            delta=f"{target_height - original_height} px"
        )
    with col3:
        reduction = ((original_width * original_height) - (target_width * target_height)) / (original_width * original_height) * 100
        st.metric(
            "Size Reduction",
            f"{reduction:.1f}%"
        )
    
    return target_width, target_height

def render_results(original, result, settings):
    """Render processing results."""
    st.header("📊 Results")
    
    # Show saved file location if available
    if 'output_path' in st.session_state:
        st.success(f"Auto-saved to: `{st.session_state.output_path}`")
    
    # Create tabs for different views
    tab_list = ["✨ Seam Carved", "🔍 Comparison", "📸 Side-by-Side"]
    if settings.get('show_energy_map', False):
        tab_list.append("🔥 Energy Map")
    
    tabs = st.tabs(tab_list)
    
    with tabs[0]:
        st.subheader("Seam Carved Result")
        st.image(result, width='stretch')
        
        result_h, result_w = result.shape[:2]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Final Size:** {result_w} × {result_h} pixels")
        with col2:
            # Download button
            result_pil = Image.fromarray(result)
            buf = io.BytesIO()
            result_pil.save(buf, format='JPEG', quality=95)
            
            st.download_button(
                label="📥 Download Result",
                data=buf.getvalue(),
                file_name=f"seam_carved_{result_w}x{result_h}.jpg",
                mime="image/jpeg"
            )
    
    with tabs[1]:
        if settings['show_comparison']:
            st.subheader("Comparison: Seam Carving vs Standard Resize")
            
            # Create standard resize
            standard = standard_resize(original, result.shape[1], result.shape[0])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Standard Resize (Distorted)**")
                st.image(standard, width='stretch')
            
            with col2:
                st.markdown("**Seam Carving (Content-Aware)**")
                st.image(result, width='stretch')
            
        else:
            st.info("Enable 'Show standard resize comparison' in the sidebar to see comparison.")
    
    with tabs[2]:
        st.subheader("Side-by-Side Comparison")
        
        orig_h, orig_w = original.shape[:2]
        result_h, result_w = result.shape[:2]
        
        st.info(f"**Original:** {orig_w}×{orig_h} pixels  |  **Result:** {result_w}×{result_h} pixels")
        
        col1, col2 = st.columns([orig_w, result_w])
        
        with col1:
            st.markdown(f"**Original ({orig_w}×{orig_h})**")
            st.image(original)
        
        with col2:
            st.markdown(f"**Seam Carved ({result_w}×{result_h})**")
            st.image(result)
    
    # Energy Map Tab (if enabled)
    if settings.get('show_energy_map', False) and len(tabs) > 3:
        with tabs[3]:
            st.subheader("🔥 Energy Map Visualization")
            st.info("""
            The energy map shows pixel importance:
            - **Bright areas**: High energy (important features, edges)
            - **Dark areas**: Low energy (less important, will be removed first)
            """)
            
            # Calculate and show energy map
            try:
                from seam_carving import backward_energy
                
                # Convert to BGR for energy calculation
                original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR).astype(np.float64)
                
                # Calculate energy
                energy = backward_energy(original_bgr)
                
                # Normalize for visualization
                energy_normalized = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Apply colormap
                energy_colored = cv2.applyColorMap(energy_normalized, cv2.COLORMAP_JET)
                energy_rgb = cv2.cvtColor(energy_colored, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Image**")
                    st.image(original, width='stretch')
                with col2:
                    st.markdown("**Energy Map (Heatmap)**")
                    st.image(energy_rgb, width='stretch')
                
            except Exception as e:
                st.error(f"Could not generate energy map: {str(e)}")

def render_instructions():
    """Render instructions when no image is uploaded."""
    st.info("👆 **Get Started:** Upload an image above to begin")
    
    st.markdown("---")
    st.header("💡 How to Use This Application")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("""
        - Click the upload button
        - Select your image
        - Supported: JPG, PNG, BMP
        - Recommended: 800-2000px wide
        """)
    
    with col2:
        st.markdown("### 2️⃣ Configure")
        st.markdown("""
        - Choose resize method
        - Set target dimensions
        - Preview changes
        - Adjust as needed
        """)
    
    with col3:
        st.markdown("### 3️⃣ Process")
        st.markdown("""
        - Click 'Start Processing'
        - Wait for processing
        - View results
        - Download output
        """)
    
    st.markdown("---")
    
    st.header("🔬 Algorithm Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Forward Energy Method")
        st.markdown("""
        This implementation uses the **forward energy** algorithm, which:
        
        1. Analyzes pixel importance using gradient-based energy
        2. Identifies optimal seams (connected paths of low-energy pixels)
        3. Removes/Adds seams to resize the image
        4. Preserves important visual features and structures
        
        The forward energy method considers the cost of creating new edges 
        when removing pixels, resulting in better quality than backward energy.
        """)
    
    with col2:
        st.markdown("### Technical Details")
        st.markdown("""
        **Implementation:**
        - Language: Python 3.10+
        - Libraries: NumPy, OpenCV, SciPy, Numba
        - Algorithm: Forward Energy Seam Carving
        - Optimization: JIT compilation with Numba
        
        **Performance:**
        - Processes 500×375 image in ~5-10 seconds
        - Scales linearly with image size
        - Memory efficient implementation
        """)

def render_footer():
    """Render application footer."""
    st.markdown("---")
    st.markdown("""
        <div class='footer'>
            <p><strong>IT3910E - Project 1: Seam Carving</strong></p>
            <p>Content-Aware Image Resizing | Built with Streamlit & Python</p>
            <p style='font-size: 12px; color: #999; margin-top: 10px;'>
                © 2024 IT3910E Course Project
            </p>
        </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Apply styling
    apply_custom_css()
    
    # Setup directories
    setup_directories()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Render header
    render_header()
    
    # Render upload section
    uploaded_file = render_upload_section()
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Get image info
        img_info = get_image_info(image_array)
        
        # Display original image
        st.markdown("---")
        st.subheader("📷 Original Image")
        st.image(image_array, width='stretch')
        
        if settings['show_metrics']:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Width", f"{img_info['width']} px")
            with col2:
                st.metric("Height", f"{img_info['height']} px")
            with col3:
                st.metric("Size", f"{img_info['size_kb']:.1f} KB")
            with col4:
                st.metric("Aspect Ratio", f"{img_info['aspect_ratio']:.2f}")
        
        st.markdown("---")
        
        # Process protection mask if uploaded
        protection_mask = None
        if settings['protection_mask'] is not None:
            st.markdown("---")
            st.subheader("🛡️ Protection Mask")
            mask_image = Image.open(settings['protection_mask'])
            protection_mask = np.array(mask_image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(protection_mask, caption="Protection Mask", width=300)
            with col2:
                st.info("""
                **Protection Mask Applied**
                
                White areas in the mask will be protected during seam carving.
                The algorithm will avoid removing seams through these regions.
                """)
        
        st.markdown("---")
        
        # Render resize controls
        target_width, target_height = render_resize_controls(
            img_info['width'], 
            img_info['height']
        )
        
        st.markdown("---")
        
        # Process button
        if st.button("Starting Processing", type="primary"):
            # Validate dimensions
            if target_width >= img_info['width'] or target_height >= img_info['height']:
                st.error(" Error: Target dimensions must be smaller than original dimensions!")
                st.error(f"Original: {img_info['width']}×{img_info['height']} | Target: {target_width}×{target_height}")
            else:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(value, message):
                    progress_bar.progress(value)
                    status_text.text(message)
                
                try:
                    # Start timer
                    start_time = time.time()
                    
                    # Apply seam carving
                    update_progress(0.1, "Initializing...")
                    
                    # Show processing method
                    method_name = "Forward Energy" if settings['use_forward_energy'] else "Backward Energy"
                    mask_status = " + Protection Mask" if protection_mask is not None else ""
                    st.info(f"🔧 Processing with: **{method_name}{mask_status}**")
                    
                    result = apply_seam_carving(
                        image_array,
                        target_width,
                        target_height,
                        protection_mask=protection_mask,
                        use_forward_energy=settings['use_forward_energy'],
                        progress_callback=update_progress
                    )
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Auto-save results to output/streamlit
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Get original filename from uploaded file
                    original_name = uploaded_file.name.rsplit('.', 1)[0]
                    result_h, result_w = result.shape[:2]
                    
                    # Save seam carved result
                    output_filename = f"{original_name}_seamcarved_{result_w}x{result_h}_{timestamp}.jpg"
                    output_path = os.path.join('output/streamlit', output_filename)
                    
                    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, result_bgr)
                    
                    # Save comparison image if enabled
                    if settings['show_comparison']:
                        standard = standard_resize(image_array, target_width, target_height)
                        
                        # Convert images to BGR
                        orig_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        std_bgr = cv2.cvtColor(standard, cv2.COLOR_RGB2BGR)
                        
                        # Get max height for padding
                        max_height = max(orig_bgr.shape[0], result_bgr.shape[0], std_bgr.shape[0])
                        
                        # Pad images to same height
                        def pad_image(img, target_height):
                            h = img.shape[0]
                            if h < target_height:
                                padding = target_height - h
                                return np.pad(img, ((0, padding), (0, 0), (0, 0)), 
                                            mode='constant', constant_values=255)
                            return img
                        
                        orig_padded = pad_image(orig_bgr, max_height)
                        result_padded = pad_image(result_bgr, max_height)
                        std_padded = pad_image(std_bgr, max_height)
                        
                        # Create side-by-side comparison
                        comparison = np.hstack([orig_padded, result_padded, std_padded])
                        
                        comparison_filename = f"{original_name}_comparison_{timestamp}.jpg"
                        comparison_path = os.path.join('output/streamlit', comparison_filename)
                        cv2.imwrite(comparison_path, comparison)
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Success message with save location
                    st.success(f"Processing complete, (Time: {processing_time:.2f}s)")
                    st.info(f"Saved to: `{output_path}`")
                    
                    # Store result in session state
                    st.session_state.result = result
                    st.session_state.original = image_array
                    st.session_state.processing_time = processing_time
                    st.session_state.output_path = output_path
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
        
        # Display results if available
        if 'result' in st.session_state:
            st.markdown("---")
            render_results(
                st.session_state.original,
                st.session_state.result,
                settings
            )
            
            if settings['show_metrics']:
                st.markdown("---")
                st.subheader("⚡ Performance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
                with col2:
                    pixels_removed = (st.session_state.original.shape[0] * st.session_state.original.shape[1]) - \
                                   (st.session_state.result.shape[0] * st.session_state.result.shape[1])
                    st.metric("Pixels Removed", f"{pixels_removed:,}")
                with col3:
                    seams_removed = abs(st.session_state.original.shape[1] - st.session_state.result.shape[1]) + \
                                  abs(st.session_state.original.shape[0] - st.session_state.result.shape[0])
                    st.metric("Total Seams", f"{seams_removed}")
    
    else:
        # Show instructions when no image is uploaded
        render_instructions()
    
    # Render footer
    render_footer()

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
