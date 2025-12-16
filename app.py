#!/usr/bin/env python3

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from seam_carving import ContentAwareImageResizer
import io


st.set_page_config(
    page_title="Seam Carving Demo",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    </style>
""", unsafe_allow_html=True)


def setup_directories():
    """Create necessary directories."""
    os.makedirs('output/streamlit', exist_ok=True)


def apply_seam_carving(image_array, target_width, target_height):
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_input:
        cv2.imwrite(tmp_input.name, image_bgr)
        input_path = tmp_input.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_output:
        output_path = tmp_output.name
    
    try:
        # apply seam carving
        resizer = ContentAwareImageResizer(
            input_path,
            target_height,
            target_width
        )
        resizer.export_result(output_path)
        
        # load result and check if it exists
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file not created: {output_path}")
        
        result_bgr = cv2.imread(output_path)
        if result_bgr is None:
            raise ValueError(f"Failed to read output image: {output_path}")
        
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        return result_rgb
        
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


def create_comparison_image(original, result):
    """Create side-by-side comparison at TRUE sizes (not scaled)."""
    
    # Get actual sizes
    oh, ow = original.shape[:2]
    rh, rw = result.shape[:2]
    
    # Use original images at their true size
    orig_copy = original.copy()
    result_copy = result.copy()
    
    # Add labels with size info
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Label for original
    label_orig = f'ORIGINAL {ow}x{oh}'
    cv2.putText(orig_copy, label_orig, (10, 40), 
               font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # Label for result
    label_result = f'SEAM CARVED {rw}x{rh}'
    cv2.putText(result_copy, label_result, (10, 40), 
               font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
    
    # If heights are different, pad the shorter one to match
    max_height = max(oh, rh)
    
    if oh < max_height:
        padding = max_height - oh
        orig_copy = np.pad(orig_copy, ((0, padding), (0, 0), (0, 0)), 
                          mode='constant', constant_values=255)
    
    if rh < max_height:
        padding = max_height - rh
        result_copy = np.pad(result_copy, ((0, padding), (0, 0), (0, 0)), 
                            mode='constant', constant_values=255)
    
    # Stack horizontally at true sizes
    comparison = np.hstack([orig_copy, result_copy])
    
    return comparison


def standard_resize(image, target_width, target_height):
    """apply standard resize for comparison."""
    return cv2.resize(image, (target_width, target_height))


def main():
    # Header
    st.title("🎨 Seam Carving: Content-Aware Image Resizing")
    st.markdown("### Resize images intelligently without distorting important content")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("---")
        st.subheader("About Seam Carving")
        st.info("""
        **Seam carving** is a content-aware image resizing technique that removes 
        the least important pixels from an image.
        """)
        
        st.markdown("---")
        st.subheader("Best Images")
        st.success("""
        - Landscapes
        - City scenes
        - Photos with backgrounds
        - Architecture
        """)
        
        st.markdown("---")
        st.subheader("Use Carefully")
        st.warning("""
        - Close-up portraits
        - Images with important edge objects
        - Already small images
        """)
    
    # Setup
    setup_directories()
    st.header("Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG, BMP)",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Display original image info
        original_height, original_width = image_array.shape[:2]
        
        st.success(f"image loaded successfully.")
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image_array, use_container_width=True)
            st.info(f"**Size:** {original_width} × {original_height} pixels")
        
        with col2:
            st.subheader("🎯 Resize Settings")
            
            # Resize method selection
            resize_method = st.radio(
                "Choose resize method:",
                ["Percentage", "Exact Dimensions", "Quick Presets"]
            )
            
            if resize_method == "Percentage":
                st.markdown("**Reduce by percentage:**")
                width_pct = st.slider(
                    "Width (%)",
                    min_value=10,
                    max_value=99,
                    value=70,
                    help="Percentage of original width to keep"
                )
                height_pct = st.slider(
                    "Height (%)",
                    min_value=10,
                    max_value=99,
                    value=70,
                    help="Percentage of original height to keep"
                )
                
                target_width = int(original_width * width_pct / 100)
                target_height = int(original_height * height_pct / 100)
                
            elif resize_method == "Exact Dimensions":
                st.markdown("**Enter exact dimensions:**")
                target_width = st.number_input(
                    "Width (pixels)",
                    min_value=50,
                    max_value=original_width - 1,
                    value=int(original_width * 0.7)
                )
                target_height = st.number_input(
                    "Height (pixels)",
                    min_value=50,
                    max_value=original_height - 1,
                    value=int(original_height * 0.7)
                )
                
            else: 
                preset = st.selectbox(
                    "Select a preset:",
                    [
                        "80% size",
                        "70% size",
                        "60% size",
                        "50% size",
                        "Mobile (400px wide)",
                        "Square crop"
                    ]
                )
                
                if preset == "80% size":
                    target_width = int(original_width * 0.8)
                    target_height = int(original_height * 0.8)
                elif preset == "70% size":
                    target_width = int(original_width * 0.7)
                    target_height = int(original_height * 0.7)
                elif preset == "60% size":
                    target_width = int(original_width * 0.6)
                    target_height = int(original_height * 0.6)
                elif preset == "50% size":
                    target_width = int(original_width * 0.5)
                    target_height = int(original_height * 0.5)
                elif preset == "Mobile (400px wide)":
                    target_width = 400
                    target_height = int(original_height * 400 / original_width)
                else: 
                    size = min(original_width, original_height)
                    target_width = size
                    target_height = size
            
            # Show target size
            st.info(f"""
            **Target Size:** {target_width} × {target_height} pixels
            
            **Reduction:**
            - Width: {original_width} → {target_width} ({100*(original_width-target_width)/original_width:.1f}% reduction)
            - Height: {original_height} → {target_height} ({100*(original_height-target_height)/original_height:.1f}% reduction)
            """)
            
            # Comparison option
            show_comparison = st.checkbox(
                "Compare with standard resize",
                value=True,
                help="Show comparison between seam carving and standard resize"
            )
        
        # Process button
        st.markdown("---")
        
        if st.button("🚀 Apply Seam Carving", type="primary"):
            
            # Validate dimensions
            if target_width >= original_width or target_height >= original_height:
                st.error(f"Error: Target dimensions must be smaller than original!")
                st.error(f"Original: {original_width}x{original_height}")
                st.error(f"Target: {target_width}x{target_height}")
            else:
                with st.spinner("Processing... This may take a moment..."):
                    try:
                        # Progress bar
                        progress_bar = st.progress(0)
                        progress_bar.progress(20)
                        
                        # Apply seam carving
                        result = apply_seam_carving(
                            image_array,
                            target_width,
                            target_height
                        )
                        progress_bar.progress(80)
                        
                        # Store in session state
                        st.session_state.result = result
                        st.session_state.original = image_array
                        st.session_state.target_width = target_width
                        st.session_state.target_height = target_height
                        st.session_state.show_comparison = show_comparison
                        
                        progress_bar.progress(100)
                        st.success("Processing complete!")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
        
        # Display results if available
        if 'result' in st.session_state:
            st.markdown("---")
            st.header("📊 Results")
            
            # Create tabs for different views
            tabs = st.tabs(["Seam Carved", "Comparison", "Side-by-Side"])
            
            with tabs[0]:
                st.subheader("🎨 Seam Carved Result")
                st.image(st.session_state.result, use_container_width=True)
                
                result_h, result_w = st.session_state.result.shape[:2]
                st.info(f"**Final Size:** {result_w} × {result_h} pixels")
                
                # Download button
                result_pil = Image.fromarray(st.session_state.result)
                buf = io.BytesIO()
                result_pil.save(buf, format='JPEG')
                
                st.download_button(
                    label="📥 Download Seam Carved Image",
                    data=buf.getvalue(),
                    file_name=f"seam_carved_{result_w}x{result_h}.jpg",
                    mime="image/jpeg"
                )
            
            with tabs[1]:
                if st.session_state.show_comparison:
                    st.subheader("🔍 Comparison: Seam Carving vs Standard Resize")
                    
                    # Create standard resize
                    standard = standard_resize(
                        st.session_state.original,
                        st.session_state.target_width,
                        st.session_state.target_height
                    )
                    
                    # Show TRUE sizes - don't use use_container_width
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**Standard Resize (Distorted)**")
                        st.image(standard)  # TRUE SIZE
                        std_h, std_w = standard.shape[:2]
                        st.caption(f"Size: {std_w}x{std_h} - Objects appear stretched/squashed")
                    
                    with col2:
                        st.markdown("**Seam Carving (Content-Aware)**")
                        st.image(st.session_state.result)  # TRUE SIZE
                        result_h, result_w = st.session_state.result.shape[:2]
                        st.caption(f"Size: {result_w}x{result_h} - Objects maintain proportions")
                    
                    st.info("""
                    **Key Difference:**
                    - **Standard Resize:** Simply scales all pixels uniformly, causing distortion
                    - **Seam Carving:** Intelligently removes less important pixels, preserving content
                    
                    ⚠️ Images shown at ACTUAL size - seam carved image should appear smaller
                    """)
                else:
                    st.info("Enable 'Compare with standard resize' to see the comparison")
            
            with tabs[2]:
                st.subheader("📸 Side-by-Side Comparison at TRUE SIZE")
                
                # Show original and result at actual sizes
                orig_h, orig_w = st.session_state.original.shape[:2]
                result_h, result_w = st.session_state.result.shape[:2]
                
                st.info(f"**Original:** {orig_w}x{orig_h} pixels  |  **Seam Carved:** {result_w}x{result_h} pixels")
                
                # Create columns proportional to image widths
                col1, col2 = st.columns([orig_w, result_w])
                
                with col1:
                    st.markdown(f"**Original ({orig_w}x{orig_h})**")
                    st.image(st.session_state.original)
                
                with col2:
                    st.markdown(f"**Seam Carved ({result_w}x{result_h})**")
                    st.image(st.session_state.result)
                
                st.success("✅ Images displayed at actual pixel dimensions - notice the size difference!")
                
                # Also create downloadable comparison
                comparison = create_comparison_image(
                    st.session_state.original,
                    st.session_state.result
                )
                
                st.markdown("---")
                st.markdown("**Download Combined Comparison:**")
                
                # Download comparison
                comparison_pil = Image.fromarray(comparison)
                buf_comp = io.BytesIO()
                comparison_pil.save(buf_comp, format='JPEG')
                
                st.download_button(
                    label="📥 Download Comparison Image",
                    data=buf_comp.getvalue(),
                    file_name="seam_carving_comparison.jpg",
                    mime="image/jpeg"
                )
    
    else:
        # Show instructions when no image is uploaded
        st.info("""
        👆 **Get Started:**
        1. Upload an image using the file uploader above
        2. Choose your resize method
        3. Click "Apply Seam Carving"
        4. Download your results!
        """)
        
        # Example images section
        st.markdown("---")
        st.header("💡 Example Use Cases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🏞️ Landscapes")
            st.markdown("""
            Perfect for:
            - Removing empty sky
            - Adjusting composition
            - Social media sizing
            """)
        
        with col2:
            st.subheader("🏙️ City Scenes")
            st.markdown("""
            Great for:
            - Street photography
            - Architecture shots
            - Urban landscapes
            """)
        
        with col3:
            st.subheader("📱 Mobile Optimization")
            st.markdown("""
            Ideal for:
            - Website images
            - Mobile screens
            - Email attachments
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit | Seam Carving Algorithm | Content-Aware Image Resizing</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
