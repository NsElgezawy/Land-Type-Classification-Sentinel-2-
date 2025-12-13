import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="YOLO Satellite Detection",
    page_icon="🛰️",
    layout="wide"
)

# Custom CSS for dynamic satellite-themed background
st.markdown("""
    <style>
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 25%, #2c5364 50%, #1a3a4a 75%, #0f2027 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Add subtle satellite grid overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Floating particles effect */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(255, 255, 255, 0.3), transparent),
            radial-gradient(2px 2px at 60% 70%, rgba(0, 255, 255, 0.2), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(255, 255, 255, 0.2), transparent),
            radial-gradient(2px 2px at 80% 10%, rgba(0, 255, 255, 0.3), transparent),
            radial-gradient(1px 1px at 90% 60%, rgba(255, 255, 255, 0.2), transparent);
        background-size: 200% 200%;
        animation: particleFloat 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes particleFloat {
        0%, 100% { transform: translate(0, 0); }
        25% { transform: translate(10px, -10px); }
        50% { transform: translate(-5px, 10px); }
        75% { transform: translate(15px, 5px); }
    }
    
    /* Main content styling */
    .main > div {
        background-color: rgba(15, 32, 39, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 255, 255, 0.1);
        border: 1px solid rgba(0, 255, 255, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 32, 39, 0.95) 0%, rgba(44, 83, 100, 0.95) 100%);
        border-right: 2px solid rgba(0, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] > div {
        background-color: transparent;
    }
    
    /* Headers with glow effect */
    h1, h2, h3 {
        color: #00fff5 !important;
        text-shadow: 0 0 20px rgba(0, 255, 245, 0.5);
        font-weight: 600 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Text styling */
    p, .stMarkdown, label {
        color: #e0f7ff !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
        background: linear-gradient(135deg, #00e5ff 0%, #00b8e6 100%);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: rgba(44, 83, 100, 0.5);
        color: #e0f7ff;
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 8px;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: rgba(0, 212, 255, 0.3);
    }
    
    .stSlider > div > div > div > div > div {
        background-color: #00d4ff;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(44, 83, 100, 0.3);
        border: 2px dashed rgba(0, 255, 255, 0.4);
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        background-color: rgba(44, 83, 100, 0.4);
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #00d4ff;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: rgba(44, 83, 100, 0.4);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 255, 0.2);
    }
    
    /* Image containers */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 255, 255, 0.2);
        border: 1px solid rgba(0, 255, 255, 0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(44, 83, 100, 0.4);
        border-radius: 8px;
        color: #00fff5 !important;
    }
    
    /* Checkbox and radio */
    .stCheckbox, .stRadio {
        color: #e0f7ff !important;
    }
    
    /* Video player */
    video {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 255, 255, 0.3);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 32, 39, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 212, 255, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 212, 255, 0.7);
    }
    </style>
    """, unsafe_allow_html=True)

# Title with satellite icon
st.title("🛰️ YOLO Satellite Object Detection System")
st.markdown("### Advanced AI-Powered Satellite Imagery Analysis")
st.markdown("Upload satellite images or video feeds to detect and classify objects using deep learning")

# Sidebar for settings
st.sidebar.header("⚙️ Detection Settings")

# Model loading
@st.cache_resource
def load_model(model_path):
    """Load the YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Check if model exists
MODEL_PATH = "yolo_model.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found!")
    st.info("Please make sure 'yolo_model.pt' is in the same directory as this script.")
    st.stop()

# Load model
with st.spinner("Loading YOLO model..."):
    model = load_model(MODEL_PATH)

if model is None:
    st.stop()

st.sidebar.success("✅ Model loaded successfully!")

# Settings
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence score for object detection"
)

iou_threshold = st.sidebar.slider(
    "NMS IoU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Lower IoU = more boxes, Higher IoU = fewer boxes"
)

# Input type selection
input_type = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video", "Webcam", "Satellite Map"],
    help="Choose your input source"
)

# Function to process image
def process_image(image, model, conf_threshold, iou_threshold):
    """Process image and return annotated image with detections"""
    try:
        # Convert PIL Image to RGB mode (handles all formats)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Double check - ensure we have exactly 3 channels
        if len(img_array.shape) == 2:  # Grayscale (should not happen after convert)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[-1] == 4:  # RGBA (should not happen after convert)
            img_array = img_array[:, :, :3]  # Take only RGB channels
        elif img_array.shape[-1] == 1:  # Single channel
            img_array = np.repeat(img_array, 3, axis=2)
        elif img_array.shape[-1] > 4:  # More than 4 channels
            img_array = img_array[:, :, :3]  # Take only first 3 channels
        
         # Run inference
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=300,
            show=False
        )
        # Get annotated image
        annotated_img = results[0].plot()
        
        # Convert BGR to RGB
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Get detection info
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            detections.append({
                'class': class_name,
                'confidence': confidence
            })
        
        return annotated_img, detections
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, []

# Function to process video
def process_video(video_path, model, conf_threshold):
    """Process video and return processed frames"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            show=False,
            verbose=False
        )
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Write frame
        out.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing: {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    return output_path

# Main content based on input type
if input_type == "Image":
    st.header("📸 Satellite Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload Satellite Imagery",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
        help="Supported formats: JPG, PNG, BMP, WebP, TIFF"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📡 Original Satellite Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Process button
        if st.button("🔍 Analyze Satellite Image", type="primary"):
            with st.spinner("Running AI detection analysis..."):
                # Process image
                annotated_img, detections = process_image(
                    image, model, confidence_threshold, iou_threshold
                )
                
                if annotated_img is not None:
                    with col2:
                        st.subheader("🎯 Detection Results")
                        st.image(annotated_img, use_container_width=True)
                    
                    # Display detection results
                    if detections:
                        st.success(f"✅ Detected {len(detections)} object(s)")
                        
                        # Create metrics row
                        st.subheader("📊 Detection Summary")
                        metric_cols = st.columns(min(len(detections), 4))
                        
                        for i, det in enumerate(detections[:4]):
                            with metric_cols[i]:
                                st.metric(
                                    label=det['class'],
                                    value=f"{det['confidence']:.1%}"
                                )
                        
                        # Detailed list
                        with st.expander("📋 Detailed Detection List"):
                            for i, det in enumerate(detections, 1):
                                st.write(f"{i}. **{det['class']}** - Confidence: {det['confidence']:.2%}")
                    else:
                        st.info("No objects detected with current confidence threshold. Try adjusting the threshold in the sidebar.")

elif input_type == "Video":
    st.header("🎥 Satellite Video Analysis")
    
    uploaded_video = st.file_uploader(
        "Upload Satellite Video Feed",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video is not None:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        video_path = tfile.name
        
        # Display original video
        st.subheader("📡 Original Video Feed")
        st.video(video_path)
        
        # Process button
        if st.button("🔍 Process Video Feed", type="primary"):
            with st.spinner("Processing video feed... This may take several minutes."):
                output_path = process_video(
                    video_path, model, confidence_threshold
                )
                
                if output_path:
                    st.success("✅ Video analysis complete!")
                    st.subheader("🎯 Analyzed Video with Detections")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Analyzed Video",
                            data=f,
                            file_name="satellite_detection_results.mp4",
                            mime="video/mp4"
                        )
                    
                    # Cleanup
                    try:
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                    except PermissionError:
                        pass
        
        # Cleanup input video
        try:
            if os.path.exists(video_path):
                os.unlink(video_path)
        except PermissionError:
            pass

elif input_type == "Webcam":
    st.header("📹 Live Feed Detection")
    st.info("Enable webcam to start real-time object detection")
    
    # Webcam settings
    run_webcam = st.checkbox("Start Live Detection")
    
    FRAME_WINDOW = st.image([])
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Please check your camera connection.")
        else:
            st.success("✅ Live feed active")
            
            while run_webcam:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to grab frame")
                    break
                
                # Run inference
                results = model.predict(
                    source=frame,
                    conf=confidence_threshold,
                    show=False,
                    verbose=False
                )
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                FRAME_WINDOW.image(annotated_frame)
            
            cap.release()
            st.info("Live feed stopped")

elif input_type == "Satellite Map":
    st.header("🗺️ Interactive Satellite Map")
    st.markdown("Navigate the map to find your area of interest, then capture a screenshot to analyze")
    
    # Location input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        location_input = st.text_input(
            "Search Location (e.g., 'Cairo, Egypt' or 'Times Square, NY')",
            value="Cairo, Egypt",
            help="Enter a location to center the map"
        )
    
    with col2:
        zoom_level = st.slider("Zoom Level", 1, 20, 15, help="Adjust map zoom level")
    
    # Map type selection
    map_type = st.radio(
        "Map Type",
        ["Satellite", "Hybrid (Satellite + Labels)", "Terrain"],
        horizontal=True
    )
    
    # Map the selection to Google Maps type
    map_type_param = {
        "Satellite": "satellite",
        "Hybrid (Satellite + Labels)": "hybrid",
        "Terrain": "terrain"
    }[map_type]
    
    # Instructions
    with st.expander("📋 How to Use", expanded=True):
        st.markdown("""
        1. **Search** for your location or navigate the map manually
        2. **Adjust zoom** to get the desired view
        3. **Take a screenshot** of the map area using your device's screenshot tool
        4. **Upload** the screenshot below for AI analysis
        
        **Screenshot shortcuts:**
        - Windows: `Win + Shift + S` or `PrtScn`
        - Mac: `Cmd + Shift + 4`
        - Linux: `PrtScn` or `Shift + PrtScn`
        """)
    
    # Create the interactive map using iframe
    st.markdown("### 🌍 Interactive Google Maps")
    
    # Google Maps embed URL
    map_url = f"https://www.google.com/maps/embed/v1/place?key=YOUR_API_KEY&q={location_input}&zoom={zoom_level}&maptype={map_type_param}"
    
    # Note: Since we can't use actual Google Maps API without a key, we'll create an HTML map using OpenStreetMap
    # which provides free satellite imagery
    
    map_html = f"""
    <div style="width: 100%; height: 600px; border-radius: 12px; overflow: hidden; border: 2px solid rgba(0, 255, 255, 0.3);">
        <iframe 
            width="100%" 
            height="100%" 
            frameborder="0" 
            scrolling="no" 
            marginheight="0" 
            marginwidth="0" 
            src="https://www.google.com/maps?q={location_input.replace(' ', '+')}&t={map_type_param[0]}&z={zoom_level}&output=embed"
            style="border-radius: 12px;">
        </iframe>
    </div>
    """
    
    st.markdown(map_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Screenshot upload section
    st.markdown("### 📸 Upload Map Screenshot for Analysis")
    
    screenshot_file = st.file_uploader(
        "Upload your map screenshot",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a screenshot of the satellite map above",
        key="map_screenshot"
    )
    
    if screenshot_file is not None:
        # Display uploaded screenshot
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📡 Map Screenshot")
            screenshot_image = Image.open(screenshot_file)
            st.image(screenshot_image, use_container_width=True)
        
        # Process button
        if st.button("🔍 Analyze Map Screenshot", type="primary", key="analyze_map"):
            with st.spinner("Running AI detection analysis..."):
                # Process image
                annotated_img, detections = process_image(
                    screenshot_image, model, confidence_threshold
                )
                
                if annotated_img is not None:
                    with col2:
                        st.subheader("🎯 Detection Results")
                        st.image(annotated_img, use_container_width=True)
                    
                    # Display detection results
                    if detections:
                        st.success(f"✅ Detected {len(detections)} object(s)")
                        
                        # Create metrics row
                        st.subheader("📊 Detection Summary")
                        metric_cols = st.columns(min(len(detections), 4))
                        
                        for i, det in enumerate(detections[:4]):
                            with metric_cols[i]:
                                st.metric(
                                    label=det['class'],
                                    value=f"{det['confidence']:.1%}"
                                )
                        
                        # Detailed list
                        with st.expander("📋 Detailed Detection List"):
                            for i, det in enumerate(detections, 1):
                                st.write(f"{i}. **{det['class']}** - Confidence: {det['confidence']:.2%}")
                    else:
                        st.info("No objects detected with current confidence threshold. Try adjusting the threshold in the sidebar.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 System Information")
st.sidebar.info(
    "**YOLO Satellite Detection System**\n\n"
    "This system uses YOLOv8 deep learning architecture for real-time object detection "
    "in satellite imagery. Adjust detection parameters to optimize results for different "
    "satellite image conditions and object types."
)

# Display model info
with st.sidebar.expander("🤖 Model Details"):
    if model:
        st.write(f"**Architecture:** {type(model).__name__}")
        st.write(f"**Total Classes:** {len(model.names)}")
        st.write(f"**Confidence:** {confidence_threshold:.0%}")
        if hasattr(model, 'names'):
            st.write("**Detectable Objects:**")
            for idx, name in model.names.items():
                st.write(f"• {name}")

# Performance tips
with st.sidebar.expander("💡 Tips for Best Results"):
    st.markdown("""
    - Use high-resolution satellite imagery
    - Adjust confidence threshold for precision
    - Process videos in smaller segments for speed
    - Ensure good image quality and clarity

    """)
