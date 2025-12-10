# Computer Vision Streamlit Application

A complete Streamlit web application for computer vision that supports multiple models and input types.

## Features

### Supported Models

1. **YOLO Object Detection** (`yolo_model.pt`)
   - Detects and localizes multiple objects in images, videos, and webcam streams
   - Displays bounding boxes with class labels and confidence scores
   - Configurable confidence and IOU thresholds

2. **Baseline Classification** (`model_baseline.keras`)
   - Image classification using a baseline CNN model
   - Supports image input only
   - Shows top-3 predictions with probabilities

3. **VGG16 Classification** (`model_vgg16.keras`)
   - Image classification using VGG16 architecture
   - Supports image input only
   - Shows top-3 predictions with probabilities

### Supported Input Types

- 📸 **Image Upload**: Upload JPG, JPEG, PNG, or BMP images
- 🎥 **Video Upload**: Upload MP4, AVI, MOV, or MKV videos (YOLO only)
- 📹 **Webcam**: Live webcam stream processing (YOLO only)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster processing

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model files**:
   Place the following model files in the same directory as `app.py`:
   - `yolo_model.pt` - YOLO object detection model
   - `model_baseline.keras` - Baseline classification model
   - `model_vgg16.keras` - VGG16 classification model

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Application

1. **Select a Model**: Use the sidebar to choose between YOLO Object Detection, Baseline Classification, or VGG16 Classification.

2. **Configure Settings** (YOLO only):
   - Adjust the confidence threshold (default: 0.25)
   - Adjust the IOU threshold (default: 0.45)

3. **Choose Input Type**:
   - **Image**: Upload an image file and click the detection/classification button
   - **Video**: Upload a video file and click "Process Video" (YOLO only)
   - **Webcam**: Check the "Start Webcam" box to begin live processing (YOLO only)

4. **View Results**:
   - For YOLO: See annotated images/videos with bounding boxes and a list of detected objects
   - For Classification: See top-3 predictions with confidence percentages

## Model Requirements

### YOLO Model
- Format: `.pt` (PyTorch)
- Loaded using Ultralytics YOLO library
- Should be trained for object detection tasks

### Keras Models
- Format: `.keras` (Keras 3.0+) or `.h5` (legacy)
- Input shape: 224x224x3 (RGB images)
- Output: Softmax probabilities for 10 classes (default: CIFAR-10)

## Customization

### Changing Class Names

Edit the class name lists in `app.py`:

```python
BASELINE_CLASS_NAMES = [
    "class1", "class2", "class3", ...
]

VGG16_CLASS_NAMES = [
    "class1", "class2", "class3", ...
]
```

### Adjusting Input Size

Modify the `preprocess_image_for_classification` function:

```python
def preprocess_image_for_classification(image, target_size=(224, 224)):
    # Change target_size to match your model's input requirements
```

## Troubleshooting

### Model Loading Errors
- Ensure model files are in the correct directory
- Check that model files are not corrupted
- Verify TensorFlow/Keras version compatibility

### Webcam Issues
- Grant camera permissions to your browser
- Check that no other application is using the webcam
- Try restarting the Streamlit application

### Video Processing Slow
- Consider using a GPU for faster processing
- Reduce video resolution or frame rate
- Adjust confidence threshold to reduce detections

## Technical Details

### Dependencies
- **streamlit**: Web application framework
- **opencv-python**: Image and video processing
- **numpy**: Numerical operations
- **Pillow**: Image handling
- **tensorflow**: Deep learning framework for Keras models
- **ultralytics**: YOLO implementation

### Performance Optimization
- Models are cached using `@st.cache_resource` decorator
- Only loaded once per session
- Video processing shows progress bar
- Temporary files are cleaned up automatically

## License

This application is provided as-is for educational and research purposes.

## Support

For issues or questions, please refer to the documentation of the respective libraries:
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [TensorFlow/Keras](https://www.tensorflow.org/guide/keras)
