import streamlit as st
from PIL import Image
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device

@st.cache_resource
def load_model(weights, device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    model = AutoShape(model)
    return model

def main():
    st.title("YOLOv9 Smart Cane Interface")
    st.write("Object Detection for the Visually Impaired Project")

    # Sidebar options
    st.sidebar.header("Settings")
    weights = st.sidebar.text_input("Model Weights Path", str(ROOT / 'yolov9e.pt'))
    conf_thres = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    iou_thres = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45)

    # Load model
    try:
        model = load_model(weights)
        model.conf = conf_thres
        model.iou = iou_thres
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.error(f"Failed to load model from {weights}. Please check the path.")
        return

    # Tabs for different modes
    tab1, tab2 = st.tabs(["Image Upload", "About"])

    with tab1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])

        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button('Detect Objects'):
                with st.spinner('Running detection...'):
                    # Inference
                    results = model(image)

                    # Render
                    results.render()  # updates results.ims with boxes
                    result_image = Image.fromarray(results.ims[0])

                    with col2:
                        st.image(result_image, caption='Detections', use_column_width=True)

                    # Show dataframe
                    st.write("### Detected Objects List")
                    df = results.pandas().xyxy[0]
                    st.dataframe(df)

                    # Simple summary
                    counts = df['name'].value_counts()
                    st.write("### Summary")
                    for name, count in counts.items():
                        st.write(f"- {count} {name}(s)")

    with tab2:
        st.header("About Ishare Smart Cane")
        st.write("""
        The "Ishare" project intends to create a working prototype of a smart cane to help blind people safely and freely navigate their environment.
        The smart cane uses YOLOv9 for object identification.
        """)
        st.write("This interface allows testing the object detection model.")

if __name__ == "__main__":
    main()
