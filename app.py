# Importing the necessary libraries
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Setting page configuration
st.set_page_config(page_title="Aerial Object Classification & Detection",
    layout="wide")

# Loading models using cache_resource to avoid reloading
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("../models/cnn_model_final.h5")
    tl = tf.keras.models.load_model("../models/tl_model_best.h5")
    yolo = YOLO("../models/weights/best.pt")
    return cnn, tl, yolo

cnn_model, tl_model, yolo_model = load_models()

# Setting app title
st.title("Aerial Object Classification & Detection")

# Creating tabs
tab1, tab2, tab3 = st.tabs(["CNN", "Transfer Learning", "YOLO (Best Model)"])

# CNN tab
with tab1:
    st.header("CNN Model")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="cnn")

    if uploaded_file is not None:
        try:
            # Reading and displaying image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=400)

            with st.spinner("Processing with CNN model..."):
                # Preprocessing image for CNN
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Making prediction
                prediction = cnn_model.predict(img_array)[0][0]

            # Interpreting result
            label = "Drone" if prediction > 0.5 else "Bird"
            confidence = prediction if prediction > 0.5 else 1 - prediction

            # Displaying result
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"Error processing image: {e}")

# Transfer Learning tab
with tab2:
    st.header("Transfer Learning Model")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="tl")

    if uploaded_file is not None:
        try:
            # Reading and displaying image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=400)

            with st.spinner("Processing with Transfer Learning model..."):
                # Preprocessing image for TL model
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                # Making prediction
                prediction = tl_model.predict(img_array)[0][0]

            # Interpreting result
            label = "Drone" if prediction > 0.5 else "Bird"
            confidence = prediction if prediction > 0.5 else 1 - prediction

            # Displaying result
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"Error processing image: {e}")

# YOLO tab
with tab3:
    st.header("YOLO Model (Best Model)")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="yolo")

    # Setting confidence threshold slider
    conf_threshold = st.slider("Confidence Threshold:", 0.1, 1.0, 0.5)

    if uploaded_file is not None:
        try:
            # Reading and displaying image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=400)

            with st.spinner("Running YOLO detection..."):
                # Converting image to array for YOLO
                img_array = np.array(image)

                # Running YOLO prediction with confidence threshold
                results = yolo_model.predict(img_array, conf=conf_threshold)

                # Drawing bounding boxes on image
                annotated_image = results[0].plot()

                # Converting BGR to RGB for display
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Displaying annotated image
            st.image(annotated_image, caption="Detection Result", width=400)

            # Extracting and displaying detected objects
            boxes = results[0].boxes
            names = results[0].names

            if boxes is not None and len(boxes) > 0:
                st.subheader("Detected Objects:")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"{names[cls_id]} ({conf:.2f})")
            else:
                st.warning("No objects detected.")

        except Exception as e:
            st.error(f"Error processing image: {e}")