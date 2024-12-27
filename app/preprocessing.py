import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import random
import tensorflow as tf

def show_preprocessing():
    # Give a title to the page
    st.title("Preprocessing")
    # st.subheader("Preprocessing steps")
    st.write("""
    We explored several preprocessing options including masking, normalization, standardization,
    resizing and augmentation.
    """)
    st.image('Images/Preprocessing.png', use_column_width=True, 
             caption ="Representation of preproseccing options applied to an example image of each class")

    st.write("""
    Based on model performance and interpretability of the predicted results, we
    decided to apply the following for all models:             
    - Image size and number of channels were adapted to each model
    - Unmasked images
    - Data augmentation (details below)
    """)
    #st.markdown("## Data Augmentation")

    data = {
        "Augmentation Technique": [
        "Rotation", "Horizontal Flip", "Vertical Flip", 
        "Scaling (Shifting)", "Zooming", 
        "Brightness Adjustment", "Contrast Adjustment"],
        "Description": [
        "Randomly rotates the image by up to 10 degrees to introduce variety in orientation.",
        "Flips the image horizontally with a 50% chance.",
        "Flips the image vertically with a 50% chance.",
        "Shifts the image dimensions by up to 10% to simulate slight variations in position.",
        "Randomly zooms in or out of the image by up to 10%.",
        "Randomly changes the brightness of the image.",
        "Randomly alters the contrast of the image to enhance visual features."]
    }
    
    df = pd.DataFrame(data)

    html_table = df.to_html(index=False, classes='table table-striped')

    st.markdown("### Data Augmentation Techniques")

    with st.expander("View Data Augmentation Techniques", expanded=False):
        st.markdown(html_table, unsafe_allow_html=True)

    def augment_weighted_image(weighted_image):

        if tf.random.uniform([]) > 0.5:
            weighted_image = tf.image.flip_left_right(weighted_image)

        if tf.random.uniform([]) > 0.5:
            weighted_image = tf.image.flip_up_down(weighted_image)

        angles = tf.random.uniform([], minval=-10, maxval=10, dtype=tf.float32) * (3.14159 / 180)
        k = tf.cast(angles / (3.14159 / 2), tf.int32) % 4
        weighted_image = tf.image.rot90(weighted_image, k=k)

        height = tf.shape(weighted_image)[0]
        width = tf.shape(weighted_image)[1]

        max_width_shift = tf.cast(tf.cast(width, tf.float32) * 0.1, tf.int32)
        max_height_shift = tf.cast(tf.cast(height, tf.float32) * 0.1, tf.int32)
        width_shift = tf.random.uniform([], -max_width_shift, max_width_shift, dtype=tf.int32)
        height_shift = tf.random.uniform([], -max_height_shift, max_height_shift, dtype=tf.int32)

        padding_height = tf.maximum(0, height_shift)
        padding_width = tf.maximum(0, width_shift)
        weighted_image = tf.image.pad_to_bounding_box(
            weighted_image, padding_height, padding_width, height + tf.abs(height_shift), width + tf.abs(width_shift)
        )

        cropped_height_offset = tf.maximum(0, -height_shift)
        cropped_width_offset = tf.maximum(0, -width_shift)
        weighted_image = tf.image.crop_to_bounding_box(weighted_image, cropped_height_offset, cropped_width_offset, height, width)

        zoom_factor = tf.random.uniform([], 0.9, 1.1)
        new_height = tf.cast(tf.cast(height, tf.float32) * zoom_factor, tf.int32)
        new_width = tf.cast(tf.cast(width, tf.float32) * zoom_factor, tf.int32)
        weighted_image = tf.image.resize(weighted_image, [new_height, new_width])

        weighted_image = tf.image.resize_with_crop_or_pad(weighted_image, target_height=height, target_width=width)

        weighted_image = tf.image.resize(weighted_image, [256, 256])

        return weighted_image
    
    def load_and_preprocess_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [256, 256])
        image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32) / 255.0

        weighted_image = augment_weighted_image(image)

        return weighted_image
    
    def load_and_preprocess_image_original(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [256, 256])
        image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32) / 255.0

        weighted_image = image

        return weighted_image
    
    image_paths = [
        r"Images\\COVID-31.png",
        r"Images\\Normal-21.png",
        r"Images\\Lung_Opacity-3.png",
        r"Images\\Viral_Pneumonia-9.png" ]

    if st.button("Apply Random Data Augmentation"):
        random_image_path = random.choice(image_paths)
    
        original_image = load_and_preprocess_image_original(random_image_path)
    
        augmented_image = load_and_preprocess_image(random_image_path)

        #st.subheader(f"Original and Augmented Image")
        col1, col2 = st.columns(2)

        with col1:
            st.image(original_image.numpy(), caption="Original Image", use_column_width=True)
    
        with col2:
            st.image(augmented_image.numpy(), caption="Augmented Image", use_column_width=True)

    # st.markdown("## Masking")

    # st.image(r"Images\Masking.png", 
    #                   caption="Grad-CAM images for masked (left) and unmasked (right) versions of ResNet50-SVM, EfficientNetB1 and CNN 1.1 models", 
    #                   use_column_width=True)

