# 1. preparation
###
# Define paths

# Philipp:
# cd "D:\Google Drive\Data Science\Team Project X-Rays\Streamlit"

# Run code in Streamlit
# streamlit run streamlit_master.py

# import Streamlit 
import streamlit as st


# 2. Streamlit website
###
# Sidebar menu for navigation
st.sidebar.title("Viral Pneumonia & COVID-19 Detection")
menu = st.sidebar.radio("Menu", ["Home", 
                                 "Data", 
                                 "Deep Learning Models", 
                                 "Interpretability", 
                                 "Demonstration", 
                                 "Outlook"])

# Introduction & Aim page
if menu == "Home":
    st.title("Viral Pneumonia and COVID-19 Detection using Chest X-ray Images")
    st.subheader("Introduction & Aim")
    st.write("""
        The aim of this project is to explore deep learning models for detecting Viral Pneumonia and COVID-19 from chest X-ray images. 
        This is a crucial task for aiding in the early detection and diagnosis of respiratory diseases, especially during the COVID-19 pandemic. 
        Our goal is to find a model that achieves high accuracy, avoids false negatives, and offers interpretability to ensure clinical reliability. 
        Various deep learning and machine learning models have been trained and evaluated, with a focus on performance and applicability in clinical settings.
    """)


# Data page
elif menu == "Data":
    st.title("Data")
    st.write("""
        The dataset consists of chest X-ray images categorized into four classes: 
        - Normal 
        - Viral Pneumonia 
        - Lung Opacity
        - COVID-19.
        
        The data has been sourced from public repositories and preprocessed for training deep learning models.
    """)
    # You can display a sample of images or statistics from the dataset
    # st.image('path_to_image')
    # st.dataframe('sample_dataframe')

# Deep Learning Models page
elif menu == "Deep Learning Models":
    st.title("Deep Learning Models")
    st.write("""
        For this project, we explored several deep learning architectures, including:
        - Convolutional Neural Networks (CNN)
        - Transfer Learning using pre-trained models like VGG16, ResNet50, etc.
        

    """)

# Interpretability page
elif menu == "Interpretability":
    st.title("Interpretability")
    st.write("""
        Interpretability is crucial in medical applications. 
        For this, we applied techniques like Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions of the X-ray that influenced the model's predictions.
    """)
    # You could provide a visualization of interpretability maps here
    # st.image('path_to_gradcam_image')

# Demonstration page
elif menu == "Demonstration":
    st.title("Demonstration")
    st.write("Upload a chest X-ray image to see the model's prediction.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")
    
    if uploaded_file is not None:
        # Placeholder code for model prediction (to be replaced with actual model)
        st.image(uploaded_file, caption="Uploaded X-ray image", use_column_width=True)
        st.write("Predicting...")
        # Add code here to load your model and predict the result on the uploaded image
        # For example:
        # model = load_model('your_model.h5')
        # result = model.predict(process_image(uploaded_file))
        # st.write(f"Prediction: {result}")

# Outlook page
elif menu == "Outlook":
    st.title("Outlook")
    st.write("""
        Moving forward, there are several areas to explore:
        1. Enhancing the model's accuracy through additional data and techniques.
        2. Collaborating with healthcare professionals for real-world testing.
        3. Deploying the system in a cloud-based application for easier access.
    """)