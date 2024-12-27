import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import random
import tensorflow as tf

def show_models():
    st.title("Deep Learning Models")

    st.write("""
        The modelling section provides an overview of the two deep learning approaches developed 
        for detecting Covid-19 and viral pneumonia from chest X-ray images. It highlights the 
        architecture of both the custom CNN model and the transfer learning model based on EfficientNetB1. 
        Additionally, the training process is discussed, showcasing the loss and accuracy functions used 
        to evaluate model performance during training.
    """)

    st.markdown("## Architecture")
    
    model_choice = st.selectbox("Select a model", ["CNN 1.1", "TL Model (EfficientNetB1)"],
                                key="architecture_model_choice")
    
    with st.expander("View Architecture", expanded=False):
        if model_choice == "CNN 1.1":
            st.image(r"Images\CNN_architecture.png", 
                      caption="CNN 1.1 Architecture", use_column_width=True)
        else:
            st.image(r"Images\EfficientNetB1_architecture.png", 
                      caption="TL Model (EfficientNetB1) Architecture", use_column_width=True)
    
    st.markdown("## Loss and Training History")
    
    model_choice = st.selectbox("Select a model", ["CNN 1.1", "TL Model (EfficientNetB1)"],
                                key="loss_training_model_choice")
    
    with st.expander("View Loss and Training History", expanded=True):
        if model_choice == "CNN 1.1":
            st.image(r"Images\cnn_training.png", 
                      caption="CNN 1.1 Loss and Training History", use_column_width=True)
        else:
            st.image(r"Images\EfficientNet_training.png", 
                      caption="TL Model (EfficientNetB1) Loss and Training History", use_column_width=True)
    
    st.markdown("## Evaluation")

    st.write("""
        In the evaluation section, key metrics such as accuracy, F1 scores, and recall are analyzed 
        to determine the effectiveness of the two deep learning models developed for detecting Covid-19 
        and viral pneumonia from chest X-ray images. Additionally, the execution time for each model is 
        discussed to provide insights into their efficiency during inference. The models presented here 
        are marked in grey in the accompanying table, while the other models represent prior versions 
        that differ in either data preprocessing, model architecture, or both. Confusion matrices are also 
        provided to visualize the models' classification performance and identify areas of misclassification.
    """)

    st.markdown("### Model Performance Metrics")

    st.image(r"Images\overall_results.png", 
                      caption="Overall results", use_column_width=True)
    
    st.markdown("### Confusion Matrix")
    
    model_choice = st.selectbox("Select a model", ["CNN 1.1", "TL Model (EfficientNetB1)"],
                                key="confusion_matrix_model_choice")
    
    with st.expander("View Confusion Matrix", expanded=True):
        if model_choice == "CNN 1.1":
            st.image(r"Images\confusion_matrix_cnn.png", 
                      caption="CNN 1.1 Confusion Matrix", use_column_width=True)
        else:
            st.image(r"Images\confusion_matrix_EfficientNetB1.png", 
                      caption="TL Model (EfficientNetB1) Confusion Matrix", use_column_width=True)


            
