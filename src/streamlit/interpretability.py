import streamlit as st

def show_interpretability():
    st.title("Interpretability")
    st.write("""
        In the interpretability section, we aim to understand how the models make decisions and whether they 
        focus on relevant areas of the X-ray images. To achieve this, we use Grad-CAM to visualize which parts 
        of the image influenced the model's predictions. For each of the four images, we display the original 
        X-ray, a heatmap highlighting important regions, and the corresponding Grad-CAM visualization.
    """)
    # Optionally, you can add Grad-CAM visualizations here
    # st.image('path_to_gradcam_image')

    st.title("Grad-CAM")
    
    model_choice = st.selectbox("Select a model", ["CNN 1.1", "TL Model (EfficientNetB1)"],
                                key="gradcam_model_choice")
    
    with st.expander("View Grad_CAM Visualization", expanded=True):
        if model_choice == "CNN 1.1":
            st.image(r"Images\GradCam-CNN.png", 
                      caption="CNN 1.1 Grad-CAM", use_column_width=True)
        else:
            st.image(r"Images\GradCam-EfficientNet.png", 
                      caption="TL Model (EfficientNetB1) Grad-CAM", use_column_width=True)
            
    st.markdown("## Masking")

    st.write("""
        In our project, we applied masking by weighting images with binary masks to guide the model's focus on the lung area. However, the results 
        indicated that this approach did not improve the model's performance. Even with the masks applied, the model continued to identify areas within 
        the masked region as important. This finding suggests that in our project, masking did not enhance the model's ability to focus on the lung area.
    """)

    st.image(r"Images\Masking.png", 
                      caption="Grad-CAM images for masked (left) and unmasked (right) versions of ResNet50-SVM, EfficientNetB1 and CNN 1.1 models", 
                      use_column_width=True)