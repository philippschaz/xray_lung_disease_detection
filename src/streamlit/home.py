# Philipp:
# cd "D:\Google Drive\Data Science\Team Project X-Rays\Streamlit"

# Execute Streamlit in Terminal
# streamlit run master.py


import streamlit as st

def show_home():
    st.title("Viral Pneumonia and COVID-19 Detection using Chest X-ray Images")
    st.subheader("Introduction")
    st.write("""
        The COVID-19 pandemic highlighted the need for rapid and accurate diagnostic tools. Machine learning models applied to chest X-rays 
        have the potential in detecting lung diseases like COVID-19 and Viral Pneumonia. This project explores various deep learning 
        algorithms to identify the most effective model for diagnosing these diseases using X-ray images.
                   
        **Aim**            
             
        The project presents different deep learning models for detecting COVID-19 and Viral Pneumonia from chest X-rays, with a focus on 
        maximizing the F1 Score, particularly emphasizing recall to avoid false negatives, which could have severe consequences in a clinical setting.
        Additionally, the project seeks to minimize overfitting by using techniques like data augmentation and regularization, ensuring the model generalizes well to unseen data. 
        This, combined with interpretability, aims to make the model accurate and reliable for potential deployment in clinical environments.
             

        **Data**
             
        The data is sourced from the 
        [Kaggle COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) 
        with 21,165 X-ray images sourced from medical professionals around the globe. This data features chest X-ray images that are
        categorized into Normal (10,192 images), COVID-19 (3,616 images), Lung Opacity (6,012 images), and Viral Pneumonia (1,345 images) classes.
             
        **Sample Images**
    """)

   
    # Dropdown menu for selecting class and displaying corresponding X-ray image
    #st.subheader("Sample Chest X-ray Images")
    class_option = st.selectbox("Choose a class to view a sample image:", 
                                ["Normal", "COVID-19", "Lung Opacity", "Viral Pneumonia"])

    # Sample images for each class (replace with the actual image paths)
    if class_option == "Normal":
        st.image("Images/Normal-47.png", caption="Normal Chest X-ray", use_column_width=True)
    elif class_option == "COVID-19":
        st.image("Images/COVID-13.png", caption="COVID-19 Chest X-ray", use_column_width=True)
    elif class_option == "Lung Opacity":
        st.image("Images/Lung_Opacity-84.png", caption="Lung Opacity Chest X-ray", use_column_width=True)
    elif class_option == "Viral Pneumonia":
        st.image("Images/Viral Pneumonia-48.png", caption="Viral Pneumonia Chest X-ray", use_column_width=True)
