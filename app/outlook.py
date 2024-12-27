import streamlit as st

def show_outlook():
    st.title("Outlook")
    st.image("Images/banner2.png", use_column_width="auto")

    st.write("Select one of three areas of improvement of the model to diagnose lung diseases using X-ray images:")

   # Expander for "Data Limitations and Improvement of Labeling"
    with st.expander("**Data Limitations and Improvement of Labeling**"):
        st.write("""
            The outlook for deploying the algorithm highlights key areas for improvement to enhance its clinical effectiveness. 
            Current limitations in data, particularly with **lung opacity classification**, affect the model's accuracy. Lung opacity is sometimes 
            misclassified as normal due to similar pixel distributions of these classes. Improving labeling, possibly through **expert 
            validation or self-supervised learning**, could resolve this issue. Additionally, extending the model with **multimodal data** 
            (e.g., patient history, symptoms) could boost diagnostic accuracy.
        """)

    # Expander for "Model Improvements"
    with st.expander("**Model Improvements**"):
        st.write("""
            In terms of model improvements, exploring alternative base models and **fine-tuning approaches** (e.g., unfreezing layers, changing kernel sizes, 
            or adding layers) could enhance performance. Experimenting with **ensemble methods** could also yield more robust predictions. Another consideration 
            is whether to **remove the lung opacity class** altogether to simplify the model.
        """)

    # Expander for "Extending the Labels to Include Further Diseases"
    with st.expander("**Extending the Labels to Include Further Diseases**"):
        st.write("""
            Looking ahead, extending the model to detect **more lung diseases** (such as lung cancer, heart failure, tuberculosis, sarcoidosis, etc.) 
            would greatly increase its clinical value.
            This could involve **collecting additional data** or applying self-supervised learning to the current data. Addressing these improvements will 
            be crucial for deploying the model in real-world healthcare settings.
        """)
