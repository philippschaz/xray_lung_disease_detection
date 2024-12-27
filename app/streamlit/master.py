# This is the master of the Streamlit website. 
# Work on each subfile and then execute only this file in the Terminal to build the Streamlit.


# Define paths
###
# Philipp:
# cd "D:\Google Drive\Data Science\Team Project X-Rays\Streamlit"
# cd "G:\Meine Ablage\Data Science\Team Project X-Rays\Streamlit"

# Install packages
###
# pip install -r requirements.txt

# Execute Streamlit in Terminal
###
# streamlit run master.py

# Load packages and subfiles
import streamlit as st
import home
import visualization
import models
import interpretability
import demonstration
import outlook
import helpers
import about
import preprocessing


# Sidebar menu for navigation
st.sidebar.title("Viral Pneumonia & COVID-19 Detection")
menu = st.sidebar.radio("Menu", ["Home", 
                                 "Visualization", 
                                 "Preprocessing",
                                 "Deep Learning Models", 
                                 "Interpretability", 
                                 "Demonstration", 
                                 "Outlook",
                                 "About"])

# Call the respective page based on the user's selection
if menu == "Home":
    home.show_home()
elif menu == "Visualization":
    visualization.show_data()
elif menu == "Preprocessing":
    preprocessing.show_preprocessing()
elif menu == "Deep Learning Models":
    models.show_models()
elif menu == "Interpretability":
    interpretability.show_interpretability()
elif menu == "Demonstration":
    demonstration.show_demonstration()
elif menu == "Outlook":
    outlook.show_outlook()
elif menu == "About":
    about.show_about()
