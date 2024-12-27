Viral Pneumonia and COVID-19 Detection from Chest X-Ray Images

Welcome to the repository for our Viral Pneumonia and COVID-19 detection project! This collaborative data science project leverages state-of-the-art classic machine learning (XGboost, SVM) and deep learning techniques, interpretable machine learning models, and GPU-accelerated image processing to classify chest X-ray images with high precision. Here's everything you need to dive into the code, data, and results.

Authors: Dr. Philipp Schaz, Dr. Anne Ast and Tanja Schröder

🔗 Repository Structure
app/

The deployment-ready Streamlit application for real-time predictions and visualization.

    master.py: Main entry point for the app.
    Features:
        Upload chest X-ray images.
        Get predictions for Viral Pneumonia, COVID-19, Lung Opacity and Normal.
        Visualize Grad-CAM heatmaps for interpretability.

notebooks/

A collection of Jupyter notebooks documenting the entire modeling lifecycle, from exploratory data analysis (EDA) to training high-performance classic ML and deep learning models. These were developed using Google Colab for large-scale image processing with GPU acceleration.


output/

Contains all the outputs of the project, including:

    models/: Serialized models ready for deployment.
    reports/: Comprehensive analysis and insights into the modeling pipeline.

src/

Modularized Python scripts for preprocessing, visualization, and reusable components.

    Subdirectories:
        preprocessing/: All preprocessing logic, including image resizing, normalization and feature extraction.
        data_viz/: Custom scripts for generating visualizations and data exploration.

data/

The backbone of the project, this directory contains all unprocessed chest X-ray images.


📊 Project Overview
Objective

To develop a machine learning model that accurately predicts lung diseases from chest X-rays.

Our goal is to maximize predictive performance, minimize false negatives, and ensure model interpretability for practical clinical deployment.
Models

We explored a variety of deep learning architectures, with the standout performer being:

    ResNet50 Hybrid-SVM:
        Accuracy: 91%
        F1-Scores:
            Normal: 92%
            Viral Pneumonia: 95%
            COVID-19: 92%
        Execution Time: 12 minutes

This model strikes a balance between accuracy and computational efficiency, making it suitable for real-world clinical applications.

🚀 Getting Started
Setup

    Clone the repository:

git clone https://github.com/your-username/viral-pneumonia-covid19-xray.git
cd viral-pneumonia-covid19-xray

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

    streamlit run app/app.py

Requirements

    Python 3.9+
    TensorFlow, PyTorch
    Streamlit
    OpenCV, Matplotlib, Seaborn

📈 Key Features

    Data Exploration: EDA to uncover trends and distribution of labels.
    Image Preprocessing: Augmentation to improve generalization on unseen data.
    Deep Learning Models: Transfer learning with VGG16, ResNet50, and CNN architectures.
    Interpretability: Grad-CAM visualizations to ensure the models focus on lung regions.
    Streamlit App: User-friendly interface for real-time predictions and heatmap visualizations.

🔍 Insights

    Masked images did not enhance interpretability or performance.
    Overfitting was mitigated via image augmentation.
    The ResNet50 Hybrid-SVM model delivers state-of-the-art performance with a quick runtime.

💡 Outlook

Future improvements could include:

    Extending the dataset to enhance model generalization.
    Exploring hybrid explainability techniques for finer-grained insights.
    Deploying the model as a cloud-based API for broader accessibility.

📜 License

This project is licensed under the MIT License. See LICENSE for details.

Contributions are welcome! 🛠️ If you have ideas for improving the project, feel free to open an issue or submit a pull request.
