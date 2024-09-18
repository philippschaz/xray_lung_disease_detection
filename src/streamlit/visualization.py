import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go


def show_data():
    # Define function to load a data frame and cache it
    @st.cache_data()
    def load_data(filepath):
        data = pd.read_excel(filepath)
        return data

    # Give a title to the page
    st.title("Visualization")
    st.write("""
        The dataset consists of chest X-ray images (299 x 299 pixel) and
        corresponding lung masks (256 x 256 pixel), both in grayscale. The images are of high quality, 
        as all files are readable and correctly labeled, with no corrupted
        or mis-sized images and only a small number of duplications.
        Data is categorized into four classes and collected
        from eight public sources.
    """)

    st.subheader("Exploratory data analysis")
    # Show bar graph on data composition (checkbox)
    with st.expander("**Composition of the dataset**"):
    # if st.checkbox('**Composition of the dataset**'):
        # Import the dataframe
        df = load_data('Dataframes/df_stat_Anne.xlsx')
        st.write("""
        Analyzing the composition of the dataset revealed an imbalanced class distribution. 
        Whereas normal normal and lung opacity cases are over represented, COVID and viral pneumonia cases
        represented by a smaller number of images. In addition, we observed a bias towards
        a particular source for each of the four classes.
    """)

        # Mapping of URLs to shorter labels (S1, S2, etc.)
        url_mapping = {
            df['URL'].unique()[0]: 'S1',
            df['URL'].unique()[1]: 'S2',
            df['URL'].unique()[2]: 'S3',
            df['URL'].unique()[3]: 'S4',
            df['URL'].unique()[4]: 'S5',
            df['URL'].unique()[5]: 'S6',
            df['URL'].unique()[6]: 'S7',
            df['URL'].unique()[7]: 'S8'
        }

        # Replace URLs in the dataframe with their shorter label
        df['URL'] = df['URL'].map(url_mapping)

        # Group the data by 'Case' and 'URL'
        df_agg = df.groupby(['Case', 'URL']).size().unstack().fillna(0)

        # Create stacked bar chart using Plotly
        fig = go.Figure()

        # Add a bar for each URL (source)
        for url in df_agg.columns:
            fig.add_trace(go.Bar(
                x=df_agg.index,  # x-axis is 'Case'
                y=df_agg[url],   # y-axis is count per URL
                name=url,        # Legend will show 'S1', 'S2', etc.
                #hoverinfo='y',   # Display y values on hover
                hovertemplate=[f'{int(y)} images' for y in df_agg[url]],  # Custom hover text
                # hovertemplate='%{text}',
            ))

        # Customize the layout
        fig.update_layout(
            barmode='stack',
            xaxis_title="Cases",
            yaxis_title="Number of Images",
            legend_title="Source",
            xaxis_tickangle=-45,  # Rotate x-axis labels
        )
        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig)

        # Add a caption including the sources using markdown
        st.markdown(
        """
        **Figure 1**: Bar chart showing the number of images per class and source  
        S1 - [BIMCV-COVID](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711) |
        | S2 - [Eurorad](https://eurorad.org) |
        | S3 - [COVID-CXNet](https://github.com/armiro/COVID-CXNet) |
        | S4 - [Hannover Medical School](https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png) |
        | S5 - [Italian Society of Radiology](https://sirm.org/category/senza-categoria/covid-19/) |
        | S6 - [COVID-19 image data collection](https://github.com/ieee8023/covid-chestxray-dataset) |
        | S7 - [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) |
        | S8 - [Guangzhou Women and Children’s Medical Center](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
        """,
        unsafe_allow_html=True
        )
        # Insert an empty line to make it look nicer
        st.write("""
          

         """)

    # Show box plots - source dependent image props (checkbox)
    with st.expander("**Source dependent optical properties**"):
    # if st.checkbox('**Source dependent optical properties**'):
        st.write("""
        The images’ optical properties as well as the lung sizes are dependent on the source of the image. 
        This effect was tested to be significant using ANOVA statistics.
        Images collected from source 7 and 8 appear to have a lower mean pixel intensity 
        while their standard deviation of pixel intensity seems higher. 
        In addition we observed a significant difference of lung size between sources.
         """)
        st.image('Images/Img_Prop_Source.png', use_column_width=True)
        st.markdown(
        """
        **Figure 2**: Box plots illustrating the source-dependent distribution of mean pixel intensity, 
        standard deviation, and lung size (pixel count of the lung area)  
        S1 - [Italian Society of Radiology](https://sirm.org/category/senza-categoria/covid-19/) |
        | S2 - [COVID-19 image data collection](https://github.com/ieee8023/covid-chestxray-dataset) |
        | S3 - [BIMCV-COVID](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711) |
        | S4 - [Hannover Medical School](https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png) |
        | S5 - [Eurorad](https://eurorad.org) |
        | S6 - [COVID-CXNet](https://github.com/armiro/COVID-CXNet) |
        | S7 - [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) |
        | S8 - [Guangzhou Women and Children’s Medical Center](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
        """,
        unsafe_allow_html=True
        )
        # Insert an empty line to make it look nicer
        st.write("""
          

         """)

    # Show histograms - class dependent image props (checkbox)
    with st.expander("**Class dependent optical properties**"):
    # if st.checkbox('**Class dependent optical properties**'):
        st.write("""
        The four classes exhibit distinct distributions of pixel intensity statistics.
        COVID-19 cases tend to have a higher mean pixel intensity, more spread-out standard
        deviation, and a lower inter quartile range (IQR) compared to other classes.
        While the COVID-19 class shows a clear distinction, the respective distributions for
        normal and viral pneumonia look more similar. The strongest similarity of optical
        properties is observed between the distributions of lung opacity and normal cases.
        """)
       
        st.image('Images/Img_Prop_Class.png', use_column_width=True)
        st.markdown(
        """
        **Figure 3**: Histograms showing the distribution of mean pixel intensity (blue), 
        standard deviation of pixel intensity (green) and  inter quartile range (red) per class
        """,
        unsafe_allow_html=True
        )
       
        # Insert an empty line to make it look nicer
        st.write("""
          

         """)

   