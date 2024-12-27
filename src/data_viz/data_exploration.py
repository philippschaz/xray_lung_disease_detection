# This code loads the X-Ray lung data set
# Source: https://www.kaggle.com/code/yahyouhh/cv-project-covid


# 0. Define Working Directory
###

# Desktop:
###

# Define local path with image data
path = "D:/3_Work/Datascience/X-Ray Project/data"

# Define working directory path
path_wd = "D:/3_Work/Datascience/X-Ray Project"

# Define output path
path_out = "D:/3_Work/Datascience/X-Ray Project/output"

# Define working directory
os.chdir(path_wd)

# Show current working directory
print(os.getcwd())

# Laptop
###

# 1. Load packages
###
# install pandas, numpy, opencv and path
# install pillow, tensorflow, keras, keras-models, scikit-learn, setuptools, jax
import pandas as pd 
import numpy as np 
import cv2
import os 
import setuptools
import sys
import keras
import matplotlib.pylab as plt

from PIL import Image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping



# 2. import data
###
# define function to read and normalize images
def read_image_and_normalize(image_path):
    img = cv2.imread(image_path)
    img_array = np.array(img)
    # Standardize by 255 to have pixel channels between [0,1]
    normalized_img = img_array / 255.0  
    return normalized_img


# Create data frame containing for each image the path and label
# Define content of directory as list
folders=os.listdir(path)
print(folders)

# Create empty list of data and label
data=[]
label=[]

# Loop over folders in directory
for folder in folders:
    item_path = os.path.join(path, folder)
    if os.path.isdir(item_path):
        print(folder)
        
        data_folder = os.listdir(item_path)
        for image_folder in data_folder:
            if image_folder == "images":
                image_folder_path = os.path.join(item_path, image_folder)
                
                for i in range(int(len(os.listdir(image_folder_path)) * 0.3)) :
                      image = os.listdir(image_folder_path)[i]
                      data.append(os.path.join(image_folder_path,image))
                      label.append(folder)

# Define data frame with image path and labels
df=pd.DataFrame({"image_path": data, "label": label})

# Show data frame
print(df["image_path"].head(20))
df.shape

# Create array with all labels of the images
label = np.array(label)
print("The labels are:", label)

# Transform the labels to categorical variable as dummies
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(label)
label_cat = to_categorical(encoded_labels)
print("The encoded labels are:", label_cat)

# Create Memory-mapped file to access only small segments of large files on disk, 
# without reading the entire file into memory

# Define the shape of the memory map and create data output file (overwrite mode)
memmap_shape = (len(label_cat), 299, 299, 3)  # Question: is this now a 4 dimensional matrix
memmap_array = np.memmap("normalized_images.dat", dtype = np.float32, mode='w+', shape = memmap_shape)  


# Apply image normalization function to all images, as defined earlier
for i, image_path in enumerate(df.image_path):
    memmap_array[i] = read_image_and_normalize(image_path)


# 3. Data exploration
###
# Load memory map in reading mode
memmap_array = np.memmap("normalized_images.dat", dtype=np.float32, mode='r', shape=memmap_shape)

# Show data from memory map, which is now a pixel array
print(memmap_array)

# Check Type: is memmap array
type(memmap_array)

# Check Shape: Now this is a 4 dimensional matrix 
print(memmap_array.shape)

# Show labels
print(label)

# Show occurences of different labels
df.label.value_counts()

# Example image: no. 20
###
img_20 = memmap_array[20]

# Show pixel values
print(img_20)

# Show dimension: here 3-dimensional matrix (Height, width, 3 channels: RGB)
print(img_20.shape)

# Display image with label
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_20)
img_20_label = "Image 20" + " (" + df.label[20] + ")"
plt.title(img_20_label)
ax.axis("off")
plt.show()

# Display the three different RGB Channels of the image
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img_20[:,:,0], cmap="Reds")
axs[1].imshow(img_20[:,:,1], cmap="Greens")
axs[2].imshow(img_20[:,:,2], cmap="Blues")
axs[0].set_title("Red channel")
axs[1].set_title("Green channel")
axs[2].set_title("Blue channel")
plt.show()

# We can see that the color channels cover the areas differently and thus convey different information

# histogram of flattened pixel values
pd.Series(img_20.flatten()).plot(kind="hist", 
                                bins=50, 
                                title = "Distribution of pixel values (image no. 20)")
plt.show()



# 4. Extraction of Descriptive Statistics
####

# 4.1 Model Selection: Timing different Pixel Transformation options
####
import time
start_time = time.time()

# Define sample size
memmap_array_select= memmap_array[1:100]
len(memmap_array_select)


# Model 1: Lists + Numpy array (np.column_stack)
###
# Starting Column
descr = pd.Series(memmap_array_select[1].flatten()).to_list()
#a = pd.Series(memmap_array_select[2].flatten()).to_list()

# # Adding the remaining columns (numpy array)
for i in range(2, len(memmap_array_select)):
    a = pd.Series(memmap_array_select[i].flatten()).to_list()
    descr = np.column_stack([descr, a])

#descr = np.column_stack([descr, a])
descr.shape
type(descr)

# Transform Numpy Array into Data Frame
descr = pd.DataFrame(descr)
type(descr)
descr.head()

# Measure time
model1_time = time.time() - start_time

# Model 2: Series + Data Frame (pd.concat)
###
import time
start_time = time.time()

# Starting Column
descr = pd.Series(memmap_array_select[1].flatten())

# Adding the remaining columns (Series and Dataframe)
for i in range(2, len(memmap_array_select)):
    a = pd.Series(memmap_array_select[i].flatten())
    descr = pd.concat([descr, a], axis = 1)

descr.shape
type(descr)

# Transform Numpy Array into Data Frame
descr = pd.DataFrame(descr)
type(descr)
descr.head()

# Measure time
model2_time = time.time() - start_time

# Model 3: Construct statistics on the fly with list, then collect data frame
###
import time
start_time = time.time()

# Starting Column
descr = pd.Series(memmap_array_select[1].flatten())

# i) Flatten pixel
# ii) Construct the distributional statistics 
# iii) Add the remaining columns as loop (Series and Dataframe)
descr_mean = []
for i in range(len(memmap_array_select)):
    a = pd.Series(memmap_array_select[i].flatten()).mean()
    descr_mean.append(a)
type(descr_mean)


# Measure time
model3_time = time.time() - start_time

# Compare times across models
print("Model 1 --- %s seconds ---" % model1_time)
print("Model 2 --- %s seconds ---" % model2_time)
print("Model 3 --- %s seconds ---" % model3_time)

# Choose Model 3 is factor 40 faster than model 1


# 4.2 Data extraction with Model 3
####
# For efficiency, accumulate data with lists and not Series/DataFrame, as this is a much lighter data structure
import time
start_time = time.time()

# Select sample size
###
# Smaller size:
# N = 100
# memmap_array_select= memmap_array[1:N]

# Full size:
memmap_array_select= memmap_array
len(memmap_array_select)

# i) Flatten pixel for each image
# ii) Construct the distributional statistics (defined as list)
# iii) Add the remaining columns as loop with append (add to list)
# iv) Construct DataFrame from list of lists
descr_mean = []
descr_std = []
descr_kurtosis = []
descr_q1 = []
descr_q2 = []
descr_q3 = []
descr_iqr = []

for i in range(len(memmap_array_select)):
    mean = pd.Series(memmap_array_select[i].flatten()).mean()
    std = pd.Series(memmap_array_select[i].flatten()).std()
    kurtosis = pd.Series(memmap_array_select[i].flatten()).kurtosis() 
    q1 = pd.Series(memmap_array_select[i].flatten()).quantile(0.25) 
    q2 = pd.Series(memmap_array_select[i].flatten()).quantile(0.5) 
    q3 = pd.Series(memmap_array_select[i].flatten()).quantile(0.75) 
    iqr = q3 - q1

    descr_mean.append(mean)
    descr_std.append(std)
    descr_kurtosis.append(kurtosis)
    descr_q1.append(q1)
    descr_q2.append(q2)
    descr_q3.append(q3)
    descr_iqr.append(iqr)
type(descr_mean)

# Create DataFrame
pixel_descr = pd.DataFrame({"mean": descr_mean, 
                            "std": descr_std,
                            "kurtosis": descr_kurtosis,
                            "Q1": descr_q1,
                            "Q2": descr_q2,
                            "Q3": descr_q3,
                            "IQR": descr_iqr})

# Check cosntruction
type(pixel_descr)
pixel_descr.head()

# Create index as img number
pixel_descr["img_no"] = range(len(memmap_array_select))
pixel_descr = pixel_descr.set_index("img_no")
pixel_descr.head()

# Add labels
# Define length according to data subset above
length = pixel_descr.shape[0]
pixel_descr["label"] = df.label[:length]

# Show data frame
pixel_descr.head()
pixel_descr.tail()

# Measure time
model3_time = time.time() - start_time
print("Model 3 --- %s seconds ---" % model3_time)


# 5. Descriptive Statistics: Visualisation
####

# Compare Histograms by label
pixel_descr.hist("mean", by = "label", bins = 30, color = "lightblue", sharex = True)
plt.show()

pixel_descr.hist("std", by = "label", bins = 30, color = "lightgreen", sharex = True)
plt.show()

pixel_descr.hist("kurtosis", by = "label", bins = 30, color = "blue", sharex = True)
plt.show()

pixel_descr.hist("Q1", by = "label", bins = 30, color = "orange", sharex = True)
plt.show()

pixel_descr.hist("Q2", by = "label", bins = 30, color = "marineblue", sharex = True)
plt.show()

pixel_descr.hist("Q3", by = "label", bins = 30, color = "red", sharex = True)
plt.show()

pixel_descr.hist("IQR", by = "label", bins = 30, color = "pink", sharex = True)
plt.show()

