import sys
print(sys.executable)

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.stats import skew, kurtosis

#Step1: Checking readability of images and masks

#Function to check readability of images and masks
def is_img_readable(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img is not None

#Step2: Checking if images and masks are correctly labelled

#Function to check label of images and masks
def is_label_correct(pattern, img):   
    return re.match(pattern, img) is not None

#Step3: Checking for uniform image size/shape

#Function to check for correct shape
def is_shape_correct(shape, img_path):
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    return img_shape == shape

#Set path to main folder    
folder_path = "D:/COVID-19_Radiography_Dataset"
#Set paths to subfolders
subfolder_paths = ["COVID/images", "Lung_Opacity/images", "Normal/images", "Viral Pneumonia/images",
                   "COVID/masks", "Lung_Opacity/masks", "Normal/masks", "Viral Pneumonia/masks"]

#Create dictionary with the respective label patterns
patterns = {
    'COVID/images': r"COVID-[0-9]{1,4}\.png",   
    'Lung_Opacity/images': r"Lung_Opacity-[0-9]{1,4}\.png", 
    'Normal/images': r"Normal-[0-9]{1,5}\.png",
    'Viral Pneumonia/images': r"Viral Pneumonia-[0-9]{1,4}\.png",
    'COVID/masks': r"COVID-[0-9]{1,4}\.png", 
    'Lung_Opacity/masks': r"Lung_Opacity-[0-9]{1,4}\.png",
    'Normal/masks': r"Normal-[0-9]{1,5}\.png",
    'Viral Pneumonia/masks': r"Viral Pneumonia-[0-9]{1,4}\.png",
}

#Create dictionary with the respective shapes
shapes = {
    'images': (299, 299),   
    'masks': (256, 256)
}

#Count correct and incorrect cases
total = 0
readable = 0
not_readable = 0
label_correct = 0
label_incorrect = 0
shape_correct = 0
shape_incorrect = 0

errors = []

#Check each subfolder in the main folder
for subfolder in subfolder_paths:
    subfolder_path = os.path.join(folder_path, subfolder)

    #Retrieve respective pattern from patterns dictionary
    pattern = patterns[subfolder]

    #Retrieve respective shape from shapes dictionary
    shape = shapes[subfolder.split('/')[-1]]
    
    #Check each image in the subfolder
    for img in os.listdir(subfolder_path):
        img_path = os.path.join(subfolder_path, img)

        total += 1
        
        #Check if image is a readable
        if is_img_readable(img_path):
            readable += 1
        else:
            errors.append((f"Image file is not readable: {img_path}"))
            not_readable += 1

        #Check if image is labelled correctly
        if is_label_correct(pattern, img):
            label_correct += 1
        else:
            errors.append(f"{img} in {subfolder} is incorrectly labelled!")
            label_incorrect += 1

        #Check if image has correct shape
        if is_shape_correct(shape, img_path):
            shape_correct += 1
        else:
            errors.append(f"{img} in {subfolder} is of incorrect shape!")
            shape_incorrect += 1

#Write results of quality check to txt
file_path = 'D:/COVID-19_Radiography_Dataset/errors_summary.txt'
with open(file_path, 'w') as file:
    #Add title
    file.write('---- Quality check ----\n')
    file.write('\n')

    #Write the errors summary to text
    file.write(f"Total number of images and masks: {total}" + '\n')
    file.write('\n')
    file.write(f"Readable: {readable}" + '\n')
    file.write(f"Not readable: {not_readable}" + '\n')
    file.write('\n')
    file.write(f"Label_correcct: {label_correct}" + '\n')
    file.write(f"Label_incorrect: {label_incorrect}" + '\n')
    file.write('\n')
    file.write(f"Shape_correct: {shape_correct}" + '\n')
    file.write(f"Shape_incorrect: {shape_incorrect}" + '\n')
    
    #Add space and heading
    file.write('\n')
    file.write('\n')
    file.write('---- Errors ----\n')
    
    #Write the errors to text
    for error in errors:
        file.write(error + '\n')

#Step4: Converting images to greyscale

#Function to convert image to grayscale
def convert_to_greyscale(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey_img

#Step5: Masking image

#Function to resize and appply mask
def apply_mask(img, msk):
    msk_resized = cv2.resize(msk, dsize = (299,299), interpolation = cv2.INTER_LINEAR)
    msk_img = cv2.bitwise_and(img, img, mask=msk_resized)
    #Exclude the black pixels
    return msk_img[msk_resized == 255]

#Step6: Normalizing pixel values

#Function to normalize pixel values
def normalize_pixel(img):
    img = np.asarray(img)
    mini = np.min(img)
    maxi = np.max(img)
    if maxi > mini:
        normalized_img = (img - mini) / (maxi - mini)
    else:
        normalized_img = np.zeros_like(img)
    return normalized_img  #Normalization to [0, 1]

#Step7: Reducing noise of the images
#There are different options to reduce the noise in an image, like Gaussian filtering (applied here), median filtering, bilateral filtering,
#non-local means denoising and Wavelet denoising.
#Gaussian filtering: Smooths an image by averaging pixel values with their neighbors, weighted by a Gaussian kernel. The kernel size (here: 5x5 pixel) and standard deviation (σ) 
#(here 0, meaning standard deviation is calculated based on kernel size) of the Gaussian function determine the extent of smoothing.

#Function to reduce noise
def reduce_noise(img):
    denoised_img = cv2.GaussianBlur(img, (3, 3), 0)
    return denoised_img

#Step8: Enhancing contrast
#There are different options to enhance contrast in an image, like histogram equalization (applied here), Contrast Limited Adaptive Histogram Equalization (CLAHE),
#Image Complement and Gamma correction.
#Histogram equalization: Redistributes the intensity values of pixels. Aims at spreading out the most frequent intensity values to enhance the contrast of the image.

#Function to enhance contrast
def enhance_contrast(img):
    enhanced_img = cv2.equalizeHist((img * 255).astype(np.uint8)) / 255.0
    #alternative
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #enhanced_img = clahe.apply(np.uint8(denoised_img * 255))
    return enhanced_img

#Step9: Retrieving descriptive statistics

#Function to retrieve statistics
def get_statistics(img):
    #Convert to float
    img = img.astype('float64')

    #Convert from 2D to 1D array
    flat_img = img.flatten()

    #Calculate statistics
    count = len(flat_img)
    meani = np.mean(flat_img)
    std = np.std(flat_img)
    mini = np.min(flat_img)
    maxi = np.max(flat_img)
    med = np.median(flat_img)
    q25 = np.percentile(flat_img, 25)
    q75 = np.percentile(flat_img, 75)
    skewness = skew(flat_img)
    kurt = kurtosis(flat_img)
    
    return count, meani, std, mini, maxi, med, q25, q75, skewness, kurt

#Step10: Retrieving image meta information

#Function to retrieve meta information
def get_meta(img):
    img_short = img.split('-')[0]
    img_png = img.split('.')[0]
    if img_short == 'COVID':
        return covid_meta[covid_meta['FILE NAME'] == img_png]['URL'].iloc[0]
    elif img_short == 'Normal':
        img_png = img_png.upper()
        return normal_meta[normal_meta['FILE NAME'] == img_png]['URL'].iloc[0]
    elif img_short == 'Lung_Opacity':
        return lung_meta[lung_meta['FILE NAME'] == img_png]['URL'].iloc[0]
    else:
        return viral_meta[viral_meta['FILE NAME'] == img_png]['URL'].iloc[0]

#Read meta data
covid_meta = pd.read_excel('D:/COVID-19_Radiography_Dataset/COVID.metadata.xlsx', sheet_name='Sheet1', usecols=['FILE NAME', 'URL'])
normal_meta = pd.read_excel('D:/COVID-19_Radiography_Dataset/Normal.metadata.xlsx', sheet_name='Sheet1', usecols=['FILE NAME', 'URL'])
lung_meta = pd.read_excel('D:/COVID-19_Radiography_Dataset/Lung_Opacity.metadata.xlsx', sheet_name='Sheet1', usecols=['FILE NAME', 'URL'])
viral_meta = pd.read_excel('D:/COVID-19_Radiography_Dataset/Viral Pneumonia.metadata.xlsx', sheet_name='Sheet1', usecols=['FILE NAME', 'URL'])

subfolder_paths = ["COVID/images", "Lung_Opacity/images", "Normal/images", "Viral Pneumonia/images"]
output_dir = "D:/COVID-19_Radiography_Dataset"

#Set list for output
dataframe1 = []
dataframe2 = []
dataframe3 = []

#Check each subfolder in the main folder
for subfolder in subfolder_paths:
    subfolder_path = os.path.join(folder_path, subfolder)

    subfolder_path_mask = f"{folder_path}/{subfolder.split('/')[0]}/masks"

    #Check each image in the subfolder
    for img in os.listdir(subfolder_path):

        msk = img 

        img_path = os.path.join(subfolder_path, img)
        msk_path = os.path.join(subfolder_path_mask, msk)

        img_string = img.split('.')[0]

        #Read image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(msk_path)

        #Pipe image through the pre-processing steps
        grey_img = convert_to_greyscale(image)
        grey_msk = convert_to_greyscale(mask)
        mask_img = apply_mask(grey_img, grey_msk)
        norm_img = normalize_pixel(grey_img)
        norm_img2 = normalize_pixel(mask_img)
        denoised_img = reduce_noise(norm_img2)
        enhanced_img = enhance_contrast(denoised_img)
        count, mean, std, mini, maxi, med, q25, q75, skewness, kurt = get_statistics(norm_img)
        count2, mean2, std2, mini2, maxi2, med2, q252, q752, skewness2, kurt2 = get_statistics(norm_img2)
        count3, mean3, std3, mini3, maxi3, med3, q253, q753, skewness3, kurt3 = get_statistics(enhanced_img)
        url = get_meta(img)

        #Append statistics and meta to list
        dataframe1.append({'Name': img_string, 'Case' :  img_string.split('-')[0], 'Img_path': img_path, 'Mask_path': msk_path,
                          'URL' : url, 'Count' : count,'Mean': mean, 'Std_dev' : std, 'Min' : mini, 'Max' : maxi, 
                          'Median': med, 'Q25' : q25, 'Q75' : q75, 'Skewness' : skewness, 'Kurtosis' : kurt})
        
        dataframe2.append({'Name': img_string, 'Case' :  img_string.split('-')[0], 'Img_path': img_path, 'Mask_path': msk_path,
                          'URL' : url, 'Count' : count2, 'Mean': mean2, 'Std_dev' : std2, 'Min' : mini2, 'Max' : maxi2, 
                          'Median': med2, 'Q25' : q252, 'Q75' : q752, 'Skewness' : skewness2, 'Kurtosis' : kurt2})
        
        dataframe3.append({'Name': img_string, 'Case' :  img_string.split('-')[0], 'Img_path': img_path, 'Mask_path': msk_path,
                          'URL' : url, 'Count' : count3, 'Mean': mean3, 'Std_dev' : std3, 'Min' : mini3, 'Max' : maxi3, 
                          'Median': med3, 'Q25' : q253, 'Q75' : q753, 'Skewness' : skewness3, 'Kurtosis' : kurt3})

#Convert list to dataframe
df_out = pd.DataFrame(dataframe1)
df_out2 = pd.DataFrame(dataframe2)
df_out3 = pd.DataFrame(dataframe3)

#Export dataframe as xlsx
df_out.to_excel('D:/COVID-19_Radiography_Dataset/df_stat_v2.xlsx', index=False, sheet_name='Statistics', float_format='%.4f')
df_out2.to_excel('D:/COVID-19_Radiography_Dataset/df_stat_masked_v2.xlsx', index=False, sheet_name='Statistics', float_format='%.4f')
df_out3.to_excel('D:/COVID-19_Radiography_Dataset/df_stat_preprocessed_v2.xlsx', index=False, sheet_name='Statistics', float_format='%.4f')
#df_out.to_csv('D:/Test_Output/df_stat_test.csv', index=False, float_format='%.4f')