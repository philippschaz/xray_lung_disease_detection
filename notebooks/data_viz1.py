import sys
print(sys.executable)

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#Function to pre-process images
def preprocess_img(img, msk):

    #Converting to greyscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey_msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

    #Masking image
    msk_resized = cv2.resize(grey_msk, dsize = (299,299), interpolation = cv2.INTER_LINEAR)
    msk_img = cv2.bitwise_and(grey_img, grey_img, mask=msk_resized)

    #Normalizing pixel values
    img_norm = np.asarray(msk_img)
    mini = np.min(img_norm)
    maxi = np.max(img_norm)
    if maxi > mini:
        normalized_img = (img - mini) / (maxi - mini)
    else:
        normalized_img = np.zeros_like(img)
    normalized_img_uint8 = (normalized_img * 255).astype(np.uint8)

    #Reducing noise
    denoised_img = cv2.GaussianBlur(normalized_img, (3, 3), 0)
    denoised_img_uint8 = (denoised_img * 255).astype(np.uint8)

    #Denoising alternative
    #denoised_img = cv2.fastNlMeansDenoising((normalized_img * 255).astype(np.uint8), None, h=5, templateWindowSize=7, searchWindowSize=21) / 255.0
    #denoised_img_uint8 = (denoised_img * 255).astype(np.uint8)

    #Enhance image contrast
    #enhanced_img = cv2.equalizeHist((normalized_img * 255).astype(np.uint8)) / 255.0
    #enhanced_img_uint8 = (enhanced_img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(np.uint8(denoised_img * 255))
    enhanced_img_uint8 = (enhanced_img * 255).astype(np.uint8)

    return grey_img, msk_img, normalized_img_uint8, denoised_img_uint8, enhanced_img_uint8

#Load/read example images for each case
img_normal = cv2.imread("D:/Test/Normal/images/Normal-21.png")
img_covid = cv2.imread("D:/Test/COVID/images/COVID-31.png")
img_lung_opacity = cv2.imread("D:/Test/Lung_Opacity/images/Lung_Opacity-3.png")
img_viral_pneumonia = cv2.imread("D:/Test/Viral Pneumonia/images/Viral Pneumonia-9.png")

#Load/read respective masks
msk_normal = cv2.imread("D:/Test/Normal/masks/Normal-21.png")
msk_covid = cv2.imread("D:/Test/COVID/masks/COVID-31.png")
msk_lung_opacity = cv2.imread("D:/Test/Lung_Opacity/masks/Lung_Opacity-3.png")
msk_viral_pneumonia = cv2.imread("D:/Test/Viral Pneumonia/masks/Viral Pneumonia-9.png")

#Apply pre-processing function to images
grey_norm, msk_norm, normalized_norm_uint8, denoised_norm_uint8, enhanced_norm_uint8 = preprocess_img(img_normal, msk_normal)
grey_covid, msk_covid, normalized_covid_uint8, denoised_covid_uint8, enhanced_covid_uint8 = preprocess_img(img_covid, msk_covid)
grey_lung, msk_lung, normalized_lung_uint8, denoised_lung_uint8, enhanced_lung_uint8 = preprocess_img(img_lung_opacity, msk_lung_opacity)
grey_viral, msk_viral, normalized_viral_uint8, denoised_viral_uint8, enhanced_viral_uint8 = preprocess_img(img_viral_pneumonia, msk_viral_pneumonia)

#Set rows and columns of figure, set figure size
fig, axes = plt.subplots(5, 4, figsize=(12, 15))

#Define annotations
annotations = ['Grayscale', 'Masked', 'Normalized', 'Denoised', 'Enhanced']

#Define column titles
col_titles = ['Normal', 'COVID', 'Lung Opacity', 'Viral Pneumonia']

#Define placement of each subplot
axes[0, 0].imshow(grey_norm, cmap='gray')
axes[1, 0].imshow(msk_norm, cmap='gray')
axes[2, 0].imshow(normalized_norm_uint8, cmap='gray')
axes[3, 0].imshow(denoised_norm_uint8, cmap='gray')
axes[4, 0].imshow(enhanced_norm_uint8, cmap='gray')

axes[0, 1].imshow(grey_covid, cmap='gray')
axes[1, 1].imshow(msk_covid, cmap='gray')
axes[2, 1].imshow(normalized_covid_uint8, cmap='gray')
axes[3, 1].imshow(denoised_covid_uint8, cmap='gray')
axes[4, 1].imshow(enhanced_covid_uint8, cmap='gray')

axes[0, 2].imshow(grey_lung, cmap='gray')
axes[1, 2].imshow(msk_lung, cmap='gray')
axes[2, 2].imshow(normalized_lung_uint8, cmap='gray')
axes[3, 2].imshow(denoised_lung_uint8, cmap='gray')
axes[4, 2].imshow(enhanced_lung_uint8, cmap='gray')

axes[0, 3].imshow(grey_viral, cmap='gray')
axes[1, 3].imshow(msk_viral, cmap='gray')
axes[2, 3].imshow(normalized_viral_uint8, cmap='gray')
axes[3, 3].imshow(denoised_viral_uint8, cmap='gray')
axes[4, 3].imshow(enhanced_viral_uint8, cmap='gray')

#Remove axis labels for all the subplots
for ax in axes.flatten():
    ax.axis('off')

#Add column titles
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=18, fontweight='bold')

#Add annotations to subplots
for i in range(5):
    for j in range(4):
        ax = axes[i, j]
        ax.annotate(annotations[i], xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15, ha='left', va='top', color='red')

#Optimize spaces between subplots
plt.tight_layout()

#Save image
plt.savefig('D:/Test_Output/preprocessing3.png')

#Display image
plt.show()