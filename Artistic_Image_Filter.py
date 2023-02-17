import streamlit as st
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Set up plot
def plot(img1, img2):
    
    fig = plt.figure(figsize = (20,10))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Filtered Image")

    plt.show()

#Black and white filtered
def bw_filter(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

#Sepia filtered
def sepia(img):
    img_sepia = img.copy()
    # Converting to RGB as sepia matrix below is for RGB.
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB) 
    img_sepia = np.array(img_sepia, dtype = np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
    # Clip values to the range [0, 255].
    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia, dtype = np.uint8)
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
    return img_sepia

#Vintage filtered
def vintage(img, level = 3):
    height, width = img.shape[:2]  

    # Generate vignette mask using Gaussian kernels.
    X_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height/level)
        
    # Generating resultant_kernel matrix.
    # H x 1 * 1 x W
    kernel = Y_resultant_kernel * X_resultant_kernel.T 
    mask = kernel / kernel.max()
    
    img_vignette = np.copy(img)
        
    # Applying the mask to each channel in the input image.
    for i in range(3):
        img_vignette[:,:,i] = img_vignette[:,:,i] * mask
    
    return img_vignette
#Pencil Sketch filtered
def pencil_Sketch(img):
    img_blur = cv2.GaussianBlur(img, (5,5), 0, 0)
    img_sketch_bw, img_sketch_color = cv2.pencilSketch(img_blur)
    return img_sketch_bw

def main_loop():
    #Show title
    st.title("Image Filter App")
    st.text("I use OpenCV and Streamlit to filter image.")
    #Updload file:
    img_file = st.file_uploader("Upload your image: ",type=['jpg','png','jpeg'])
    if not img_file:
        return None
    #Processing image:
    original_img = Image.open(img_file)
    original_img = np.array(original_img)
    #Choose your filter in selected box
    option_filtered = ["Black and White", "Sepia", "Vintage", "Pencil Sketch"]
    selected_filtered = st.selectbox("Choose your filter for your image: ",option_filtered)
    #All image after filtered
    img_bw = bw_filter(original_img)
    img_sepia = sepia(original_img)
    img_vintage = vintage(original_img)
    img_pencilSketch = pencil_Sketch(original_img)
    #Layout
    col1, col2 = st.columns(2)
    col1.write('Original Image')
    col2.write('Filtered Image')
    col1.image(original_img)
    #Show original image and filtered image
    if selected_filtered == 'Black and White':
        col2.image(img_bw)
    elif selected_filtered == 'Sepia':
        col2.image(img_sepia)
    elif selected_filtered == 'Vintage': 
        col2.image(img_vintage)
    elif selected_filtered == 'Pencil Sketch':
        col2.image(img_pencilSketch)

if __name__ == '__main__':
    main_loop()