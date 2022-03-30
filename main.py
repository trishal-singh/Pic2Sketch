import cv2
import numpy as np
import streamlit as st
from PIL import Image
from numpy import asarray

def detect_edge(image):
    if (image is not None):
        gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image,(5,5),0)
        sigma=0.33
        edges=cv2.Canny(blurred_image,int(max(0,(1-sigma)*np.median(blurred_image))),int(min(0,(1+sigma)*np.median(blurred_image))))
        M=np.ones(edges.shape,dtype="uint8")*255
        sketch=cv2.subtract(M,edges)

    return sketch
def main():
    st.title("Convert Image To Sketch")
    file=st.file_uploader("Choose an image...", type=["jpg",'jpeg','png','webp'])
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Original Image', use_column_width=True)
        output=detect_edge(asarray(image))
        st.image(output, caption='Sketch', use_column_width=True)


main()