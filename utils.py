import cv2 
import numpy as np

# convert image to grayscale 
def img_grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# image noise removal 
def img_remove_noise(image):
    image = cv2.medianBlur(image,5)
    return image
 
# image thresholding
def img_thresholding(image):
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return image

# image dilation
def img_dilation(image):
    kernel = np.ones((5,5),np.uint8)
    image = cv2.dilate(image, kernel, iterations = 1)
    return image
    
# image erosion
def img_erosion(image):
    kernel = np.ones((5,5),np.uint8)
    image = cv2.erode(image, kernel, iterations = 1)
    return image

# image canny edge detection
def img_canny(image):
    image = cv2.Canny(image, 100, 200)
    return image