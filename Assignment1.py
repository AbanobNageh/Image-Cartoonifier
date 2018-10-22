import cv2 as cv
import numpy as np

showSteps = True

def display_image(image):
    cv.imshow(cv.namedWindow("image", cv.WINDOW_AUTOSIZE), image)
    cv.waitKey(0)
    return

def RGB_to_greyscale(image):
    greyImage = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    if showSteps == True:
        display_image(greyImage)
    return greyImage

def blur_image(image):
    blurredImage = cv.medianBlur(image, 7) 
    if showSteps == True:
        display_image(blurredImage)
    return blurredImage    

def laplace_filter(image):
    laplaceImage = cv.Laplacian(image,-1, ksize=5)
    if showSteps == True:
        display_image(laplaceImage)
    return laplaceImage

def threshold_image(image):
    ret, thresholdImage = cv.threshold(image, 125, 255, cv.THRESH_BINARY_INV)
    if showSteps == True:
        display_image(thresholdImage)
    return thresholdImage

def bilateral_filter(image, repetitionCount):
    bilateralImage = image
    for i in range(repetitionCount):
        bilateralImage = cv.bilateralFilter(bilateralImage, 9, 9, 7)
    if showSteps == True:
        display_image(bilateralImage)
    return bilateralImage

def add_images(image, mask):
    row, columns, channels = image.shape
    darkImage = np.zeros([row, columns, channels], dtype=np.uint8)
    finalImage = cv.add(image, darkImage, mask=mask)
    return finalImage

image = cv.imread("image.jpg", 1)
display_image(image)

grayImage = RGB_to_greyscale(image)
blurredImage = blur_image(grayImage)
laplaceImage = laplace_filter(blurredImage)
thresholdImage = threshold_image(laplaceImage)
bilateralImage = bilateral_filter(image, 7)

finalImage = add_images(bilateralImage, thresholdImage)
display_image(finalImage)