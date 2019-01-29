import cv2 as cv
import numpy as np

showSteps = True

def display_image(image):
    """
    Displys an image and waits for a key press from the user.
    :param image:
        The image to display as a numpy array.
    :return:
        Nothing.
    """

    cv.imshow(cv.namedWindow("image", cv.WINDOW_AUTOSIZE), image)
    cv.waitKey(0)
    return

def RGB_to_greyscale(image):
    """
    Converts an RGB image to grayscale.
    :param image:
        The image to convert as a numpy array.
    :return:
        The grayscale image.
    """

    greyImage = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    if showSteps == True:
        display_image(greyImage)
    return greyImage

def blur_image(image, ksize = 7):
    """
    Blurs an image using a median filter.
    :param image:
        The image to Blur as a numpy array.
    :param ksize:
        The size of the mask, default is 7.
    :return:
        The blurred image.
    """

    blurredImage = cv.medianBlur(image, ksize) 
    if showSteps == True:
        display_image(blurredImage)
    return blurredImage

def laplace_filter(image):
    """
    Applies a laplace filter.
    :param image:
        The image as a numpy array.
    :return:
        The image after applying the filter.
    """

    laplaceImage = cv.Laplacian(image,-1, ksize=5)
    if showSteps == True:
        display_image(laplaceImage)
    return laplaceImage

def threshold_image(image):
    """
    Applies thresholding to the input image.
    :param image:
        The image to threshold as a numpy array.
    :return:
        The image after thresholding.
    """

    ret, thresholdImage = cv.threshold(image, 125, 255, cv.THRESH_BINARY_INV)
    if showSteps == True:
        display_image(thresholdImage)
    return thresholdImage

def bilateral_filter(image, repetitionCount):
    """
    Applies bilateral filter to the input image.
    :param image:
        The image to threshold as a numpy array.
    :param repetitionCount:
        The number of times to apply the filter.
    :return:
        The image after applying the bilateral filter.
    """

    bilateralImage = image
    for i in range(repetitionCount):
        bilateralImage = cv.bilateralFilter(bilateralImage, 9, 9, 7)
    if showSteps == True:
        display_image(bilateralImage)
    return bilateralImage

def add_images(image, mask):
    """
    Add a mask to an image.
    :param image:
        The input image.
    :param mask:
        the mask to add to the imge.
    :return:
        The image after adding the input image and the mask.
    """

    row, columns, channels = image.shape
    darkImage = np.zeros([row, columns, channels], dtype=np.uint8)
    finalImage = cv.add(image, darkImage, mask=mask)
    return finalImage

# read the image.
image = cv.imread("image2.jpg", 1)
display_image(image)

# apply the cartoon effect.
grayImage = RGB_to_greyscale(image)
blurredImage = blur_image(grayImage)
laplaceImage = laplace_filter(blurredImage)
thresholdImage = threshold_image(laplaceImage)
bilateralImage = bilateral_filter(image, 7)

# display the final image.
finalImage = add_images(bilateralImage, thresholdImage)
display_image(finalImage)