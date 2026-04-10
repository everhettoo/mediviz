import cv2
import numpy as np


def normalize_image(img):
    img = img.astype("float32")
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img


def median_filter(img, ksize=3):
    """
    Apply a median filter to an image using OpenCV's medianBlur.
    This function reduces noise in the image by replacing each pixel value with the median
    value of neighboring pixels within a kernel of specified size. The kernel size must be
    an odd integer to ensure symmetry.
    Parameters:
        img (numpy.ndarray): The input image to be filtered. Should be a grayscale or color image.
        ksize (int, optional): The size of the kernel (filter window). Must be an odd positive integer.
                               Defaults to 3.
    Returns:
        numpy.ndarray: The filtered image with the same shape and type as the input.
    Raises:
        ValueError: If ksize is not an odd integer.
    """

    # Ensure kernel size is odd
    if ksize % 2 == 0:
        raise ValueError("Kernel size must be odd")

    filtered = cv2.medianBlur(img, ksize)

    return filtered


def adjust_contrast(image, clipLimit=3.0, tileGridSize=(8, 8)):
    # Convert to uint8 AFTER proper scaling
    image_uint8 = (image * 255).astype("uint8")

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    contrast_adjusted = clahe.apply(image_uint8)

    return contrast_adjusted
