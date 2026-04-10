import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from skimage import data, exposure
from skimage.feature import graycomatrix, graycoprops, hog


def read_dicom_data(path, attribute):
    dicom_data = pydicom.dcmread(path)
    return getattr(dicom_data, attribute, "Unknown")


def read_dicom_image(path):
    """
    Read a DICOM image file and extract pixel data.

    Converts pixel data to float32 format if the image has more than 8 bits
    of stored data to ensure proper processing.

    Args:
        path (str): File path to the DICOM image file.

    Returns:
        np.ndarray: Pixel array data from the DICOM file, converted to float32
                   if BitsStored > 8, otherwise in original format.

    Raises:
        FileNotFoundError: If the DICOM file does not exist at the given path.
        InvalidDicomError: If the file is not a valid DICOM file.
    """
    dicom_data = pydicom.dcmread(path)
    image_data = dicom_data.pixel_array

    # Ensure that image data is in the correct format (float32 for processing)
    if dicom_data.get("BitsStored", 0) > 8:
        image_data = image_data.astype(np.float32)

    return image_data


def get_image(folder, filename=None):
    # Verify if the folder exists
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")

    # List all .dcm files in the folder
    dicom_files = []

    # Walk through the folder to get .dcm files
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))

    # If no DICOM files are found, raise an error
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found in '{folder}'.")

    # If filename is provided, return the specific file
    if filename:
        specific_file = os.path.join(folder, filename)
        if specific_file in dicom_files:
            return specific_file
        else:
            raise FileNotFoundError(f"File '{filename}' not found in '{folder}'.")

    # Otherwise, select a random file
    return random.choice(dicom_files)


def extract_feature_hog(image):
    """
    Extracts HOG features and returns an annotated image
    for visualization in matplotlib.
    """
    # 1. Extract HOG features
    # orientations: number of orientation bins
    # pixels_per_cell: size of the cell for which gradients are calculated
    # cells_per_block: number of cells in each block (for normalization)
    # visualize=True: returns an image representation of the HOG
    fd, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None,
    )

    # 2. Rescale the HOG image for better visualization (Contrast Stretching)
    # This makes the "annotations" (gradients) pop against the background
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_image_rescaled


def extract_feature_glcm(image):
    # 1. Force Image to uint8 (0-255)
    # If input is float (0.0 - 1.0), scale it. If it's already 0-255, leave it.
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    # 2. Global GLCM Feature Extraction
    glcm = graycomatrix(
        image_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )

    extracted_data = {
        "contrast": np.mean(graycoprops(glcm, "contrast")),
        "homogeneity": np.mean(graycoprops(glcm, "homogeneity")),
        "energy": np.mean(graycoprops(glcm, "energy")),
    }

    # 3. Create a 3-channel RGB image for color annotation
    # This is critical: if you stay in grayscale, (255, 0, 0) just becomes "white"
    annotated_img = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    h, w = image_uint8.shape
    patch_size = 64

    # Calculate a dynamic threshold based on global contrast
    # Areas significantly higher than the average contrast are flagged
    global_contrast = extracted_data["contrast"]
    threshold = global_contrast * 1.5

    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = image_uint8[y : y + patch_size, x : x + patch_size]

            # Local GLCM for the patch
            patch_glcm = graycomatrix(
                patch, [1], [0], levels=256, symmetric=True, normed=True
            )
            local_contrast = graycoprops(patch_glcm, "contrast")[0, 0]

            # Draw Bright Red box if local contrast is high (potential abnormality)
            if local_contrast > threshold:
                # cv2.rectangle(image, start_point, end_point, color, thickness)
                cv2.rectangle(
                    annotated_img,
                    (x, y),
                    (x + patch_size, y + patch_size),
                    (255, 0, 0),
                    3,
                )  # Increased thickness to 3

    return extracted_data, annotated_img
