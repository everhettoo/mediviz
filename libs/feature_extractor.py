# Libraries used.

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from libs import dicom_helper as dicom
from libs import image_processor as processor


def extract_feature_lbp(img, radius, method):
    # Calculate number of points based on radius
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method=method)

    # Determine the number of bins based on the method
    if method == "uniform":
        # Uniform patterns yield (n_points + 2) possible values
        n_bins = int(n_points + 2)
    else:
        # Standard LBP yields 2^n_points possible values
        # Warning: This grows exponentially with radius!
        n_bins = int(2**n_points)

    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum() + 1e-6

    return lbp_hist


def extract_feature_lbp_only(img, radius, method):
    # Calculate number of points based on radius
    n_points = 8 * radius
    return local_binary_pattern(img, n_points, radius, method=method)


def perform_single_lda(X_train, y_train, new_sample, n_components):
    # 1. Scaling the data
    # LBP histograms are usually in similar ranges, but scaling helps LDA converge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 2. Initialize LDA
    # n_components must be < number of classes. For 2 classes, it must be 1.
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    # 3. Fit and Transform the data
    # Note: LDA requires y_labels during the fit process (Supervised)
    X_lda = lda.fit_transform(X_scaled, y_train)

    # Scale and transform the new sample.
    x_new_scaled = scaler.transform(new_sample)
    x_new_lda = lda.transform(x_new_scaled)

    print(f"Original shape: {X_train.shape}")
    print(f"Reduced shape: {X_lda.shape}")

    # Optional: Check how much 'variance' is explained by the new component
    print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
    return x_new_lda


def perform_lda(X_train, y_train, n_components):
    # 1. Scaling the data
    # LBP histograms are usually in similar ranges, but scaling helps LDA converge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 2. Initialize LDA
    # n_components must be < number of classes. For 2 classes, it must be 1.
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    # 3. Fit and Transform the data
    # Note: LDA requires y_labels during the fit process (Supervised)
    X_lda = lda.fit_transform(X_scaled, y_train)

    print(f"Original shape: {X_train.shape}")
    print(f"Reduced shape: {X_lda.shape}")

    # Optional: Check how much 'variance' is explained by the new component
    print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
    return X_lda


def resize_image(img: np.ndarray, height: int, width: int) -> np.ndarray:
    # Get current dimensions of the image
    current_height, current_width = img.shape[:2]

    # If the image is square, just resize it
    if current_height == current_width:
        return cv2.resize(img, (width, height))

    # Calculate padding
    if current_height > current_width:
        pad_left = (current_height - current_width) // 2
        pad_right = current_height - current_width - pad_left
        padded_img = cv2.copyMakeBorder(
            img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    else:
        pad_top = (current_width - current_height) // 2
        pad_bottom = current_width - current_height - pad_top
        padded_img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    # Resize the padded image
    resized_img = cv2.resize(padded_img, (width, height))

    return resized_img


def preprocess_cxr(img_path, target_size=(256, 256)):
    img = dicom.read_dicom_image(img_path)

    denoised = processor.median_filter(img)

    normalized = processor.normalize_image(denoised)

    contrast = processor.adjust_contrast(normalized)

    resized = cv2.resize(contrast, target_size)

    # Resizing for prediction loses accuracy. So, for now resizing only for display, not for prediction.
    contrast = resize_image(contrast, 1024, 1024)

    return resized, contrast
