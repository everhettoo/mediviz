# Libraries used.
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import libs.installer_lib as installer
from libs import feature_extractor as extractor


def load_dataset(dataset, split, radius, method):
    X = []
    y = []

    split_path = os.path.join(installer.resource_path(dataset.ROOT_DIR), split)
    failed = 0

    for label in ["normal", "pneumonia"]:
        class_path = os.path.join(split_path, label)

        for file in tqdm(os.listdir(class_path), desc=f"{split}-{label}"):
            img_path = os.path.join(class_path, file)

            try:
                # print(img_path)
                preprocessed, _ = extractor.preprocess_cxr(img_path)
                vectors = extractor.extract_feature_lbp(preprocessed, radius, method)
                X.append(vectors)
                y.append(0 if label == "normal" else 1)
            except Exception as e:
                print(f"exception: {e}")
                failed += 1
                continue

    print(f"Failed: {failed}")
    return np.array(X), np.array(y)


def scatter_plot(
    X_train,
    y_train,
    radius=1,
    method="default",
    patient_id=None,
    img_dir=None,
    resized=True,
):
    # Standardize the original dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Apply PCA (reduce to 2 dimensions)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot the results
    plt.figure(figsize=(8, 6))

    # Plot the original data points (normal vs pneumonia)
    plt.scatter(
        X_pca[y_train == 0, 0],
        X_pca[y_train == 0, 1],
        color="blue",
        label="Normal",
        alpha=0.7,
    )
    plt.scatter(
        X_pca[y_train == 1, 0],
        X_pca[y_train == 1, 1],
        color="red",
        label="Pneumonia",
        alpha=0.7,
    )

    if patient_id is not None and patient_id != "":
        if os.path.isdir(img_dir):
            path = os.path.join(img_dir, f"{patient_id}.dcm")
            if resized:
                pre_normal, _ = extractor.preprocess_cxr(path)
            else:
                _, pre_normal = extractor.preprocess_cxr(path)
            X_new = extractor.extract_feature_lbp(pre_normal, radius, method)
            # Shape (1, 1024)
            X_new = X_new.reshape(1, -1)

            # Apply PCA transformation to the new sample
            # Standardize the new sample
            X_new_scaled = scaler.transform(X_new)
            # Apply PCA to the new sample
            X_new_pca = pca.transform(X_new_scaled)

            # Plot the new sample
            plt.scatter(
                X_new_pca[0, 0],
                X_new_pca[0, 1],
                color="green",
                s=200,
                marker="*",
                label="New Sample",
                edgecolors="black",
            )
        plt.title("PCA: Normal vs Pneumonia with New Sample")
    else:
        plt.title("PCA: Normal vs Pneumonia")

    # Add labels and title
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()


def scatter_plot_ex(
    dataset,
    radius=1,
    method="default",
    patient_id=None,
    img_dir=None,
):
    # Load training set for scatter plot projection.
    X_train, y_train = load_dataset(dataset, "train", radius, method)

    # Standardize the original dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Apply PCA (reduce to 2 dimensions)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot the results
    plt.figure(figsize=(8, 6))

    # Plot the original data points (normal vs pneumonia)
    plt.scatter(
        X_pca[y_train == 0, 0],
        X_pca[y_train == 0, 1],
        color="blue",
        label="Normal",
        alpha=0.7,
    )
    plt.scatter(
        X_pca[y_train == 1, 0],
        X_pca[y_train == 1, 1],
        color="red",
        label="Pneumonia",
        alpha=0.7,
    )

    if patient_id is not None and patient_id != "":
        if os.path.isdir(img_dir):
            path = os.path.join(img_dir, f"{patient_id}.dcm")
            pre_normal, _ = extractor.preprocess_cxr(path)
            X_new = extractor.extract_feature_lbp(pre_normal, radius, method)
            # Shape (1, 1024)
            X_new = X_new.reshape(1, -1)

            # Apply PCA transformation to the new sample
            # Standardize the new sample
            X_new_scaled = scaler.transform(X_new)
            # Apply PCA to the new sample
            X_new_pca = pca.transform(X_new_scaled)

            # Plot the new sample
            plt.scatter(
                X_new_pca[0, 0],
                X_new_pca[0, 1],
                color="green",
                s=200,
                marker="*",
                label="New Sample",
                edgecolors="black",
            )

    # Add labels and title
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA: Normal vs Pneumonia with New Sample")

    # Add legend
    plt.legend()
    # Show the plot
    plt.show()


def lbp_difference_map(sample_lbp, patient_lbp, patient_img):
    # Difference map (absolute) using LBP values (not histogram)
    diff = np.abs(patient_lbp - sample_lbp)

    # Normalize to 0–1
    diff_norm = diff / (diff.max() + 1e-8)

    # Overlay heatmap with pneumonia - to show regions with high irregularities
    heatmap = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(
        cv2.cvtColor(patient_img, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0
    )

    return overlay_img


def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)
