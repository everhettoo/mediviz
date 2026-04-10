import os

"""
This module defines the paths to the RSNA's subset (500 rows for each class) dataset files.
"""

sample_size = 500
filename = f"rsna-{sample_size}.csv"
ROOT_DIR = os.path.join("data", f"rsna-{sample_size}")

normal_dir = os.path.join(ROOT_DIR, "normal/")
pneumonia_dir = os.path.join(ROOT_DIR, "pneumonia/")
abnormal_dir = os.path.join(ROOT_DIR, "abnormal/")
train_labels = os.path.join(ROOT_DIR, filename)
