import os

"""
This module defines the paths to the RSNA dataset files.
"""

ROOT_DIR = os.path.join("data", "rsna-kaggle")

train_labels = os.path.join(ROOT_DIR, "stage_2_train_labels.csv")
class_info = os.path.join(ROOT_DIR, "stage_2_detailed_class_info.csv")
images_dir = os.path.join(ROOT_DIR, "stage_2_train_images/")

# Class labels to be used in the code.
normal = 0
pneumonia = 1
abnormal = 2
