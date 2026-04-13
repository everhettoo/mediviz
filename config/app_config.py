import os
import sys

import libs.installer_lib as installer
from libs.dataset import rsna1000

# def resource_path(relative_path):
#     """Get absolute path to resource, works for dev and for PyInstaller"""
#     try:
#         # PyInstaller creates a temp folder and stores path in _MEIPASS
#         base_path = sys._MEIPASS
#     except Exception:
#         base_path = os.path.abspath(".")

#     return os.path.join(base_path, relative_path)


dataset = rsna1000
normal_patient_id = ""
sample_normal_cxr = installer.resource_path("resources/sample-normal-cxr.dcm")
# Configured with the winner model and its dataset for scatter plot.
model_path = installer.resource_path("models/lr_model_lda_default_1.pkl")
train_dataset_path = installer.resource_path("resources/train_dataset.h5")
radius = 1
method = "default"
lda = True
