# Mediviz
A prototype for viewing and analyzing medical images using CV and AI; regularly updated for stability.

## Project contents:
- config/ – configure model path, dataset (for scatter-plot analysis), and LBP parameters
- models/ – contains winner model. Other's can be found <a href="https://drive.google.com/drive/folders/1k3odl1l5r9wkupQVh2W7f1euLRhd2X5Y">here</a>: 
- resources/ – images used by app
- libs/ – custom libraries used by the app
- app.py – the prototype app

## Parameter configuration:
Default configuration in `config/app_config.py` using the winner model:
``` python
    dataset = rsna1000
    normal_patient_id = ""
    sample_normal_cxr = "resources/sample-normal-cxr.dcm"
    model_path = "models/svm_model_lda_default_1.pkl"
    train_dataset_path = "resources/train_dataset.h5"
    radius = 1
    method = "default"
    lda = True
```

## To run the code

Pre-requisite: Python was installed and configured in the local machine.
1. Clone/download the code to local machine: 
2. Locate into the code directory (in terminal from here): `cd mediviz`
3. Create virtual environment (to abstract execution env): `python -m venv venv`
4. Activate the virtual environment (bash cmd): `source venv/bin/activate`
5. Activate the virtual environment (windows): `venv\Scripts\activate`
6. Install required libraries: `pip install -r requirements.txt`
7. Run the application in terminal: `python app.py`
