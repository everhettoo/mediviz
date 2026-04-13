# Mediviz
A prototype for viewing and analyzing medical images using CV and AI; Incremental feature updates are pushed regularly..

## Ways to run the app:
A. Run the pre-built (executable) - the fastest \
B. Run the python app \
C. Build and run on local machine

## A. Run the pre-built (executable)
The fastest and easiest is running the pre-built app (for macOS).
1. Download the `mediviz` app from  <a href="https://drive.google.com/drive/u/0/folders/1RmV2LB0CJiYligOeRxa08z5hOWYEPiZ7">`app-dev`</a> folder (in this repo) to your local machine
2. Double click `mediviz` app to run

## B. Run the python app

Before running the python app, get familiar with the project contents:
- config/ – configure model path, dataset (for scatter-plot analysis), and LBP parameters
- models/ – contains built models except the larger one (everything can be found here: https://drive.google.com/drive/folders/1k3odl1l5r9wkupQVh2W7f1euLRhd2X5Y)
- resources/ – images used by app
- libs/ – custom libraries used by notebooks and app
- app.py – the prototype app

Pre-requisite: Python was installed and configured in the local machine.
1. locate into the dir after cloning the code locally: `cd mediviz`
2. create virtual environment (to abstract pip installation): `python -m venv venv`
3. activate the virtual environment (bash cmd): `source venv/bin/activate`
4. activate the virtual environment (windows): `venv\Scripts\activate`
5. install required libraries: `pip install -r requirements.txt`
6. run the application in terminal: `python app.py`

Sample in `config/app_config.py` using the winner model:
``` python
    dataset = rsna1000
    normal_patient_id = ""
    sample_normal_cxr = "resources/sample-normal-cxr.dcm"
    model_path = "models/svm_model_lda_default_1.pkl"
    radius = 1
    method = "default"
    lda = True
```

## C. Build and run on local machine
Pre-requisite: Step B. \
This will build the app using `pyinstaller` for running on local machine (macOS).
1. While still inside `mediviz/`, type the following in terminal: 
```python
pyinstaller \
    --name mediviz \
    --windowed \
    --icon=resources/mediviz.icns \
    --add-data "resources:resources" \
    --add-data "data:data" \
    --add-data "models:models" \
    --add-data "config:config" \
    --hidden-import sklearn.pipeline \
    --hidden-import sklearn.utils \
    --hidden-import sklearn.preprocessing \
    app.py
```
2. Locate to `dist/` and click the `mediviz` to run the app
