# Mediviz
A prototype for viewing and analyzing medical images using CV and AI.

## Ways to run the app:
1. Run the built (executable) from `dist/` for mac - the fastest
2. Run the python app
3. Build and run on local machine

## 1. Run the built (executable) from `dist/` for mac
The fastest and easiest.
1. Download the `mediviz` app from `dist` folder (in github) to your local machine
2. Double click `mediviz` app to run.

## 2. Run the python app

The project contents:
- config/ – configure model path, dataset (for scatter-plot analysis), and LBP parameters
- models/ – contains built models except the larger one (everything can be found here: https://drive.google.com/drive/folders/1k3odl1l5r9wkupQVh2W7f1euLRhd2X5Y)
- resources/ – images used by app
- libs/ – custom libraries used by notebooks and app
- app.py – the prototype app

To run the application:
Pre-requisite: Python was installed and configured in the local machine.
- locate into the dir after cloning the code locally: `cd mediviz`
- create virtual environment (to abstract pip installation): `python -m venv venv`
- activate the virtual environment (bash cmd): `source venv/bin/activate`
- activate the virtual environment (windows): `venv\Scripts\activate`
- install required libraries: `pip install -r requirements.txt`
- run the application in terminal: `python app.py`

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

```pyinstaller \
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
    app.py```