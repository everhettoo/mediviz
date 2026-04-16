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

## To run the app
Pre-requisite: Python (v3.12.9) was installed and configured in the local machine.

### Configure python version manager (PVM)
Python applications rely on specific interpreter versions and packages, and may break if run with a different version. A Python version manager (like pyenv) lets you install and switch between multiple Python versions so different projects can run without conflicts.

Install PyEnv:
1.	Verify if `pyenv` was installed: `pyenv --version`
2.	Continue if not installed. Use homebrew: `brew install pyenv`

Install and configure python 3.12.9 using PyEnv:
1. To install python version 3.12.9: `pyenv install 3.12.9`
2. To list installed python versions: `pyenv versions`
3. Set the global intepretor to 3.12.9: `pyenv global 3.12.9`
4. Verify the python version: `python -V`
5. Restart the terminal and try again if old version was displayed in step-4.

If you used `python3 -V` instead `python -V` (in step-5), pyenv needs to be configured globally. This will change system to call the standard `python` instead `python3`.
1. Update bash profile in the terminal: `echo 'eval "$(pyenv init --path)"' >> ~/.zshrc`


### Run the python code (app)
Run the code once the python intepretor is pointing to correct version (`3.12.9`).
1. Clone/download the code to local machine: 
2. Locate into the code directory (in terminal from here): `cd mediviz`
3. Create virtual environment (to abstract execution env): `python -m venv venv`
4. Activate the virtual environment (bash cmd): `source venv/bin/activate`
5. Activate the virtual environment (windows): `venv\Scripts\activate`
6. Install required libraries: `pip install -r requirements.txt`
7. Run the application in terminal: `python app.py`
