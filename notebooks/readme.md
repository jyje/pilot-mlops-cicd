# notebooks

## Description
This notebooks are for PoC of training and testing a model.


## Facts
We select setting up a python virtual environment for this notebook.

- Python: 3.12.8 with [pyenv](https://github.com/pyenv/pyenv)
- Library: [PyTorch 2.5](https://pytorch.org)
- Experiment Environment: MacOS


## Install
We use pyenv to manage python versions. If you don't have it, you can install it by following the instructions [here](https://github.com/pyenv/pyenv) or use another method to create a python virtual environment.

Run the following commands to install the python version and create a virtual environment.
```sh
git clone https://github.com/jyje/pilot-mlops-cicd.git -b main jyje-pilot-mlops-cicd
cd jyje-pilot-mlops-cicd

cd notebooks
pyenv install 3.12.8
pyenv virtualenv 3.12.8 jyje-pilot-mlops-cicd-notebook
pyenv local jyje-pilot-mlops-cicd-notebook
```

Then, run notebook what you want

## Notebook
- [note-0-just-test.ipynb](note-0-just-test.ipynb)
- [note-1-train-and-test.ipynb](note-1-train-and-test.ipynb)
