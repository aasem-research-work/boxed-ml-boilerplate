# boxed-ml-boilerplate

Deploy your machine learning tasks (train, predict, etc)

## create environment

```sh
git clone https://github.com/aasem-research-work/boxed-ml-boilerplate.git
chdir boxed-ml-boilerplate

conda create -n boxedml python=3.9 --yes
conda activate boxedml
pip install -r requirements.txt
```

PS: Use ```pip freeze > requirements.txt``` when new packages are installed

## Deploy/Run

```sh
FLASK_APP=app.py FLASK_DEBUG=1 TEMPLATES_AUTO_RELOAD=1 flask run
```  

