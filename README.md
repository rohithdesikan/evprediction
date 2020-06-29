evprediction
==============================

This project uses a 1500 home dataset to predict whether or not the house has an electric vehicle and when it is charging. 


# Overview
The goal of this project is to take in a data stream of smart meter power data for a home and do 2 things:
- Predict whether or not the house has an electric vehicle based on energy meter data. (Done)
- Predicts exactly when the house is charging (In Progress)

This repo is a package that anyone can download, use and develop further. 


### Data Format
- 2 months of half hour interval power reading data from smart meters some with electric vehicles and some without.
- The data split is ~70% without EVs and 30% with EVs
- The meter + labels data can be found in the data/interim folder


### Data Size
- 1590 houses with 2880 time stamped data points (Essentially 2 months of 1/2 hour interval data)


### EDA
There is some exploratory data analysis in the notebooks folder that shows the difference in profiles between houses with and without EVs


### Model Selection
Task 1: For this, an LSTM model (built in Tensorflow) is used where the 2880 points are split up into 60 time steps of 48 features each. Each day of meter data is a single time step. After experimenting with various architectures, this has been the most successful. 

Features are built and the model is set up and trained within the evprediction package (a local package installation as shown in setup.py)


### Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── .gitignore         <- Set up to ignore data files and 
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │── app                <- Contains the Flask app to which a post request can be sent to make predictions. No front end here, only post requests
    │
    ├── Dockerfile         <- The Dockerfile set up for the Flask app
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── evprediction       <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
