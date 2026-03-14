# Machine Learning Approach to Pickleball Strategy

## Problem Statement
Pickleball is a game of rapid transitions and strategic positioning. While players often rely on "feel," there is a lack of quantitative modeling to determine the statistically "best" next shot in a given scenario.

**Project Goal:** 
This project utilizes the pklmart dataset to build a machine learning model that predicts shot outcomes based on spatial positioning, helping players optimize their decision-making to increase rally win probability.

**Users:** 
Pickleball players, pickleball analysts, and coaches

**Stakeholders:**
Pickleball club owners and professional organizations

## The Dataset

This project utilizes the pklmart Competitive Pickleball Extracts dataset to build a predictive machine learning model aimed at optimizing player decision-making. By leveraging shot-level spatial data, we recreate high-fidelity in-game scenarios to analyze how specific shot selections—such as placement and depth directly affect a team's probability of winning the rally.

** The Dataset Description can be found on [Kaggle](https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts)**

### Distribution & Access

Our project notebooks, scripts and materials are publicly available on GitHub and can be downloaded by any interested party.

### Data Source Rights
Our project uses the **Pklmart's Competitive Pickleball Extracts** hosted on Kaggle.

- **Source:** [Pklmart's Competitive Pickleball Extracts](https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts)
- **License:** CC BY-NC-SA 4.0
- **Our work:** Data collection, cleaning, feature engineering, preprocessing, and documentation
- **Attribution:** Dataset credits Pklmart as source


## 🛠 Methodology
This project follows a standard Data Science Lifecycle:

### Data Acquisition

All data acquisition is performed in *Notebook 1: (01_data_wrangling.ipynb)* using the [Pklmart's Competitive Pickleball Extracts]((https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts)) from Kaggle.

### Data Cleaning and Feature Engineering

Data Cleaning can be found in *Notebook 1: (01_data_wrangling.ipynb)*. The data was cleaned based on the following steps:
1. Merging the Shot and Rally Data
2. Handling null values within the data
3. Added additional features to the dataset including:
     - Shot Distance
     - Shot Angle
     - Change in x, y location
4. Standardizing the coordinate system

### Exploratory Data Analysis

EDA was perfromed in *Notebook 2: (02_eda.ipynb)*. The following is an overview of the EDA performed in this notebook:
1. Plotted univariate and bivariate distributions
    * Distributions showed trends in skill_lvl where players with skill_lvl of 4.5 and above played similarly and could be grouped into a larger class called 'Advanced'
    * Saw that cardinality of `shot_type` was high and needed to be reduced
2. Calculated adjacency matrices for my shot vs opponent's shot and my shot vs my next shot
    * While interesting, decided to go elsewhere with the project and pursue sequence prediction. Can come back to this later
3. Fit `rally_len` distributions to weibull distributions and log-normal
    * Lower skill levels are better fit with a weibull distribution
    * high skill levels had a longer tail and are better fit with a log-normal distribution
    * Fitted distributions are nice to have, but moving on with sequence prediction using RNNs


### Data Preprocessing

Preprocessing was performed in *Notebook 3: (03_Preprocessing_Feature_Engineering.ipynb)*. This is an overview of the steps performed:
1. Preprocessed data further for training
2. Created pipeline which grabs appropriate columns from dataframe and creates train, test splits
3. Trained a model using XGBoost to demonstrate process

### Modeling

We began drafting up initial model ideas for the shot outome prediction model in *Notebook 4: (04_av_training.ipynb)*. Currently remains a work in progress.

### 🏆 Results and Evaluation
Will be updated in the future....
 

### 💡 Insights and Conclusions
Will be updated in the future....


## How to Recreate This Project

**Option 1: Run the Notebooks (Recommended)**
Our notebooks (`01`, `02`, `03`) follow the project methodology stated above and should be run in that order:
1.  *Notebook 1: (01_data_wrangling.ipynb)* to clean the data and be ready for EDA.
2.  *Notebook 2: (02_eda.ipynb)* for EDA.
3.  *Notebook 3: (03_Preprocessing_Feature_Engineering.ipynb)* to create final dataframe for modeling phase.


**Option 2: Module Usage**
Alternatively you can reproduce our work using the following scripts:
- cleaning.py
- preprocessing.py
- train_classifier.py
  
**An example of how the scripts can be used in a new notebook can be found the the following lcation `test/project_module_test.ipynb`**


## Getting Started

### Dependencies
Reference `requirements.txt` in the root directory
* Python 3.8 or higher
* Required Python packages:
  * pandas
  * requests
  * json (built-in)
  * datetime (built-in)
  * dateutil
  * pprint (built-in)
* Google Colab or Jupyter Notebook environment; if using Colab, requires a Google Drive account for file storage

### Installing

1. Clone the repository to your local machine or Google Drive
```
git clone https://github.com/Rybus07/in-a-pickle.git
```

2. Install required packages (if running locally) by referencing `requirements.txt` in the root directory

3. If using Google Colab, mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```


### **4. GitHub File Structure:**
```
in-a-pickle/
├── data/
│   ├── raw/
│   │   ├── ball_type_ref.csv
│   │   ├── game.csv
│   │   ├── player.csv
│   │   ├── rally.csv
│   │   ├── shot.csv
│   │   ├── shot_type_ref.csv
│   │   └── team.csv
│   ├── interim/
│   │   ├── pre_clean_mdl_data.csv
│   │   ├── shot.csv
│   │   ├── shot_rally.csv
│   │   └── shot_rally_cleaned.csv
│   └── processed/
│       ├── clean_mdl_data.csv
│       └── clean_mdl_data_av.csv
├── notebooks/
│   ├── 01_data_wrangling.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_Preprocessing_Feature_Engineering.ipynb
│   ├── 04_av_trainig.ipynb
│   └── 04_av_trainig_ref.ipynb
├── scripts/
│   ├── cleaning.py
│   ├── preprocessing.py 
│   └── train_classifier.py 
├── test/
│   └── project_module_test.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
``` 

## Authors

DSCI 521 Sports Analytics Group

* [Ryan Peters](https://github.com/Rybus07) - rap369@drexel.edu
* [Andy Vong](https://github.com/PhillipJRoman) - av888@drexel.edu

## Version History

* 1.0 (March 2026)
    * Initial release - Complete data collection and data preprocessing pipeline
    * 
    * 

## License

This project is for educational purposes on the Data analysis and interpretation pipeline for [Pklmart's Competitive Pickleball Extracts]((https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts)) from Kaggle, for DSCI 521 group term project at Drexel University.


## References
* [Pklmart Website](https://pklmart.com/)
* [Pklmart's competitive pickleball extracts Data set](https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts)
* [XGBoost documentation (Release 3.2.0)]( https://xgboost.readthedocs.io/en/release_3.2.0/)
* [Scikit-learn Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
* [Scikit-learn Cross-validation: Evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
* [Scikit-learn Tuning the hyper-parameters of an estimator (Grid Search)](https://scikit-learn.org/stable/modules/grid_search.html)
