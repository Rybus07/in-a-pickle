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

### Distribution & Access

Our project notebooks, scripts and materials are publicly available on GitHub and can be downloaded by any interested party as a TSV or recreated using our acquisition and cleaning code.

### Data Source Rights
Our project uses the **Pklmart's Competitive Pickleball Extracts** hosted on Kaggle.

- **Source:** Pklmart's Competitive Pickleball Extracts [Pklmart's Competitive Pickleball Extracts](https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts)
- **License:** Public API with [entire database freely available to all](https://thespacedevs.com/llapi), no documented restrictions for educational use
- **Our work:** Data collection, cleaning, integration, and documentation
- **Attribution:** Dataset credits Launch Library 2 API as source


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
**An example of how the scripts can be used in a new notebook can be found the the following lcation `test/project_module_test.ipynb`**


## Dataset Contents

Within `data/cleaned data` folder:

- **merged_data.tsv** - Complete dataset (7,336 launches, 39 attributes)
- **clean_rocket_data.tsv** - Rocket specifications (15 attributes)
- **clean_launch_data.tsv** - Launch details (16 attributes)
- **clean_mission_data.tsv** - Mission parameters (10 attributes)

Within `data/raw data` folder:

- **raw_baseline_launches_Group7.json.zip** - The complete dataset containing all collected launches. The notebook automatically compresses the raw JSON into this ZIP archive to comply with GitHub file size limits.
- **raw_baseline_launches_Group7_TEST.json.zip** - A smaller sample dataset (1,200 launches) generated when `01_Acquisition.ipynb` is run in `TEST_MODE`.

The `data_dictionary.csv` (located in the root directory of this repository) contains detailed information about each column in `merged_launch_data.tsv`, including data type and units when applicable.

More information, tables, figures, and code to interact with the final dataset are available in `Production Code/03_Merge/ipynb`.

### Dataset Overview

Quick statistics about the collected data:

| Metric | Value |
|--------|-------|
| Total Launches | 7,336 |
| Time Span | 1957-2025 |
| Countries | 47 |
| Unique Rockets | 450+ |
| Total Attributes | 39 |
| File Size | 4.6 MB |


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

4. All code is written to execute locally in-place without rearranging the directory structure. Crucially, each notebook is scripted to load and save data directly to the corresponding `Data/` folder within this repository (using relative paths), rather than a specific local absolute directory. If running in `Google Colab` or if a different directory structure is desired, update the code and file paths as needed.


### **4. GitHub File Structure:**
```
in-a-pickle/
├── data/
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
│   └── Project_Proposal.ipynb
|   |__ Project Report.ipynb
├── test/
│   └── project_module_test.ipynb
├── .gitignore
├── LICENSE
├── data_dictionary.csv
├── README.md
└── requirements.txt
```
For a detailed breakdown of all variables, data types, and units, please refer to our `data_dictionary.csv` located in the root directory of the repository. 

## Challenges, Limitations, and Alternatives

### API Limitations and Acquisition Strategy
The project faced significant hurdles regarding data access. We initially attempted to contact a different API provider but received no response, necessitating a pivot to the Launch Library 2 API. This API imposed a strict rate limit of 15 calls per hour with a maximum of 100 launches per call. This bottleneck required a custom pagination loop with a sleep timer, resulting in a total data acquisition time of approximately 5 to 6 hours. This process is documented in `01_Acquisition.ipynb`. During this long collection window, we also encountered intermittent network timeouts due to connectivity issues, requiring robust error handling in our scripts to ensure the loop could resume without data loss.

### Data Quality and Formatting
The final dataset contains null values in 21 out of 39 columns. This was largely due to inconsistent historical records, particularly missing data from early spaceflight launches. Additionally, we found that certain string variables contained commas, which interfered with standard CSV parsing. To resolve this, we adopted the Tab-Separated Values (TSV) format for storing our data.

We also faced challenges with variable specificity. For example, payload mass capability varies significantly with the target orbit, so a single "payload mass" column was insufficient. We addressed this by combining related variables to create new, more descriptive columns.

### Time Constraints and Workflow
Due to the project's limited timeline, we were unable to scour additional sources to fill in every missing variable. Managing the repository across multiple users also introduced complexity, as we frequently had to resolve merge conflicts within the GitHub repository.

### Alternatives Explored
To address missing values, we explored web scraping data from Wikipedia to supplement the API results. We considered utilizing the Wikipedia API directly to fill specific null values where possible. While we ultimately did not merge or include this supplemental data in our final dataset, this remains as a potential future direction.

Regarding the API rate limit, we considered an alternative acquisition strategy: filtering by time rather than simple pagination. This would allow a user to acquire all launches for a single specific year. While we ultimately chose the full pagination method for the final dataset, the code developed for the time-filtering approach is preserved under the `testing notebooks/API Calls` folder in `API call by year - test 1.ipynb` and `API test for 5 years data Interval.ipynb`.

## Help

**Common Issues:**

* **FileNotFoundError**: Our notebooks now use **Robust Directory Creation** to automatically build missing folders. If you see this error, ensure you are running the latest version of the code.
* **Google Drive Mount Issues**: You **do not** need to mount Google Drive to run our notebooks. The Hybrid Loading system streams data directly from the repository.
* **API Rate Limiting**: The Space Devs API allows 15 requests/hour. The pagination code includes automatic pausing.
* **Missing Values After Merge**: Normal - not all launches have complete data in the API.
* **Google Colab Session Timeout**: Files saved to Colab's temporary storage will be lost after the session timeout. Save important files to Google Drive to prevent data loss.
* **Size of JSON**: The full collection JSON exceeds 340MB. The acquisition script now automatically compresses this into a `.zip` archive (~30MB) so it can be safely committed to GitHub without hitting file size limits.

For questions about the project, contact team members (see the Authors section).

## Authors

DSCI 521 Sports Analytics Group

* [Ryan Peters](https://github.com/Rybus07) - rap369@drexel.edu
* [Andy Vong](https://github.com/PhillipJRoman) - av888@drexel.edu

## Version History

* 1.0 (March 2026)
    * Initial release - Complete data collection and preprocessing pipeline
    * 7,336 launches collected and cleaned
    * Three-way merge functionality implemented

## License

This project is for educational purposes on the Data analysis and interpretation pipeline for [Pklmart's Competitive Pickleball Extracts]((https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts)) from Kaggle, for DSCI 521 group term project at Drexel University.


## Acknowledgments

* [The Space Devs](https://thespacedevs.com/) - Launch Library 2 API
* [Space Devs API Documentation](https://ll.thespacedevs.com/docs/)
* [Stack Overflow pagination example](https://stackoverflow.com/questions/56206038/how-to-loop-through-paginated-api-using-python)
* DSCI 511 course materials and instructor and TAs
* Python standard libraries (`zipfile`, `io`, `os`) for enabling cloud-compatible data streaming
* [DataScientyst Guide](https://datascientyst.com/how-to-read-csv-directly-from-a-url-in-pandas-and-requests/) - Tutorial on reading CSV files from URLs using Pandas and Requests
* [Requests Documentation](https://requests.readthedocs.io/en/latest/user/quickstart/) - Official quickstart guide to stream data from URL
