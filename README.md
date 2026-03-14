## Machine Learning Approach to Pickleball Strategy
## DSCI 521 Term Project

Data analysis and interpretation pipeline for pklmart data from the Space Devs [Launch Library 2 API](https://thespacedevs.com/llapi), for DSCI 521 group term project at Drexel University.  

## Description

This project utilizes the pklmart's Competitive Pickleball Extracts dataset hosted on kaggle with the aim to create a machine learning model that can help players have better shot making 

## Distribution & Access

Our project notebooks, scripts and materials are publicly available on GitHub and can be downloaded by any interested party as a TSV or recreated using our acquisition and cleaning code.

### Data Source Rights
Our dataset uses the **Launch Library 2 API**, which provides public space launch data for research and educational purposes.

- **Source:** The Space Devs [Launch Library 2 API](https://ll.thespacedevs.com/)
- **License:** Public API with [entire database freely available to all](https://thespacedevs.com/llapi), no documented restrictions for educational use
- **Our work:** Data collection, cleaning, integration, and documentation
- **Attribution:** Dataset credits Launch Library 2 API as source

### How to Use This Dataset

**Option 1: Run the Notebooks (Recommended)**
Our notebooks (`02a`, `02b`, `02c`, `03`) include a **Hybrid Loading System**. They will automatically:
1.  Check your local machine for the data.
2.  If missing, stream the data directly from GitHub.
*This means you can run our analysis in Google Colab without manually uploading any files.*

**Option 2: Manual Loading**
If you wish to use the data in your own separate script:
```python
import pandas as pd
# Load the final merged dataset
df = pd.read_csv('data/cleaned data/merged_data.tsv', sep='\t')
```

**Quick overview:**
```python
print(f"Total launches: {len(df)}")
print(f"Date range: {df['Launch Date'].min()} to {df['Launch Date'].max()}")
print(f"Attributes: {len(df.columns)}")
```

### Dataset Contents

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

### Reproducing Our Collection
All collection and cleaning code is provided in this repository in the `Production Code` folder (see **Getting Started** for more detailed instructions):
1. Run `01_Acquisition.ipynb` to collect data
2. Run individual cleaning notebooks for each parameter set:
    * `02a_Rocket_Extraction.ipynb` - Extracts rocket parameters
    * `02b_Launch_Extraction.ipynb` - Extracts launch parameters
    * `02c_Mission_Extraction.ipynb` - Extracts mission parameters
4. Run `03_Merge.ipynb` to integrate

**Note:** Full collection takes ~5-6 hours due to API rate limiting (15 requests/hour, with a maximum of 100 launches/request)

### Data Quality
- Historical launches (pre-2000s) have incomplete technical specifications
- Missing values are documented in cleaning notebooks
- Launch IDs preserved for verification against API

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
git clone https://github.com/Rybus07/space-legends-data.git
```

2. Install required packages (if running locally) by referencing `requirements.txt` in the root directory

3. If using Google Colab, mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. All code is written to execute locally in-place without rearranging the directory structure. Crucially, each notebook is scripted to load and save data directly to the corresponding `Data/` folder within this repository (using relative paths), rather than a specific local absolute directory. If running in `Google Colab` or if a different directory structure is desired, update the code and file paths as needed.


### Executing Program

The project consists of three main phases:

**Phase 1: Data Collection**
* Open `01_Acquisition.ipynb` in `Production Code` folder
* Set TEST_MODE flag to `False` for full collection (5-6 hours due to rate limiting) or `True` for a quick validation run (~8 mins).
* Run notebook to collect raw launch data from API
* **Note:** The script automatically handles directory creation and file compression. It is fully compatible with Google Colab.
* Output: `raw_baseline_launches_Group7.json.zip` (Production) or `..._TEST.json.zip` (Test Mode).

**Phase 2: Data Cleaning**
* Open cleaning notebooks in `Production Code` folder:
  * `02a_Rocket_Extraction.ipynb` - Extracts rocket parameters
  * `02b_Launch_Extraction.ipynb` - Extracts launch parameters
  * `02c_Mission_Extraction.ipynb` - Extracts mission parameters
* Run each notebook to generate cleaned parameter files
* **Note:** All cleaning notebooks feature **Hybrid Data Loading**. They will automatically download the raw data from GitHub if run in a cloud environment (Colab).

**Phase 3: Data Merging**
* In `Production Code` folder, run `03_Merge.ipynb` to combine all cleaned data
* Output: `merged_data.tsv` saved to `data/cleaned data` folder - Final dataset ready for analysis

### **4. GitHub File Structure:**
```
in-a-pickle/
├── Production Code/
│   ├── 01_Acquisition.ipynb
│   ├── 02a_Rocket_Extraction.ipynb
│   ├── 02b_Launch_Extraction.ipynb
│   ├── 02c_Mission_Extraction.ipynb
│   └── 03_Merge.ipynb
├── Project materials/
│   ├── Presentation.pdf
│   └── Project_Proposal.ipynb
|   |__ Project Report.ipynb
├── data/
│   ├── cleaned data/
│   │   ├── clean_rocket_data.tsv
│   │   ├── clean_launch_data.tsv
│   │   ├── clean_mission_data.tsv
│   │   └── merged_data.tsv
│   └── raw data/
│       ├── raw_baseline_launches_Group7.json.zip
│       └── raw_baseline_launches_Group7_TEST.json.zip
├── testing notebooks/
│   └── [various testing notebooks]
├── .gitignore
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

* 1.0 (November 2025)
    * Initial release - Complete data collection and preprocessing pipeline
    * 7,336 launches collected and cleaned
    * Three-way merge functionality implemented

## License

This project is for educational purposes as part of DSCI 511 coursework at Drexel University.

Data sourced from [The Space Devs Launch Library 2 API](https://thespacedevs.com/llapi).


## Acknowledgments

* [The Space Devs](https://thespacedevs.com/) - Launch Library 2 API
* [Space Devs API Documentation](https://ll.thespacedevs.com/docs/)
* [Stack Overflow pagination example](https://stackoverflow.com/questions/56206038/how-to-loop-through-paginated-api-using-python)
* DSCI 511 course materials and instructor and TAs
* Python standard libraries (`zipfile`, `io`, `os`) for enabling cloud-compatible data streaming
* [DataScientyst Guide](https://datascientyst.com/how-to-read-csv-directly-from-a-url-in-pandas-and-requests/) - Tutorial on reading CSV files from URLs using Pandas and Requests
* [Requests Documentation](https://requests.readthedocs.io/en/latest/user/quickstart/) - Official quickstart guide to stream data from URL
