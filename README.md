# Heart Disease Prediction: Binary Classification (Multi-Dataset Study)

## Project Overview

This repository documents a machine learning project focused on **binary classification of cardiovascular disease (CVD)** risk using four distinct public datasets. The primary goal is to assess the model's robustness and generalization capability across datasets sourced from different geographical regions and patient populations.

### Classification Goal

* **Binary Target:** `0` (No CVD / Healthy) or `1` (At Risk / Presence of CVD).
* **Original Data Adjustment:** Except Hungarian dataset, all three datasets initially contained multi-class risk levels (1-4). These have been merged into a single binary class (`1`) to focus solely on the presence or absence of the disease.

## Datasets Used

This project utilizes four well-known heart disease datasets, which were combined for comprehensive evaluation:

| Dataset Name | Source Location | Total Rows | Target Class Balance (Approx.) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Cleveland** | Cleveland Clinic Foundation | 303 | ~54% Healthy / 46% Risk | Commonly used for benchmarking. |
| **Hungarian** | Hungarian Institute of Cardiology | 294 | ~62% Healthy / 38% Risk | Contains a high number of missing values. |
| **Switzerland** | University Hospital, Zurich | 123 | ~54% Healthy / 46% Risk | Smaller, unique patient group. |
| **Long Beach** | V.A. Medical Center, Long Beach | 200 | ~62% Healthy / 38% Risk | Used to evaluate external validity. |
 
Dataset downloaded from : https://archive.ics.uci.edu/dataset/45/heart%2Bdisease?

Data Source Information:
   - (a) Creators: 
       -- 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
       -- 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
       -- 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
       -- 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
             Robert Detrano, M.D., Ph.D.
   - (b) Donor: David W. Aha (aha@ics.uci.edu) (714) 856-8779   
   - (c) Date: July, 1988


 ## Methodology and Key Steps

### 1. Data Preprocessing & Cleaning

This project utilized a multi-stage data cleaning strategy tailored to the specific nature of missing values identified across the heart disease sub-datasets (Cleveland, Hungarian, Switzerland, and Long Beach VA).

#### **1. Data Cleaning**
* The data was available in both raw and processed form with 13 features which were common in all 4 datasets, we used the 13-feature processed version of the dataset.

#####  Attribute Information

| Index | Feature Name | Description | Value Interpretation |
| :---: | :--- | :--- | :--- |
| 1 | `age` | Age in years | - |
| 2 | `sex` | Sex | `1` = Male; `0` = Female |
| 3 | `cp` | Chest pain type | `1` = Typical angina, `2` = Atypical angina, `3` = Non-anginal pain, `4` = Asymptomatic |
| 4 | `trestbps` | Resting blood pressure (mm Hg on admission) | - |
| 5 | `chol` | Serum cholestoral (mg/dl) | - |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | `1` = True; `0` = False |
| 7 | `restecg` | Resting electrocardiographic results | `0` = Normal, `1` = ST-T wave abnormality, `2` = Probable or definite left ventricular hypertrophy |
| 8 | `thalach` | Maximum heart rate achieved | - |
| 9 | `exang` | Exercise induced angina | `1` = Yes; `0` = No |
| 10 | `oldpeak` | ST depression induced by exercise relative to rest | - |
| 11 | `slope` | The slope of the peak exercise ST segment | `1` = Upsloping, `2` = Flat, `3` = Downsloping |
| 12 | `ca` | Number of major vessels (0-3) colored by flourosopy | - |
| 13 | `thal` | Thalassemia | `3` = Normal; `6` = Fixed defect; `7` = Reversable defect |
| 14 | `num` (Target) | Diagnosis of heart disease (Angiographic status) | `0` = < 50% diameter narrowing (No disease); `1` = > 50% diameter narrowing (Disease) |


* The raw data files, which lacked headers, were loaded and assigned clear, descriptive column names based on the UCI dataset dictionary.
* The non-standard missing value placeholder (`?`) was converted to the standard `NaN` to ensure correct handling by pandas.

##### Label Distribution in Four Datasets

![image](images/label_counts.png)


#### **2. Handling Missing Values (Two-Pronged Approach)**

##### Missing Values in four datasets
![image](images/missing_values.png)

Missing values were categorized into two types: Random Missing Values (RMV) and Systematic Missing Values (SMV).

* **Systematic Missing Values (SMV):**
    * For the Hungarian, Switzerland, and Long Beach VA datasets (which showed high systematic missingness), the **Attribute Deletion for Missing Value Handling (ADMVH)** technique was applied.
    * Any column missing **more than 50%** of its data was **removed** from the analysis to preserve data quality.

* **Random Missing Values (RMV):**
    * For the Cleveland dataset, and the remaining attributes of the hybrid datasets, the **Most Common Missing Value Imputation (MCMVI)** method was used.
    * Remaining `NaN` values were imputed using the **mode** (most frequent value) of their respective column.

 
### 2. Exploratory Data Analysis (EDA)
* **Missingness Visualization:** Used the `missingno` library (matrix, bar, and heatmaps) to visualize and compare missing data patterns across the four datasets.
* **Interactive Controls:** Implemented `ipywidgets` to dynamically switch between dataset views for comparison.

### 3. Model Training and Evaluation

Various established machine learning algorithms were used for classifications including ensemble methods like XGBoost and Random Forest, deep learning approaches like MLP, and classical models such as Logistic Regression and Support Vector Classifier (SVC). 
 
### 4. Results
 
The table below summarizes the performance metrics for all models used in the experiment, with values rounded to two decimal places.

| Model | f1 | accuracy | precision | recall | specificity | auroc |
|:---|---:|---:|---:|---:|---:|---:|
| GaussianNB | 0.76 | 0.75 | 0.77 | 0.75 | 0.50 | 0.75 |
| LogisticRegression | 0.73 | 0.78 | 0.72 | 0.78 | 0.12 | 0.72 |
| MLP | 0.67 | 0.72 | 0.63 | 0.72 | 0.00 | 0.65 |
| DecisionTree | 0.69 | 0.68 | 0.72 | 0.68 | 0.38 | 0.58 |
| KNeighbors | 0.72 | 0.75 | 0.69 | 0.75 | 0.12 | 0.61 |
| RandomForest | 0.70 | 0.72 | 0.68 | 0.72 | 0.12 | 0.61 |
| AdaBoost | 0.68 | 0.70 | 0.67 | 0.70 | 0.12 | 0.47 |
| SVC | 0.71 | 0.80 | 0.64 | 0.80 | 0.00 | 0.44 |
| XGBoost | 0.71 | 0.80 | 0.64 | 0.80 | 0.00 | 0.55 |

Across the nine evaluated classification models, the Gaussian Naive Bayes (GaussianNB) classifier demonstrated the most balanced performance, achieving the highest F1-score (0.760) and the best Area Under the Receiver Operating Characteristic (AUROC) value (0.754), alongside a specificity of 0.500. While the ensemble models XGBoost and SVC achieved the highest overall accuracy (0.800), this was paired with a concerning specificity of 0.000, indicating a complete inability to correctly identify negative cases. Conversely, the MLP model yielded the lowest F1-score (0.672), and the SVC model recorded the poorest discriminative power with the lowest AUROC (0.441). These results highlight a trade-off between maximizing raw accuracy and maintaining a balanced predictive capability across both classes, with the simplicity of the GaussianNB model providing superior generalization as measured by F1 and AUROC.


## How to Run

### Prerequisites

* Python 3.8+
* Jupyter Notebook or JupyterLab
* Dataset is assumed to be placed in `data/downloaded/` folder in the parent directory of this repo (this repo and data are in the same directory level).


1.  **Clone the repository:**
    ```bash
    git clone https://github.com/darshanz/ML-for-Cardiovascular-Disease-Diagnosis.git
    ML-for-Cardiovascular-Disease-Diagnosis.git
    ```
 
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ``` 

3.  **Run Experiments:**
     
     1. Data Exploration and Missing Valie Imputation (Notebook)
     2. Data Preparation for Training
     3. Experiment scripts
    ```bash
        cd src
        python main.py
    ```
     
 