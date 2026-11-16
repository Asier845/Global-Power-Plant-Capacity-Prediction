# ⚡ Global Power Plant Capacity Prediction

An end-to-end Machine Learning and Data Science project focused on analyzing global energy infrastructure and predicting installed power plant capacity (`capacity_mw`) using the **Global Power Plant Database**.

The project covers the complete Machine Learning lifecycle, from exploratory data analysis and data preprocessing to model selection, hyperparameter optimization, model evaluation, API development, Docker containerization, and cloud deployment.

The final solution uses an optimized **Random Forest Regressor** served through a REST API built with Flask, containerized with Docker, and deployed on **Render**.

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Dataset](#3-dataset)
4. [Project Workflow](#4-project-workflow)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Model Development and Benchmarking](#7-model-development-and-benchmarking)
8. [Hyperparameter Optimization](#8-hyperparameter-optimization)
9. [Model Evaluation](#9-model-evaluation)
10. [Model Interpretation](#10-model-interpretation)
11. [API Development with Flask](#11-api-development-with-flask)
12. [Instructions on How to Run the Project Locally](#12-instructions-on-how-to-run-the-project-locally)
13. [Cloud Deployment (Render)](#13-cloud-deployment-render)
14. [Technologies and Tools](#14-technologies-and-tools)
15. [Key Findings and Limitations](#15-key-findings-and-limitations)
16. [Future Improvements](#16-future-improvements)

---

## 1. Project Overview

The main objective is to develop a regression model capable of predicting the installed electricity generation capacity of power plants worldwide.

This project uses the **Global Power Plant Database**, provided by the World Resources Institute (WRI).

The dataset can be downloaded from:

https://datasets.wri.org/dataset/globalpowerplantdatabase

After downloading the dataset, place the CSV file in the following directory:

```text
data/
└── raw/
    └── global_power_plant_database.csv
```

Using geographical, structural, and operational characteristics, the model estimates the installed capacity of a power plant in megawatts (MW).

The project follows an end-to-end Machine Learning workflow, emphasizing:

- Reproducible data preprocessing.
- Robust model evaluation.
- Cross-validation and hyperparameter optimization.
- Prevention of data leakage.
- Model serialization and deployment.
- REST API development.
- Containerization with Docker and cloud hosting on Render.

The final model achieved an **R² score of 0.8677** on the hold-out test set, demonstrating strong predictive performance.

---

## 2. Business Problem

Understanding the installed capacity of power plants is essential for analyzing electricity generation infrastructure and evaluating energy systems.

Accurate capacity estimates can support:

- Energy infrastructure analysis.
- Electricity generation planning.
- Comparative analysis between countries.
- Renewable and non-renewable energy assessments.
- Identification of large-scale generation facilities.

This project approaches the problem as a supervised regression task.

**Target variable:** `capacity_mw`

**Prediction objective:** Estimate the installed capacity of a power plant based on its available characteristics.

---

## 3. Dataset

The project uses the **Global Power Plant Database**, a publicly available dataset containing information about power generation facilities worldwide.

| Characteristic | Description |
|---|---|
| Source | Global Power Plant Database |
| Approximate records | 35,000 power plants |
| Geographic coverage | 167 countries |
| Features | 36 variables |
| Target variable | `capacity_mw` |
| Problem type | Supervised Regression |

The dataset includes information about:

- Plant identification and ownership.
- Geographic coordinates.
- Country and administrative regions.
- Primary energy source.
- Secondary fuel sources.
- Commissioning year.
- Electricity generation records.
- Installed generation capacity.

---

## 4. Project Workflow

The project is divided into six main stages.

### 4.1 Exploratory Data Analysis (EDA) and Data Cleaning

An extensive exploratory analysis was conducted to understand the dataset structure, identify data quality issues, and discover relevant patterns.

Main activities:

- Dataset structure and data type inspection.
- Statistical summaries of numerical variables.
- Missing value analysis across all 36 features.
- Identification of highly sparse variables.
- Duplicate and inconsistent record inspection.
- Distribution analysis of installed capacity.
- Analysis of categorical variables.
- Geographic and country-level comparisons.
- Capacity aggregation by primary fuel type.

Special attention was given to the target variable because installed power plant capacity exhibits a highly right-skewed distribution.

### 4.2 Data Preprocessing and Feature Engineering

A reproducible preprocessing workflow was implemented using `scikit-learn` pipelines.

The preprocessing strategy included:

- Handling missing numerical values.
- Handling missing categorical values.
- Numerical feature scaling where appropriate.
- Categorical encoding using One-Hot Encoding.
- Feature selection and exclusion of identifier variables.
- Consistent transformation of training and test data.

The preprocessing pipeline was integrated with the regression models using `ColumnTransformer` and `Pipeline`.

This ensured that preprocessing transformations were fitted exclusively on the training data during cross-validation and model training.

### 4.3 Model Benchmarking

Four regression algorithms were evaluated to identify the most suitable approach.

| Model | Description |
|---|---|
| Dummy Regressor | Median prediction baseline |
| Linear Regression | Linear relationship baseline |
| Random Forest Regressor | Ensemble of decision trees |
| XGBoost Regressor | Gradient boosting algorithm |

Model performance was assessed using 5-fold cross-validation.

### 4.4 Hyperparameter Optimization

Hyperparameter optimization was performed using `RandomizedSearchCV`.

**Random Forest parameters:**
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `max_features`

**XGBoost parameters:**
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

Cross-validation was used during optimization to obtain a more robust estimate of model performance.

### 4.5 Final Evaluation and Model Diagnostics

The optimized models were evaluated on an independent hold-out test set (`X_test`).

The evaluation included:

- Root Mean Squared Error (RMSE).
- Mean Absolute Error (MAE).
- Coefficient of Determination (R²).
- Actual vs. Predicted scatter plots.
- Residual analysis.
- Feature importance analysis.

### 4.6 Model Deployment

After selecting the final model, the complete preprocessing and prediction pipeline was serialized into a `.pkl` file.

A REST API was developed using Flask to expose the model through HTTP requests.

Finally, the API was containerized using Docker, with dependencies managed through Pipenv, Gunicorn used as the WSGI server, and hosted publicly on Render.

---

## 5. Exploratory Data Analysis

The exploratory analysis revealed several relevant characteristics of the dataset.

### 5.1 Missing Values

A missing value analysis was conducted across all 36 features.

Some variables presented significant sparsity, particularly:

- Secondary fuel information.
- Historical electricity generation records.
- Certain ownership and administrative attributes.

These findings guided the preprocessing strategy.

### 5.2 Target Variable Distribution

The installed capacity variable (`capacity_mw`) presented a heavily right-skewed distribution.

Most power plants have relatively small installed capacities, while a limited number of facilities account for extremely large values.

Logarithmic scaling was used during exploratory analysis to improve visualization.

### 5.3 Geographic and Energy Analysis

Capacity distributions were also explored across:

- Countries.
- Primary fuel categories.
- Geographic regions.

These analyses helped identify differences in energy infrastructure and generation capacity across the dataset.

---

## 6. Data Preprocessing

A comprehensive preprocessing pipeline was implemented using **Scikit-learn** to prepare the dataset for regression modeling while ensuring consistency and preventing data leakage.

The target variable was defined as `capacity_mw`, representing the installed power generation capacity in megawatts.

### 6.1 Data Cleaning

The initial cleaning process included the following steps:

- Removal of non-predictive identification and metadata columns:
  - `gppd_idnr`
  - `name`
  - `source`
  - `url`
  - `wepp_id`
  - `geolocation_source`
  - `owner`
- Removal of records with missing target values (`capacity_mw`).
- Replacement of empty strings and whitespace-only values with `NaN` in categorical features.
- Conversion of numerical columns to numeric data types.
- Removal of invalid or non-finite target values.

### 6.2 Train-Test Split

The dataset was divided into training and testing subsets before fitting the preprocessing transformations.

| Parameter | Value |
|---|---|
| Training set | 80% |
| Testing set | 20% |
| Random state | 42 |
| Problem type | Regression |

The split was performed using `train_test_split`, ensuring that preprocessing transformations were fitted exclusively on the training data.

### 6.3 Feature Selection

After splitting the dataset, features were inspected using the training set.

The following columns were removed:

- Features containing only missing values.
- Features with one or fewer unique non-missing values (constant features).

The same columns were subsequently removed from the test set to guarantee identical feature structures.

### 6.4 Numerical Feature Transformation

Numerical features were processed using a Scikit-learn Pipeline consisting of two transformations:

| Transformation | Method | Purpose |
|---|---|---|
| Missing value imputation | Median Imputation | Replace missing numerical values |
| Scaling | RobustScaler | Reduce the influence of outliers during scaling |

Additionally, `SimpleImputer` was configured with `add_indicator=True` to generate missingness indicators for numerical features.

**Numerical preprocessing pipeline:**

```python
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    )),
    ("scaler", RobustScaler())
])
```

---

## 7. Model Development and Benchmarking

Four regression algorithms were evaluated.

| Model | Purpose |
|---|---|
| Dummy Regressor | Establish a baseline |
| Linear Regression | Evaluate linear relationships |
| Random Forest Regressor | Capture nonlinear relationships |
| XGBoost Regressor | Evaluate gradient boosting performance |

Five-fold cross-validation (`CV = 5`) was used to assess model performance.

The optimized Random Forest achieved the best overall results on the final test set.

---

## 8. Hyperparameter Optimization

`RandomizedSearchCV` was used to optimize the most relevant model hyperparameters.

The optimization process focused on:

- Model complexity.
- Number of estimators.
- Tree depth.
- Minimum samples required for splitting.
- Feature selection at each split.
- Learning rate and subsampling for XGBoost.

The best-performing configuration was selected based on cross-validation performance.

---

## 9. Model Evaluation

The optimized models were evaluated on an independent hold-out test set.

### 9.1 Hold-Out Test Set Results

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| **Random Forest (Optimized)** | **194.75** | **21.37** | **0.8677** |
| XGBoost (Optimized) | 229.79 | 24.80 | 0.8157 |

The Random Forest model was selected as the final model because it achieved better results across all three evaluation metrics.

### 9.2 Performance Interpretation

**R² = 0.8677**

The model explains approximately 86.77% of the variance in installed power plant capacity on the hold-out test set.

**MAE = 21.37 MW**

On average, predictions differ from the actual installed capacity by approximately 21.37 MW.

**RMSE = 194.75 MW**

The relatively high RMSE compared with MAE indicates the presence of some large prediction errors.

This is consistent with the highly skewed distribution of installed capacity and the existence of exceptionally large power plants.

---

## 10. Model Interpretation

Feature importance analysis was performed to understand which variables contributed most to the Random Forest predictions.

Additionally, Actual vs. Predicted scatter plots were generated to assess prediction quality and identify systematic errors.

### 10.1 Outlier Behavior

The diagnostic plots showed relatively strong alignment with the ideal prediction line (`y = x`) for small and medium-sized power plants.

However, the model tends to underestimate some extremely large facilities.

This behavior is expected from tree-based models because Random Forest predictions are based on averaging values from terminal leaves and cannot naturally extrapolate beyond the observed target range.

Consequently, extreme mega-plants can contribute disproportionately to RMSE.

---

## 11. API Development with Flask

To make the trained Machine Learning model accessible, a REST API was developed using **Flask**.

The API loads the serialized Random Forest pipeline (`final_rf_pipeline.pkl`), including its preprocessing transformations, and provides an interface for generating predictions from new power plant data.

### 11.1 API Architecture

The prediction service follows these steps:

1. Loads the trained Machine Learning pipeline.
2. Receives input data through HTTP POST requests.
3. Converts the JSON payload into a Pandas DataFrame.
4. Removes non-predictive columns when present.
5. Aligns the input features with the columns expected by the preprocessor.
6. Handles missing and empty values.
7. Applies the saved preprocessing transformations.
8. Generates a capacity prediction using the Random Forest model.
9. Returns the result as a JSON response.

### 11.2 Available Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks API availability |
| `/predict` | POST | Predicts installed power plant capacity |

### 11.3 Prediction Output

The `/predict` endpoint returns the estimated installed capacity in megawatts (MW), using the following JSON structure:

```json
{
    "predicted_capacity_mw": 52.4197
}
```

The API runs on port `9696` and uses **Gunicorn** as its WSGI server when deployed through Docker.

---

## 12. Instructions on How to Run the Project Locally

Follow these steps to clone the repository, install dependencies, build the Docker image, and run the prediction service locally.

### 12.1 Clone the Repository

```bash
git clone [https://github.com/Asier845/Global-Power-Plant-Capacity-Prediction.git](https://github.com/Asier845/Global-Power-Plant-Capacity-Prediction.git)
cd Global-Power-Plant-Capacity-Prediction
```

### 12.2 Install Dependencies

The project uses **Pipenv** for dependency management.

```bash
pip install pipenv
pipenv install
pipenv shell
```

### 12.3 Build the Docker Image

Make sure Docker Desktop is running, then build the image from the project root directory:

```bash
docker build -t power-plant-capacity .
```

### 12.4 Run the Docker Container

Start the container and expose port `9696`:

```bash
docker run -d --name power-plant-service -p 9696:9696 power-plant-capacity
```

The service will be available locally at `http://localhost:9696`.

### 12.5 Test the Local API

First, verify that the service is running:

```bash
curl http://localhost:9696/health
```

Expected response:

```json
{
    "status": "ok"
}
```

To generate a prediction, send a POST request with the power plant characteristics using PowerShell:

```powershell
$body = @{
    latitude = 28.6139
    longitude = 77.2090
    country = "IND"
    primary_fuel = "Solar"
    commissioning_year = 2015
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://localhost:9696/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Example output:

```text
predicted_capacity_mw
---------------------
52.4197
```

### 12.6 Docker Management

Useful commands to manage the local Docker container:

```bash
docker ps                              # Check running containers
docker logs power-plant-service        # View application logs
docker stop power-plant-service        # Stop the container
docker rm power-plant-service          # Remove the container
docker rmi power-plant-capacity        # Remove the Docker image
```

---

## 13. Cloud Deployment (Render)

The Dockerized REST API is deployed in the cloud on **Render**, making the prediction service publicly accessible without requiring local setup.

**Live Base URL:**

https://global-power-plant-capacity-prediction.onrender.com

> **Note on Free Tier Cold Starts:** Because this service runs on Render's free instance tier, the container automatically spins down after 15 minutes of inactivity. The first request after an idle period may take **30 to 50 seconds** to respond while the service wakes up; subsequent requests will execute immediately.

### 13.1 Health Check (Cloud)

Verify that the cloud service is live and responding:

```bash
curl [https://global-power-plant-capacity-prediction.onrender.com/health](https://global-power-plant-capacity-prediction.onrender.com/health)
```

Expected response:

```json
{
    "status": "ok"
}
```

### 13.2 Generate a Prediction (Cloud)

Send a `POST` request directly to the live `/predict` endpoint:

**Using `curl` (Linux / macOS / Git Bash):**

```bash
curl -X POST "[https://global-power-plant-capacity-prediction.onrender.com/predict](https://global-power-plant-capacity-prediction.onrender.com/predict)" \
     -H "Content-Type: application/json" \
     -d '{
           "latitude": 28.6139,
           "longitude": 77.2090,
           "country": "IND",
           "primary_fuel": "Solar",
           "commissioning_year": 2015
         }'
```

**Using PowerShell (Windows):**

```powershell
$body = @{
    latitude = 28.6139
    longitude = 77.2090
    country = "IND"
    primary_fuel = "Solar"
    commissioning_year = 2015
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "[https://global-power-plant-capacity-prediction.onrender.com/predict](https://global-power-plant-capacity-prediction.onrender.com/predict)" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Example output:

```text
predicted_capacity_mw
---------------------
52.4197
```

---

## 14. Technologies and Tools

- **Programming Language:** Python 3.14+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-learn, XGBoost, Joblib
- **Visualization:** Matplotlib, Seaborn
- **API & Web Server:** Flask, Gunicorn
- **Dependency Management:** Pipenv
- **Containerization & Cloud Hosting:** Docker, Render, Git/GitHub

---

## 15. Key Findings and Limitations

- **Primary Fuel & Country Dominance:** Primary energy source (`primary_fuel`) and geographic location (`country`, `latitude`, `longitude`) are among the strongest drivers of installed power plant capacity.
- **Right-Skewed Capacity Distribution:** The presence of mega-infrastructure facilities (e.g., large hydroelectric and nuclear power plants) creates heavy tails that disproportionately impact RMSE compared to MAE.
- **Extrapolation Constraints:** Tree-based ensemble methods such as Random Forest average predictions across leaf nodes, causing systematic underestimation on extreme outlier facilities beyond the typical training distribution.

---

## 16. Future Improvements

- **Target Transformation:** Experiment with log-transformation (`np.log1p(capacity_mw)`) during training to mitigate right-skewness and improve predictions on extreme mega-plants.
- **Interactive Frontend:** Build a lightweight web UI (e.g., Streamlit or HTML/JS form) connected to the Render API for interactive map-based capacity estimation.
- **CI/CD Automation:** Add GitHub Actions workflows to automatically run unit tests (`pytest`) and validate Docker builds upon every push to `main`.