# ⚡ Global Power Plant Capacity Prediction

An end-to-end Machine Learning and Data Science project focused on analyzing global energy infrastructure and predicting installed power plant capacity (`capacity_mw`) using the **Global Power Plant Database**.

The project covers the complete Machine Learning lifecycle, from exploratory data analysis and data preprocessing to model selection, hyperparameter optimization, model evaluation, API development, and Docker containerization.

The final solution uses an optimized **Random Forest Regressor** served through a REST API built with Flask and deployed inside a Docker container.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset](#dataset)
4. [Project Workflow](#project-workflow)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Development and Benchmarking](#model-development-and-benchmarking)
8. [Hyperparameter Optimization](#hyperparameter-optimization)
9. [Model Evaluation](#model-evaluation)
10. [Model Interpretation](#model-interpretation)
11. [API Development with Flask](#api-development-with-flask)
12. [Docker Deployment](#docker-deployment)
13. [Project Structure](#project-structure)
14. [Technologies and Tools](#technologies-and-tools)
15. [Installation and Setup](#installation-and-setup)
16. [Running the API](#running-the-api)
17. [Testing the Prediction Endpoint](#testing-the-prediction-endpoint)
18. [Key Findings and Limitations](#key-findings-and-limitations)
19. [Future Improvements](#future-improvements)

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
- Containerization with Docker.

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

Finally, the API was containerized using Docker, with dependencies managed through Pipenv and Gunicorn used as the WSGI server.

---

## 5. Exploratory Data Analysis

The exploratory analysis revealed several relevant characteristics of the dataset.

### Missing Values

A missing value analysis was conducted across all 36 features.

Some variables presented significant sparsity, particularly:

- Secondary fuel information.
- Historical electricity generation records.
- Certain ownership and administrative attributes.

These findings guided the preprocessing strategy.

### Target Variable Distribution

The installed capacity variable (`capacity_mw`) presented a heavily right-skewed distribution.

Most power plants have relatively small installed capacities, while a limited number of facilities account for extremely large values.

Logarithmic scaling was used during exploratory analysis to improve visualization.

### Geographic and Energy Analysis

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

### Hold-Out Test Set Results

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| **Random Forest (Optimized)** | **194.75** | **21.37** | **0.8677** |
| XGBoost (Optimized) | 229.79 | 24.80 | 0.8157 |

The Random Forest model was selected as the final model because it achieved better results across all three evaluation metrics.

### Performance Interpretation

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

### Outlier Behavior

The diagnostic plots showed relatively strong alignment with the ideal prediction line (`y = x`) for small and medium-sized power plants.

However, the model tends to underestimate some extremely large facilities.

This behavior is expected from tree-based models because Random Forest predictions are based on averaging values from terminal leaves and cannot naturally extrapolate beyond the observed target range.

Consequently, extreme mega-plants can contribute disproportionately to RMSE.

---


## 11. API Development with Flask

To make the trained Machine Learning model accessible, a REST API was developed using **Flask**.

The API loads the serialized Random Forest pipeline (`final_rf_pipeline.pkl`), including its preprocessing transformations, and provides an interface for generating predictions from new power plant data.

### API Architecture

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

### Available Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks API availability |
| `/predict` | POST | Predicts installed power plant capacity |

### Prediction Output

The `/predict` endpoint returns the estimated installed capacity in megawatts (MW), using the following JSON structure:

```json
{
    "predicted_capacity_mw": 52.4197
}
```

The API runs on port `9696` and uses **Gunicorn** as its WSGI server when deployed through Docker.

---

## 12. Instructions on How to Run the Project

Follow these steps to clone the repository, install dependencies, build the Docker image, and run the prediction service.

### 12.1 Clone the Repository

```bash
git clone https://github.com/Asier845/Global-Power-Plant-Capacity-Prediction.git

cd Global-Power-Plant-Capacity-Prediction
```

### 12.2 Install Dependencies

The project uses **Pipenv** for dependency management.

Install Pipenv:

```bash
pip install pipenv
```

Install the project dependencies:

```bash
pipenv install
```
```bash
pipenv shell
```


### 12.3 Build the Docker Image

Make sure Docker Desktop is running.

Build the Docker image from the project root directory:

```bash
docker build -t power-plant-capacity .
```

### 12.4 Run the Docker Container

Start the container and expose port `9696`:

```bash
docker run -d --name power-plant-service -p 9696:9696 power-plant-capacity
```

The service will be available at:

`http://localhost:9696`

### 12.5 Test the API

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

To generate a prediction, send a POST request with the power plant characteristics.

Example using PowerShell:

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

Useful commands to manage the Docker container:

**Check running containers:**

```bash
docker ps
```

**View application logs:**

```bash
docker logs power-plant-service
```

**Stop the container:**

```bash
docker stop power-plant-service
```

**Remove the container:**

```bash
docker rm power-plant-service
```

**Remove the Docker image:**

```bash
docker rmi power-plant-capacity
```

