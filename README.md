# 🌤️ Weather Analytics & Temperature Forecaster

An end-to-end Machine Learning project that analyzes weather and environmental data and predicts temperature using a tuned XGBoost regression model.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, time-based model evaluation, hyperparameter tuning, error analysis, and an interactive Streamlit web application for temperature prediction.

---

## 📌 Project Objective

The objective of this project is to build a machine learning-based temperature forecasting system that predicts temperature in Celsius using geographical, weather, environmental, air quality, and temporal features.

The project also analyzes weather patterns, seasonal variations, extreme temperature observations, and model prediction errors to understand the factors influencing temperature predictions.

---

## 🚀 Key Features

- 🌍 Location-based temperature prediction
- 📅 Date and time-based forecasting
- 🌦️ Weather condition analysis
- 🌫️ Air quality feature analysis
- 🌙 Seasonal and environmental feature analysis
- 📊 Exploratory Data Analysis (EDA)
- 🛠️ Feature engineering and temporal feature extraction
- 🔍 Data leakage detection and prevention
- ⏳ Time-based train-test evaluation
- 🤖 Multiple machine learning model comparison
- ⚡ Tuned XGBoost regression model
- 📈 Feature importance analysis
- ❄️ Extreme and cold-temperature analysis
- 📉 Prediction error and residual analysis
- 🌐 Interactive Streamlit web application
- 💾 Saved trained ML pipeline using Joblib

---

# 📊 Dataset & Data Analysis

The project uses a weather dataset containing observations from multiple countries and locations around the world.

## Dataset Overview

| Attribute | Value |
|---|---:|
| Total Records | 5,000 |
| Initial Features | 45 |
| Final Model Features | 31 |
| Unique Countries | 186 |
| Unique Locations | 217 |
| Duplicate Rows | 0 |
| Missing Values | 0 |
| Date Range | May 16, 2024 – May 22, 2026 |
| Target Variable | `temperature_celsius` |

## Target Variable Statistics

The target variable for the regression problem is `temperature_celsius`.

| Statistic | Temperature (°C) |
|---|---:|
| Count | 5,000 |
| Mean | 20.97 |
| Standard Deviation | 9.55 |
| Minimum | -24.20 |
| 25th Percentile | 15.20 |
| Median | 23.30 |
| 75th Percentile | 27.80 |
| Maximum | 46.70 |

## Dataset Characteristics

The dataset contains weather observations across different geographical locations and time periods.

Key observations from the analysis include:

- Data collected from **186 unique countries**.
- Data covers **217 unique locations**.
- The data includes observations from **2024, 2025, and 2026**.
- The dataset contains **25 unique months** across the available date range.
- The dataset contains **no duplicate rows**.
- No missing values were found in the analyzed dataset.
- Each location-date combination represents a unique observation in the dataset.

---

# 🛠️ Feature Engineering & Data Preprocessing

Several preprocessing and feature engineering steps were performed to prepare the dataset for machine learning.

## 1. Date-Time Feature Extraction

The `last_updated` timestamp was converted into useful temporal features:

- `year`
- `month`
- `day`
- `hour`
- `day_of_year`
- `day_of_week`

These features help the model capture seasonal and time-dependent temperature patterns.

## 2. Seasonal Feature

A categorical `season` feature was created to represent:

- Spring
- Summer
- Autumn
- Winter

The `season` feature was encoded using One-Hot Encoding during model preprocessing.

## 3. Cyclical Time Encoding

Cyclical transformations were applied to capture the continuous nature of periodic time features.

The following features were created:

- `month_sin`
- `month_cos`
- `hour_sin`
- `hour_cos`

This allows the model to understand relationships such as the similarity between December and January or between 23:00 and 00:00.

## 4. Selected Model Features

The final model uses **31 features** divided into numerical and categorical variables.

### Geographical Features

- `latitude`
- `longitude`

### Weather Features

- `wind_kph`
- `wind_degree`
- `pressure_mb`
- `precip_mm`
- `humidity`
- `cloud`
- `visibility_km`
- `uv_index`
- `gust_kph`

### Air Quality Features

- `air_quality_Carbon_Monoxide`
- `air_quality_Ozone`
- `air_quality_Nitrogen_dioxide`
- `air_quality_Sulphur_dioxide`
- `air_quality_PM2.5`
- `air_quality_PM10`
- `air_quality_us-epa-index`
- `air_quality_gb-defra-index`

### Environmental Features

- `moon_illumination`

### Temporal Features

- `year`
- `month`
- `day`
- `hour`
- `day_of_year`
- `day_of_week`

### Cyclical Features

- `month_sin`
- `month_cos`
- `hour_sin`
- `hour_cos`

### Categorical Feature

- `season`

## 5. Data Leakage Prevention

A leakage check was performed before model training.

The following features were removed because they directly represented the target temperature or were derived from it:

- `temperature_celsius`
- `temperature_fahrenheit`
- `feels_like_celsius`
- `feels_like_fahrenheit`
- `anomaly`

The final model predicts `temperature_celsius` without using these potentially leaking variables as input features.

## 6. Preprocessing Pipeline

A Scikit-learn `ColumnTransformer` and `Pipeline` were used to ensure consistent preprocessing and model training.

- Numerical features → `StandardScaler`
- Categorical feature (`season`) → `OneHotEncoder`
- Preprocessed features → Tuned XGBoost Regressor

This pipeline was saved as a single `.pkl` file using Joblib for use in the Streamlit application.

---

# 🤖 Model Development & Evaluation

Multiple regression models were trained and evaluated to identify the best-performing model for temperature prediction.

## Train-Test Strategy

A time-based train-test split was used to simulate real-world forecasting conditions.

| Dataset | Records | Years |
|---|---:|---|
| Training Set | 4,059 | 2024–2025 |
| Testing Set | 941 | 2026 |

The model was trained on historical observations from **2024 and 2025** and evaluated on unseen observations from **2026**.

This approach helps prevent future data from being used during training and provides a more realistic evaluation of forecasting performance.

## Model Comparison

| Model | MAE (°C) | RMSE (°C) | R² Score |
|---|---:|---:|---:|
| Linear Regression | 5.7460 | 7.5607 | 0.4704 |
| Random Forest | 3.0411 | 4.2558 | 0.8322 |
| XGBoost Baseline | 2.7976 | 3.9510 | 0.8554 |
| Tuned XGBoost | **2.7986** | **3.9315** | **0.8568** |
| Cold-Aware XGBoost | 2.7971 | 3.9590 | 0.8548 |

## Final Model

The **Tuned XGBoost Regressor** was selected as the final model based on its overall performance on the unseen 2026 test set.

Final performance:

- **MAE:** 2.80 °C
- **RMSE:** 3.93 °C
- **R² Score:** 0.857

On average, the model's predictions differ from the actual temperature by approximately **2.8 °C** based on the test-set MAE.

## Hyperparameter Tuning

Hyperparameter tuning was performed using cross-validation to improve the baseline XGBoost model.

The selected parameters were:

```text
learning_rate = 0.1
n_estimators = 700
max_depth = 5
min_child_weight = 5
subsample = 0.7
colsample_bytree = 0.7
