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


The best cross-validation MAE obtained during hyperparameter tuning was:

```text
2.0089 °C

🔍 Error Analysis & Key Insights

After training the models, detailed error analysis was performed to understand where the final model performs well and where it struggles.

1. Residual Analysis

The residuals were analyzed using the unseen 2026 test set.

Key residual statistics:

Statistic	Value
Mean Residual	-0.95 °C
Residual Standard Deviation	3.82 °C
Minimum Residual	-17.38 °C
Maximum Residual	11.16 °C

The negative mean residual indicates that the model tends to slightly overpredict temperatures on average.

2. Extreme Temperature Analysis

The dataset was analyzed to identify extreme temperature observations.

Temperature Range	Number of Records
-30°C to -15°C	5
-15°C to -10°C	19
-10°C to 0°C	111
0°C to 10°C	592
10°C to 20°C	1,121
20°C to 30°C	2,532
30°C to 40°C	587
40°C to 50°C	33

Additional observations:

135 observations had temperatures at or below 0°C.
24 observations had temperatures at or below -10°C.
5 observations had temperatures at or below -15°C.
160 observations had temperatures at or above 35°C.
3. Cold Temperature Performance

The model showed significantly higher prediction error for cold-weather observations.

Category	Records	MAE (°C)
Cold (≤ 0°C)	55	9.31
Normal (> 0°C)	886	2.39

This indicates that the model performs much better for normal and warmer temperatures than for extreme cold temperatures.

A cold-aware XGBoost approach was also tested using sample weighting. However, it did not improve the overall test-set performance.

Model	MAE (°C)	RMSE (°C)	R² Score
Tuned XGBoost	2.80	3.93	0.857
Cold-Aware XGBoost	2.80	3.96	0.855

Therefore, the original Tuned XGBoost model was retained as the final model.

4. Feature Importance

Feature importance analysis was performed using the final XGBoost model.

The most influential features included:

season_Summer
uv_index
season_Winter
month_cos
pressure_mb
latitude
month_sin
hour_cos
hour
longitude

These results indicate that seasonal, geographical, temporal, and atmospheric factors play an important role in temperature prediction.

5. Worst Predictions

The highest prediction errors were primarily observed in extremely cold winter conditions.

Some of the largest errors occurred for locations such as:

Riga, Latvia
Bern, Switzerland
Ottawa, Canada
Ljubljana, Slovenia
Vilnius, Lithuania
Moscow, Russia
Kyiv, Ukraine

The model often underestimated the severity of extremely low temperatures, suggesting that the dataset contains relatively fewer extreme cold observations compared with normal temperature ranges.

Key Insights

The analysis suggests that:

Seasonal information is highly important for temperature prediction.
Geographical location, especially latitude, strongly influences temperature.
UV index and atmospheric pressure are important predictive variables.
The model performs well on typical temperature ranges.
Extreme cold temperatures are significantly more difficult to predict.
Time-based evaluation provides a more realistic estimate of forecasting performance than a random split.
Additional historical observations from cold regions and extreme winter conditions could potentially improve cold-weather prediction.
🌐 Streamlit Application

An interactive web application was developed using Streamlit to allow users to generate temperature predictions using the trained machine learning model.

Application Inputs

Users can provide the following information.

📍 Location & Date/Time
Latitude
Longitude
Prediction date
Prediction time
🌦️ Weather Conditions
Wind speed
Wind direction
Atmospheric pressure
Precipitation
Humidity
Cloud cover
Visibility
UV index
Wind gust speed
🌫️ Air Quality
Carbon Monoxide (CO)
Ozone (O₃)
Nitrogen Dioxide (NO₂)
Sulphur Dioxide (SO₂)
PM2.5
PM10
US EPA Air Quality Index
UK DEFRA Air Quality Index
🌙 Seasonal & Environmental Factors
Season
Moon illumination
Prediction Output

After entering the required information, the application generates:

🌡️ Predicted temperature in Celsius
🌤️ Temperature category
📋 Prediction summary
📍 Location information
📅 Prediction date and time
🌦️ Selected weather conditions
📊 Final model performance metrics
📈 Feature importance visualization
Example Prediction

The application can generate predictions such as:

Predicted Temperature: 28.86 °C
Temperature Category: 🌤️ Warm

The prediction summary also displays the selected location, date, time, humidity, wind speed, and other input conditions.

Model Performance Displayed in Application

The Streamlit application displays the performance of the final model:

MAE: 2.80 °C
RMSE: 3.93 °C
R² Score: 0.857

The model used in the application is the Tuned XGBoost Regression model, saved as:

weather_temperature_forecaster.pkl

The saved model includes the complete preprocessing and prediction pipeline, allowing the Streamlit application to process user inputs and generate predictions.

📁 Project Structure
Weather Analytics & Temperature Forecaster/
│
├── app.py
├── weather_temperature_forecaster.pkl
├── requirements.txt
├── README.md
│
├── data/
│   └── weather_data.csv
│
├── notebooks/
│   └── Weather_Trend_Forecasting.ipynb
│
└── screenshots/
    └── streamlit_app.png

Note: Update the file and folder names above if your actual project structure is different.

⚙️ Installation
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>

Navigate to the project directory:

cd Weather-Analytics-Temperature-Forecaster
2. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

The project uses the following key libraries:

Python
Pandas
NumPy
Matplotlib
Seaborn
Plotly
Folium
Scikit-learn
XGBoost
Joblib
Streamlit
3. Run the Streamlit Application

Run:

python -m streamlit run app.py

The application will open in your browser.

🧪 Model Compatibility

The trained model was created and tested using the following environment versions:

Python: 3.12.7
Scikit-learn: 1.5.1
XGBoost: 3.3.0
Joblib: 1.4.2

The Scikit-learn version is pinned in requirements.txt to help maintain compatibility with the saved machine learning pipeline.

Important: The saved .pkl model should be loaded using compatible library versions. In particular, using a different Scikit-learn version may cause compatibility errors when loading the serialized pipeline.

📈 Evaluation Metrics

The following metrics were used to evaluate the regression models.

Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted temperatures.

Lower MAE indicates better prediction accuracy.

Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction error.

RMSE gives greater weight to larger prediction errors.

R² Score

Measures how much of the variation in the target temperature is explained by the model.

A higher R² score indicates better explanatory performance.

🔮 Future Improvements

Potential future improvements include:

Incorporating larger historical weather datasets.
Adding more observations from extreme cold-weather regions.
Integrating real-time weather APIs.
Adding multi-day temperature forecasting.
Adding historical temperature trends for individual locations.
Exploring advanced time-series models.
Testing deep learning approaches such as LSTM networks.
Improving extreme temperature prediction.
Adding interactive geographical visualizations.
Deploying the application on a cloud platform.
🏆 Final Result

The project successfully developed an end-to-end machine learning pipeline for temperature prediction.

The final Tuned XGBoost model achieved:

Metric	Result
MAE	2.80 °C
RMSE	3.93 °C
R² Score	0.857

The trained model was integrated into an interactive Streamlit application, allowing users to enter geographical, weather, environmental, air quality, and temporal information and receive a predicted temperature.

The project demonstrates a complete machine learning workflow, including:

Data preprocessing
Exploratory data analysis
Feature engineering
Data leakage prevention
Time-based model evaluation
Model comparison
Hyperparameter tuning
Error analysis
Feature importance analysis
Model serialization
Streamlit deployment
👩‍💻 Author

Devangi Bhalala

BCA Graduate | Aspiring AI/ML Engineer

⭐ Acknowledgement

This project was developed as part of a Machine Learning Minor Project focused on applying data analysis, machine learning, model evaluation, and deployment concepts to a real-world weather prediction problem.


### One important correction

In your existing README, you wrote:

> "The selected parameters were:"

and then ended the code block. Make sure you add the **cross-validation result immediately after it**, as shown above.

Also, I recommend **not adding `data/weather_data.csv` to the Project Structure if you are not actually uploading the dataset to GitHub**. If the CSV is private, very large, or not part of your repository, remove that line.

After adding the above, your README is **complete**. The next step should be **GitHub upload/push and final project verification**, not more model improvements.
