import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Weather Temperature Forecaster",
    page_icon="🌤️",
    layout="wide"
)


# ============================================
# LOAD TRAINED MODEL
# ============================================

@st.cache_resource
def load_model():
    return joblib.load(
        "weather_temperature_forecaster.pkl"
    )


model = load_model()


# ============================================
# APPLICATION TITLE
# ============================================

st.title(
    "🌤️ Weather Analytics & Temperature Forecaster"
)

st.write(
    "Predict temperature using machine learning "
    "based on weather, environmental, and air quality features."
)
# ============================================
# LOCATION AND DATE/TIME INPUT
# ============================================

st.header("📍 Location & Date/Time")

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=23.0225,
        step=0.01
    )

with col2:
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=72.5714,
        step=0.01
    )


col3, col4 = st.columns(2)

with col3:
    prediction_date = st.date_input(
        "Prediction Date"
    )

with col4:
    prediction_time = st.time_input(
        "Prediction Time"
    )

# ============================================
# WEATHER CONDITIONS INPUT
# ============================================

st.header("🌦️ Weather Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    wind_kph = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=200.0,
        value=10.0,
        step=0.1
    )

with col2:
    wind_degree = st.number_input(
        "Wind Direction (degrees)",
        min_value=0,
        max_value=360,
        value=180,
        step=1
    )

with col3:
    pressure_mb = st.number_input(
        "Pressure (mb)",
        min_value=800.0,
        max_value=1200.0,
        value=1013.0,
        step=0.1
    )


col4, col5, col6 = st.columns(3)

with col4:
    precip_mm = st.number_input(
        "Precipitation (mm)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=0.1
    )

with col5:
    humidity = st.slider(
        "Humidity (%)",
        min_value=0,
        max_value=100,
        value=60
    )

with col6:
    cloud = st.slider(
        "Cloud Cover (%)",
        min_value=0,
        max_value=100,
        value=50
    )


col7, col8, col9 = st.columns(3)

with col7:
    visibility_km = st.number_input(
        "Visibility (km)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1
    )

with col8:
    uv_index = st.number_input(
        "UV Index",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1
    )

with col9:
    gust_kph = st.number_input(
        "Wind Gust (km/h)",
        min_value=0.0,
        max_value=250.0,
        value=15.0,
        step=0.1
    )

# ============================================
# AIR QUALITY INPUT
# ============================================

st.header("🌫️ Air Quality")

col1, col2, col3 = st.columns(3)

with col1:
    air_quality_Carbon_Monoxide = st.number_input(
        "Carbon Monoxide (CO)",
        min_value=0.0,
        value=200.0,
        step=0.1
    )

with col2:
    air_quality_Ozone = st.number_input(
        "Ozone (O₃)",
        min_value=0.0,
        value=50.0,
        step=0.1
    )

with col3:
    air_quality_Nitrogen_dioxide = st.number_input(
        "Nitrogen Dioxide (NO₂)",
        min_value=0.0,
        value=20.0,
        step=0.1
    )


col4, col5, col6 = st.columns(3)

with col4:
    air_quality_Sulphur_dioxide = st.number_input(
        "Sulphur Dioxide (SO₂)",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

with col5:
    air_quality_PM2_5 = st.number_input(
        "PM2.5",
        min_value=0.0,
        value=20.0,
        step=0.1
    )

with col6:
    air_quality_PM10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=30.0,
        step=0.1
    )


col7, col8 = st.columns(2)

with col7:
    air_quality_us_epa_index = st.number_input(
        "US EPA Air Quality Index",
        min_value=1,
        max_value=6,
        value=1,
        step=1
    )

with col8:
    air_quality_gb_defra_index = st.number_input(
        "UK DEFRA Air Quality Index",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

# ============================================
# SEASONAL & ENVIRONMENTAL INPUT
# ============================================

st.header("🌙 Seasonal & Environmental Factors")

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox(
        "Season",
        options=[
            "Spring",
            "Summer",
            "Autumn",
            "Winter"
        ],
        index=1
    )

with col2:
    moon_illumination = st.slider(
        "Moon Illumination (%)",
        min_value=0,
        max_value=100,
        value=50
    )

# ============================================
# TEMPERATURE PREDICTION
# ============================================

st.header("🌡️ Temperature Prediction")


if st.button(
    "🚀 Predict Temperature",
    type="primary",
    use_container_width=True
):

    # ----------------------------------------
    # CREATE DATETIME
    # ----------------------------------------

    prediction_datetime = pd.Timestamp.combine(
        prediction_date,
        prediction_time
    )


    # ----------------------------------------
    # EXTRACT TIME FEATURES
    # ----------------------------------------

    year = prediction_datetime.year
    month = prediction_datetime.month
    day = prediction_datetime.day
    hour = prediction_datetime.hour

    day_of_year = prediction_datetime.dayofyear
    day_of_week = prediction_datetime.dayofweek


    # ----------------------------------------
    # CYCLICAL TIME FEATURES
    # ----------------------------------------

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )


    # ----------------------------------------
    # CREATE INPUT DATAFRAME
    # ----------------------------------------

    input_data = pd.DataFrame(
        {
            "latitude": [latitude],
            "longitude": [longitude],

            "wind_kph": [wind_kph],
            "wind_degree": [wind_degree],
            "pressure_mb": [pressure_mb],
            "precip_mm": [precip_mm],
            "humidity": [humidity],
            "cloud": [cloud],
            "visibility_km": [visibility_km],
            "uv_index": [uv_index],
            "gust_kph": [gust_kph],

            "air_quality_Carbon_Monoxide": [
                air_quality_Carbon_Monoxide
            ],

            "air_quality_Ozone": [
                air_quality_Ozone
            ],

            "air_quality_Nitrogen_dioxide": [
                air_quality_Nitrogen_dioxide
            ],

            "air_quality_Sulphur_dioxide": [
                air_quality_Sulphur_dioxide
            ],

            "air_quality_PM2.5": [
                air_quality_PM2_5
            ],

            "air_quality_PM10": [
                air_quality_PM10
            ],

            "air_quality_us-epa-index": [
                air_quality_us_epa_index
            ],

            "air_quality_gb-defra-index": [
                air_quality_gb_defra_index
            ],

            "moon_illumination": [
                moon_illumination
            ],

            "year": [year],
            "month": [month],
            "day": [day],
            "hour": [hour],

            "day_of_year": [
                day_of_year
            ],

            "day_of_week": [
                day_of_week
            ],

            "season": [season],

            "month_sin": [
                month_sin
            ],

            "month_cos": [
                month_cos
            ],

            "hour_sin": [
                hour_sin
            ],

            "hour_cos": [
                hour_cos
            ]
        }
    )


    # ----------------------------------------
    # MAKE PREDICTION
    # ----------------------------------------

    prediction = model.predict(
        input_data
    )[0]
    # ============================================
    # MAKE PREDICTION
    # ============================================

    prediction = model.predict(
        input_data
    )[0]

    # ============================================
    # DISPLAY PREDICTION RESULT
    # ============================================

    st.success(
        "Temperature prediction generated successfully!"
    )

    st.subheader("🌡️ Predicted Temperature")

    st.metric(
        label="Forecasted Temperature",
        value=f"{prediction:.2f} °C"
    )


    # ============================================
    # TEMPERATURE CATEGORY
    # ============================================

    if prediction < 0:
        temperature_category = "❄️ Freezing"
    elif prediction < 10:
        temperature_category = "🥶 Very Cold"
    elif prediction < 20:
        temperature_category = "🌥️ Cool"
    elif prediction < 30:
        temperature_category = "🌤️ Warm"
    elif prediction < 40:
        temperature_category = "☀️ Hot"
    else:
        temperature_category = "🔥 Extremely Hot"


    st.info(
        f"Temperature Category: **{temperature_category}**"
    )


    # ============================================
    # PREDICTION SUMMARY
    # ============================================

    st.subheader("📋 Prediction Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.write("📍 **Location**")
        st.write(
            f"Latitude: {latitude:.2f}"
        )
        st.write(
            f"Longitude: {longitude:.2f}"
        )

    with summary_col2:
        st.write("📅 **Date & Time**")
        st.write(
            prediction_datetime.strftime(
                "%d %B %Y"
            )
        )
        st.write(
            prediction_datetime.strftime(
                "%I:%M %p"
            )
        )

    with summary_col3:
        st.write("🌦️ **Weather Conditions**")
        st.write(
            f"Humidity: {humidity}%"
        )
        st.write(
            f"Wind Speed: {wind_kph:.1f} km/h"
        )
# ============================================
# MODEL PERFORMANCE
# ============================================

st.divider()

st.header("📊 Model Performance")

st.write(
    "The temperature forecasting model was evaluated "
    "using a time-based test set containing unseen "
    "2026 weather observations."
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        label="MAE",
        value="2.80 °C"
    )

with metric_col2:
    st.metric(
        label="RMSE",
        value="3.93 °C"
    )

with metric_col3:
    st.metric(
        label="R² Score",
        value="0.857"
    )

st.caption(
    "Model: Tuned XGBoost | "
    "Evaluation: Time-Based Test Set"
)
# ============================================
# FEATURE IMPORTANCE
# ============================================

st.divider()

st.header("📈 Top Factors Influencing Temperature")

st.write(
    "The following features had the greatest influence "
    "on the Tuned XGBoost model's temperature predictions."
)

feature_importance_data = pd.DataFrame(
    {
        "Feature": [
            "Summer Season",
            "UV Index",
            "Winter Season",
            "Month (Cosine)",
            "Pressure",
            "Latitude",
            "Month (Sine)",
            "Hour (Cosine)",
            "Hour",
            "Longitude"
        ],
        "Importance": [
            0.238424,
            0.132350,
            0.123224,
            0.089805,
            0.086874,
            0.076118,
            0.029952,
            0.027136,
            0.024732,
            0.023922
        ]
    }
)

st.bar_chart(
    feature_importance_data.set_index(
        "Feature"
    )
)

st.caption(
    "Feature importance values are derived from "
    "the trained Tuned XGBoost model."
)