import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/xgboost_aqi_model_20240925_185455.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, feature_names = load_model()

# Load and preprocess the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/city_day.csv', parse_dates=['Date'])
        
        # Feature engineering
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'

        df['Season'] = df['Month'].apply(get_season)

        # Calculate rolling averages
        for col in ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']:
            df[f'{col}_Rolling_Mean_7'] = df.groupby('City')[col].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

        # Create lag features
        for col in ['AQI', 'PM2.5', 'PM10']:
            df[f'{col}_Lag_1'] = df.groupby('City')[col].shift(1)

        # Save original City column
        df['Original_City'] = df['City']

        # One-hot encode the 'City' and 'Season' columns
        df = pd.get_dummies(df, columns=['City', 'Season'], prefix=['City', 'Season'])

        return df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# Function to preprocess input data
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0

    # Reorder columns to match feature_names
    input_df = input_df[feature_names]

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    input_imputed = imputer.fit_transform(input_df)

    # Scale features
    input_scaled = scaler.transform(input_imputed)

    return input_scaled

# Function to make prediction
def predict_aqi(input_data):
    preprocessed_data = preprocess_input(input_data)
    prediction = model.predict(preprocessed_data)
    return prediction[0]

# Function to simulate policy impact
def simulate_policy_impact(base_input, policy_changes):
    base_aqi = predict_aqi(base_input)
    modified_input = base_input.copy()
    for feature, change in policy_changes.items():
        if feature in modified_input:
            modified_input[feature] *= (1 + change)
    new_aqi = predict_aqi(modified_input)
    impact = new_aqi - base_aqi
    return base_aqi, new_aqi, impact

# Streamlit app
def main():
    st.title("AQI Prediction and Policy Impact Simulator")

    if df is None or model is None:
        st.error("Failed to load data or model. Please check the error messages above.")
        return

    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Policy Simulator", "Map Visualization"])

    if page == "Prediction":
        st.header("AQI Prediction")
        
        # Create input fields for features
        input_data = {}

        # City selection dropdown
        city_columns = [col for col in feature_names if col.startswith('City_')]
        selected_city = st.selectbox("Select City", [col.replace('City_', '') for col in city_columns])
        for city_col in city_columns:
            input_data[city_col] = 1 if city_col == f'City_{selected_city}' else 0

        # Season selection dropdown
        season_columns = [col for col in feature_names if col.startswith('Season_')]
        selected_season = st.selectbox("Select Season", [col.replace('Season_', '') for col in season_columns])
        for season_col in season_columns:
            input_data[season_col] = 1 if season_col == f'Season_{selected_season}' else 0

        # Sliders for numerical inputs
        numerical_features = [f for f in feature_names if not f.startswith(('City_', 'Season_'))]
        for feature in numerical_features:
            # You might want to adjust min_value and max_value based on your data
            input_data[feature] = st.slider(f"{feature}", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

        if st.button("Predict AQI"):
            prediction = predict_aqi(input_data)
            st.success(f"Predicted AQI: {prediction:.2f}")

    elif page == "Policy Simulator":
        st.header("Policy Impact Simulator")
        
        # Create input fields for base scenario
        st.subheader("Base Scenario")
        base_input = {}
        
        # City selection dropdown for base scenario
        city_columns = [col for col in feature_names if col.startswith('City_')]
        selected_city = st.selectbox("Select Base City", [col.replace('City_', '') for col in city_columns])
        for city_col in city_columns:
            base_input[city_col] = 1 if city_col == f'City_{selected_city}' else 0

        # Season selection dropdown for base scenario
        season_columns = [col for col in feature_names if col.startswith('Season_')]
        selected_season = st.selectbox("Select Base Season", [col.replace('Season_', '') for col in season_columns])
        for season_col in season_columns:
            base_input[season_col] = 1 if season_col == f'Season_{selected_season}' else 0

        # Sliders for numerical inputs in base scenario
        numerical_features = [f for f in feature_names if not f.startswith(('City_', 'Season_'))]
        for feature in numerical_features:
            base_input[feature] = st.slider(f"Base {feature}", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

        # Create input fields for policy changes
        st.subheader("Policy Changes (% change)")
        policy_changes = {}
        for feature in ['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'SO2']:
            policy_changes[feature] = st.slider(f"Change in {feature}", -100.0, 100.0, 0.0) / 100

        if st.button("Simulate Policy Impact"):
            base_aqi, new_aqi, impact = simulate_policy_impact(base_input, policy_changes)
            st.success(f"Base AQI: {base_aqi:.2f}")
            st.success(f"New AQI: {new_aqi:.2f}")
            st.info(f"Policy Impact: {impact:.2f}")

            # Visualize the impact
            fig, ax = plt.subplots()
            sns.barplot(x=['Base AQI', 'New AQI'], y=[base_aqi, new_aqi], ax=ax)
            ax.set_ylabel('AQI')
            ax.set_title('Policy Impact on AQI')
            st.pyplot(fig)

    elif page == "Map Visualization":
        st.header("AQI Map Visualization")
        
        # Prepare data for map
        city_data = df.groupby('Original_City').agg({
            'AQI': 'mean'
        }).reset_index()

        # Add latitude and longitude (you'll need to provide this data)
        city_coordinates = {
            'Delhi': (28.6139, 77.2090),
            'Mumbai': (19.0760, 72.8777),
            # Add more cities and their coordinates
        }

        city_data['Latitude'] = city_data['Original_City'].map(lambda x: city_coordinates.get(x, (0, 0))[0])
        city_data['Longitude'] = city_data['Original_City'].map(lambda x: city_coordinates.get(x, (0, 0))[1])

        # Create a map centered on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

        # Add markers for each city
        for idx, row in city_data.iterrows():
            if row['Latitude'] != 0 and row['Longitude'] != 0:
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5,
                    popup=f"City: {row['Original_City']}<br>AQI: {row['AQI']:.2f}",
                    color='red',
                    fill=True,
                    fillColor='red'
                ).add_to(m)

        # Display the map
        folium_static(m)

if __name__ == "__main__":
    main()
