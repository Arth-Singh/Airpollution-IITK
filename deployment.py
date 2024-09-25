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
    model = joblib.load('/teamspace/studios/this_studio/xgboost_aqi_model_20240925_185455.joblib')
    scaler = joblib.load('/teamspace/studios/this_studio/scaler.joblib')
    feature_names = joblib.load('/teamspace/studios/this_studio/feature_names.joblib')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('/teamspace/studios/this_studio/city_day.csv', parse_dates=['Date'])
    
    # Feature engineering (similar to your original script)
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

    # One-hot encode the 'City' and 'Season' columns
    df = pd.get_dummies(df, columns=['City', 'Season'], prefix=['City', 'Season'])

    return df

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

    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Policy Simulator", "Map Visualization"])

    if page == "Prediction":
        st.header("AQI Prediction")
        
        # Create input fields for features
        input_data = {}
        for feature in feature_names:
            if feature.startswith('City_') or feature.startswith('Season_'):
                input_data[feature] = st.selectbox(f"Select {feature}", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            else:
                input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

        if st.button("Predict AQI"):
            prediction = predict_aqi(input_data)
            st.success(f"Predicted AQI: {prediction:.2f}")

    elif page == "Policy Simulator":
        st.header("Policy Impact Simulator")
        
        # Create input fields for base scenario
        st.subheader("Base Scenario")
        base_input = {}
        for feature in feature_names:
            if feature.startswith('City_') or feature.startswith('Season_'):
                base_input[feature] = st.selectbox(f"Base {feature}", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            else:
                base_input[feature] = st.number_input(f"Base {feature}", value=0.0)

        # Create input fields for policy changes
        st.subheader("Policy Changes (% change)")
        policy_changes = {}
        for feature in ['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'SO2']:  # Add more relevant features
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
        map_data = df.groupby('City').agg({
            'AQI': 'mean',
            'Latitude': 'first',  # You need to add these columns to your dataset
            'Longitude': 'first'
        }).reset_index()

        # Create a map centered on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

        # Add markers for each city
        for idx, row in map_data.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=f"City: {row['City']}<br>AQI: {row['AQI']:.2f}",
                color='red',
                fill=True,
                fillColor='red'
            ).add_to(m)

        # Display the map
        folium_static(m)

if __name__ == "__main__":
    main()
