import streamlit as st
import pandas as pd
import joblib
import base64

# Load the saved XGBoost model
model_path = 'C:\\Users\\Hp\\Documents\\Red Wine Quality Prediction\\ml_model\\xgboost_model.pkl'
model = joblib.load(model_path)

# Define the expected columns excluding 'quality'
expected_columns = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates'
]

def classify_alcohol_strength(alcohol_percentage):
    if alcohol_percentage < 9:
        return 'LOW'
    elif alcohol_percentage < 10:
        return 'MEDIUM'
    else:
        return 'STRONG'

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    st.set_page_config(layout="wide")

    # Set the title of the web app with lighter red color
    st.markdown(
        "<h1 style='text-align: center; color: #ff6666;'>🍷 Red Wine Quality Prediction 🍷</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h2 style='text-align: center; color: #ff6666; padding: 10px; border-radius: 10px;'>Enter the characteristics of the wine to predict its alcohol percentage and strength:</h2>",
        unsafe_allow_html=True
    )

    # Convert image to base64 and use as background
    bg_image_base64 = get_base64_image('assets/bgimg.png')

    # Apply background image and style
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{bg_image_base64});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }}
        .stButton>button {{
            background-color: #800000;
            color: #FFFFFF;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #A52A2A;
        }}
        .stSlider>div {{
            color: #FFFFFF;
        }}
        .stSlider>div>div>div>input {{
            background: linear-gradient(90deg, #800000, #A52A2A);
            color: #FFFFFF;
            border-radius: 5px;
        }}
        .stTextInput>div>input {{
            background-color: #FFFFFF;
            color: #333333;
            border: 1px solid #800000;
            border-radius: 5px;
            padding: 10px;
        }}
        .stSelectbox>div>div>div>select {{
            background-color: #FFFFFF;
            color: #333333;
            border: 1px solid #800000;
            border-radius: 5px;
            padding: 10px;
        }}
        .stSubheader, .stMarkdown {{
            color: #333333;
            background-color: transparent;
            padding: 10px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create a two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Wine Characteristics')

        # Add input fields for features with improved text and help messages
        fixed_acidity = st.slider('Fixed Acidity', 4.0, 16.0, 7.4, help="The amount of non-volatile acids in the wine.")
        volatile_acidity = st.slider('Volatile Acidity', 0.1, 1.6, 0.7, help="The amount of volatile acids in the wine, which can lead to an unpleasant, vinegar taste.")
        citric_acid = st.number_input('Citric Acid', 0.0, 1.0, 0.0, step=0.01, help="The amount of citric acid in the wine, which can add freshness and flavor.")
        residual_sugar = st.number_input('Residual Sugar', 0.0, 15.0, 1.9, step=0.1, help="The amount of sugar remaining after fermentation stops.")
        chlorides = st.number_input('Chlorides', 0.01, 0.2, 0.076, step=0.01, help="The amount of salt in the wine.")
        free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 1.0, 70.0, 11.0, help="The free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion.")
        total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 6.0, 300.0, 34.0, help="The total amount of sulfur dioxide in the wine.")
        density = st.slider('Density', 0.9900, 1.0050, 0.9978, help="The density of the wine.")
        pH = st.number_input('pH', 2.0, 4.0, 3.51, step=0.01, help="Describes how acidic or basic the wine is on a scale from 0 (very acidic) to 14 (very basic).")
        sulphates = st.slider('Sulphates', 0.3, 2.0, 0.56, help="A wine additive which can contribute to the wine's aging potential.")

        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({
            'fixed acidity': [fixed_acidity],
            'volatile acidity': [volatile_acidity],
            'citric acid': [citric_acid],
            'residual sugar': [residual_sugar],
            'chlorides': [chlorides],
            'free sulfur dioxide': [free_sulfur_dioxide],
            'total sulfur dioxide': [total_sulfur_dioxide],
            'density': [density],
            'pH': [pH],
            'sulphates': [sulphates]
        })

        # Ensure columns are in the same order as during model training
        input_data = input_data[expected_columns]

    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                prediction = model.predict(input_data)
                alcohol_percentage = prediction[0]
                alcohol_strength = classify_alcohol_strength(alcohol_percentage)

                st.markdown(f"### Predicted Alcohol Percentage: **{alcohol_percentage:.2f}%**")
                st.markdown(f"### Alcohol Strength: **{alcohol_strength}**")

if __name__ == '__main__':
    main()
