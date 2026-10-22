import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Configuration (Web page ka title aur look)
st.set_page_config(page_title="Netflix Content Predictor", page_icon="🎬", layout="centered")

# Web page ki heading
st.title("🎬 Netflix Content Type Predictor")
st.write("Yeh AI web app aap ke select kiye gaye features (Year, Rating, Country) ko dekh kar predict karegi ke content **Movie** hoga ya **TV Show**.")

# 2. Trained Model aur Encoders ko load karne ka function
@st.cache_resource
def load_model():
    # Yaad rahe ke aap ki 'netflix_model.pkl' file isi folder mein honi chahiye
    with open('netflix_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.success("AI Model Successfully Load Ho Gaya!")
except FileNotFoundError:
    st.error("Error: 'netflix_model.pkl' file nahi mili! Pehle model file is folder mein rakhein.")
    st.stop()

# 3. User Input Fields (Dabbe aur Dropdowns)
st.subheader("📊 Content Ki Details Enter Karein:")

# Release Year ka slider
release_year = st.slider("Release Year Select Karein:", min_value=2000, max_value=2026, value=2021)

# Rating ka dropdown (Kaggle data ke mutabiq common ratings)
rating_options = ['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'NR']
rating = st.selectbox("Content Rating:", rating_options)

# Country ka dropdown
country_options = ['United States', 'India', 'United Kingdom', 'Japan', 'South Korea', 'Canada']
country = st.selectbox("Country of Origin:", country_options)

# 4. Encoding (Text inputs ko numbers mein badalna jo model samajhta hai)
# Agar aap ne simple LabelEncoder use kiya tha, toh hum farzi mapping de rahe hain
# kyunke model ko numbers (0,1,2...) chahiye hote hain.
rating_mapping = {val: i for i, val in enumerate(rating_options)}
country_mapping = {val: i for i, val in enumerate(country_options)}

rating_encoded = rating_mapping[rating]
country_encoded = country_mapping[country]

# 5. Prediction Button
if st.button("🔮 Predict Content Type"):
    # Input data ka array banana jo model ko chahiye (release_year, rating, country)
    input_data = np.array([[release_year, rating_encoded, country_encoded]])
    
    # Model se prediction lena
    prediction = model.predict(input_data)
    
    # Kaggle code ke mutabiq target mapping check karna (0 = Movie, 1 = TV Show)
    st.subheader("🤖 AI Prediction:")
    if prediction[0] == 0:
        st.markdown("### 🎥 Yeh Aik **Movie** Hai!")
    else:
        st.markdown("### 📺 Yeh Aik **TV Show** Hai!")

# Footer text
st.markdown("---")
st.caption("Developed with 💖 using Streamlit & Python")
