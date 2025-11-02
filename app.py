import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model
model = load_model("gender.h5")

# Set up the page
st.set_page_config(page_title="Gender Prediction", layout="centered")
st.markdown(
    """
    <style>
    .main {background-color: #f0f4f8;}
    .stButton>button {background-color: #6c63ff; color: white;}
    .stTextInput>div>input {background-color: #e0e7ff;}
    .stFileUploader>div>input {background-color: #e0e7ff;}
    .result-box {
        background: linear-gradient(90deg, #6c63ff 0%, #48c6ef 100%);
        color: white;
        padding: 1.5em;
        border-radius: 1em;
        text-align: center;
        font-size: 1.3em;
        margin-top: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Gender Prediction App")
st.write("Upload an image and let the AI predict the gender!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)[0]
    male_confidence = prediction[0] * 100
    female_confidence = (1 - prediction[0]) * 100

    if male_confidence > 50:
        predicted_label = "Male"
        predicted_confidence = male_confidence
        color = "#6c63ff"
    else:
        predicted_label = "Female"
        predicted_confidence = female_confidence
        color = "#48c6ef"

    st.markdown(
        f"""
        <div class="result-box" style="background: linear-gradient(90deg, {color} 0%, #e0e7ff 100%);">
            <strong>Predicted Gender:</strong> {predicted_label}<br>
            <strong>Confidence:</strong> {predicted_confidence:.2f}%<br>
            <span style="font-size:0.9em;">Male: {male_confidence:.2f}% | Female: {female_confidence:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )