import streamlit as st
import requests
from PIL import Image
import io

st.title("Sentiment Analysis from Images")

st.write("Upload an image to analyze its sentiment.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Prepare the image for sending to the Flask backend
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Send the image to the Flask backend for prediction
    response = requests.post("http://localhost:5000/predict", files={"image": img_bytes})

    if response.status_code == 200:
        predictions = response.json()
        st.write("Predictions:")
        st.write(f"Humour: {predictions['humour']}")
        st.write(f"Sarcasm: {predictions['sarcasm']}")
        st.write(f"Offensive: {predictions['offensive']}")
        st.write(f"Motivational: {predictions['motivational']}")
        st.write(f"Extracted Text: {predictions['extracted_text']}")
    else:
        st.write("Error in prediction:", response.json().get('error', 'Unknown error'))