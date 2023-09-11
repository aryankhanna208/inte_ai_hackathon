import streamlit as st
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
from streamlit_lottie import st_lottie

# Load a pre-trained MobileNetV2 model for image classification
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Set the page title and add some custom CSS
def load_lottieurl(url):
    r = requests.get(url)
    return r.json()


lottie_AI = load_lottieurl("https://lottie.host/532607d4-7bdd-4a68-be08-4f925ba64844/BfB1Ai3Gcz.json")

st.set_page_config(
    page_title="Image Recognition App",
    page_icon="🖼️",
    layout="wide"
)

# custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

st.title("Image Recognition App")
st.write("Upload an image to identify its contents.")

# Function to recognize the image
def recognize_image(image):
    # Preprocess the image
    img = tf.image.decode_image(image.read(), channels=3)  # Ensure 3 color channels (RGB)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)

    # Predict the image
    prediction = model.predict(img)
    decoded_prediction = tf.keras.applications.mobilenet_v2.decode_predictions(prediction)
    
    return decoded_prediction[0]



# Upload an image

st_lottie(lottie_AI, height = 500)
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Send the image to the FastAPI server for prediction
    files = {'file': ('image.jpg', uploaded_image.read(), 'image/jpeg')}
    response = requests.post("http://localhost:8000/predict/", files=files)

    if response.status_code == 200:
        result = response.json()
        class_name = result["class_name"]


    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Add a loading message
    with st.spinner("Classifying..."):
        # Make predictions
        prediction = recognize_image(uploaded_image)

    # Display the top 5 predictions
    if class_name == "pneumonia":
        st.subheader(f"Pneumonia Detected")
    else:
        st.subheader(f"Pneumonia not detected")
    

