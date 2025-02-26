import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load a pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet')

st.title("AI Object Recognition")

uploaded_file = st.file_uploader("Upload an image",
                                 type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # Preprocess the image
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    # Make prediction
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.resnet50.decode_predictions(
        predictions, top=3)[0]
    st.write("### Predictions:")
    for pred in decoded_predictions:
        st.write(f"{pred[1]}: {pred[2]*100:.2f}%")
