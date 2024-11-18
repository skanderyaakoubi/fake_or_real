
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Global variable to store the loaded model
model = None
labels = ['real', 'fake']  # Replace with your actual labels

# Function to load the trained model
def load_trained_model(model_path):
    global model
    if model is None:
        model = load_model(model_path)  # Load model if not already loaded
    return model

# Function to classify the image
def classify_image(file_path, model_path):
    # Load the trained model
    model = load_trained_model(model_path)

    # Load and preprocess the image
    image = Image.open(file_path)  # Read the image
    image = image.resize((128, 128))  # Resize image to fit the model input size
    img = np.asarray(image)  # Convert image to numpy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Normalize if required
    img = img / 255.0  # Optional: If your model expects normalized inputs

    # Predict the label
    predictions = model.predict(img)  # Get prediction from the model
    label = labels[np.argmax(predictions[0])]  # Extract label with max probability
    probab = float(round(predictions[0][np.argmax(predictions[0])]*100, 2))  # Confidence probability

    result = {
        'label': label,
        'probability': probab
    }

    return result

# Streamlit App
def main():
    # Title of the app
    st.title("Image Classification with Trained Model")

    # Upload image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # File path for the image
        file_path = uploaded_file

        # Path to your trained model
        model_path =  r"C:\Users\farid\Downloads\fakevsreal_weights.h5"  # Replace with your model path

        # Classify the uploaded image
        result = classify_image(file_path, model_path)

        # Display the result
        st.write(f"**Predicted Label**: {result['label']}")
# 
# Run the Streamlit app
if __name__ == "__main__":
    main()
