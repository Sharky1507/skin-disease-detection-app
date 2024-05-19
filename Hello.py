import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
from tensorflow.keras.models import load_model




# Load the pre-trained model
model = load_model('my_model.h5')


st.title("Skin Disease Image Classification")

        
# Define a function to preprocess the input image
def preprocess_image(image):
          # Convert PIL image to numpy array
          image_array = np.array(image)
          
          # Convert RGB to BGR
          image_array = image_array[:, :, ::-1]

          # Resize the image to 224x224 pixels
          image_array = cv2.resize(image_array, (224, 224))
          
          # Apply the same preprocessing as during training/testing
          image_array = preprocess_input(np.array([image_array]))  # Add an extra dimension for batching

          return image_array

# Create the Streamlit app
def main():
    st.title("Skin Disease Image Classification")

    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image and make a prediction
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)
        class_names = ['BA-Cellulitis','BA-impetigo','FU-athlete-foot','FU-nail-fungus','FU-ringworm','PA-cutaneous-larva-migrans','VI-chickenpox','VI-shingles']


        # Display the prediction result
        st.write(f"You have: {class_names[predicted_class]}! Please visit your Dermatologist")
        

# Run the Streamlit app
if __name__ == "__main__":
    main()
#skindiseasedetectionapp-l71m0roefqr
