import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the trained model
with open('random_forest_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define your Streamlit application
def app():
    st.image('https://innomatics.in/wp-content/uploads/2023/01/innomatics-footer-logo.png')
    st.title("Image Classification")

    # Collect user input for the image
    image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # Add custom CSS to style the text only
    st.markdown("""
    <style>
    .detected-text {
        font-size: 24px;
        font-weight: bold;
        color: #3366ff;
    }
    .predicted-text {
        font-size: 24px;
        font-weight: bold;
        color: #ff5733;
    }
    </style>
    """, unsafe_allow_html=True)

    # Make predictions using the loaded model
    if st.button("Predict"):
        if image is None:
            st.error("Please upload an image first.")
        else:
            # Open and display the image
            img = Image.open(image)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            
            # Convert the image to grayscale
            img = img.convert('L')

            # Resize the image to 8x8
            img = img.resize((10, 10))
            
            # Flatten the image
            img = np.array(img)
            img = img.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(img)[0]

            # Convert encoded label back to the original class name
            if prediction == 0:
                class_name = "Cat"
            elif prediction == 1:
                class_name = "Cheetah"
            elif prediction == 2:
                class_name = "Cow"
            elif prediction == 3:
                class_name = "Dog"
            elif prediction == 4:
                class_name = "Elephant"
            elif prediction == 5:
                class_name = "Giraffe"
            elif prediction == 6:
                class_name = "Lion"
            elif prediction == 7:
                class_name = "Panda"
            elif prediction == 8:
                class_name = "Parrot"
            elif prediction == 9:
                class_name = "Penguin"
            else:
                class_name = "Unknown"

            # Display the results with larger and bold text
            st.markdown("<div class='detected-text'>Detected as:</div><div class='predicted-text'>{}</div>".format(class_name), unsafe_allow_html=True)

# Run the Streamlit application
if __name__ == "__main__":
    app()
