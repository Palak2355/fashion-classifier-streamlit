import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pandas as pd

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Shirt', 'Dress',
               'Coat', 'Sandal', 'Sneaker', 'Bag', 'Ankle boot']

# Load the trained model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('best_deep_fashion_classifier_final.h5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_trained_model()

# Prediction function
def predict_image(image):
    # Convert to grayscale and resize to 28x28
    img = image.convert('L').resize((28, 28))
    # Convert to array, normalize, and reshape
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100
    return class_names[predicted_class_index], confidence

# Streamlit UI
st.title("Fashion Image Classifier")
st.write("Upload a fashion item image to classify it into categories like T-shirt, Dress, Sneaker, etc.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Classify"):
        if model is not None:
            predicted_label, confidence = predict_image(image)
            st.success(f"Predicted Class: **{predicted_label}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
            if confidence < 75.0:
                st.warning("Note: Confidence is low. The model may struggle with complex colors/backgrounds compared to the simple training images.")
        else:
            st.error("Model not loaded. Please check the model file.")


# Comparative Analysis Section - Minimal Version
st.header("📊 Model Comparative Analysis")

results = pd.DataFrame({
    'Model': [
        'Deep CNN (Our Model)', 
        'Multi-Layer Perceptron',
        'XGBoost',
        'Random Forest',
        'Support Vector Machine',
        'Logistic Regression',
        'AdaBoost', 
        'Decision Tree'
    ],
    'Test Accuracy': [0.927, 0.910, 0.885, 0.875, 0.864, 0.842, 0.852, 0.785],
    'Notes': [
        'Best performer - learns spatial features',
        'Strong but no spatial understanding', 
        'Gradient boosting ensemble',
        'Bagging ensemble method',
        'Kernel-based classifier',
        'Linear baseline model',
        'Adaptive boosting',
        'Prone to overfitting'
    ]
})

st.table(results.sort_values(by='Test Accuracy', ascending=False))

st.write("""
**Summary:** Our Deep CNN model outperforms all other approaches by effectively learning hierarchical features from image data, 
demonstrating the power of convolutional architectures for computer vision tasks.
""")



