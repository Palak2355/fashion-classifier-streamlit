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

# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io

# -------------------------
# 1) Use the exact class_names
#    consistent with training order
# -------------------------
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Shirt', 'Dress',
    'Coat', 'Sandal', 'Sneaker', 'Bag', 'Ankle boot'
]

# -------------------------
# 2) Load model once (cached)
# -------------------------
@st.cache_resource
def load_fashion_model(path="best_deep_fashion_classifier_final.h5"):
    return load_model(path)

model = load_fashion_model()  # ensure this file exists in the same dir

st.title("👗 Fashion Classifier (Fashion-MNIST model)")
st.write("Upload an image and I'll predict its Fashion-MNIST class. Shows preprocessed 28×28 image and top-3 probs to help debugging.")

uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

# -------------------------
# 3) Robust preprocessing function
#    - convert to grayscale
#    - center-crop square (optional)
#    - resize to 28x28
#    - optional invert if background is bright
#    - normalize to [0,1]
# -------------------------
def preprocess_image(file_bytes, target_size=(28,28), invert_if_needed=True):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale
    # optional: center crop to square so aspect ratio doesn't distort small items
    w, h = img.size
    if w != h:
        min_side = min(w,h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        img = img.crop((left, top, left + min_side, top + min_side))
    img = img.resize(target_size, Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0  # normalize 0..1

    # Decide whether to invert: Fashion-MNIST has foreground bright, background dark.
    if invert_if_needed:
        mean_pixel = arr.mean()
        # If the image background appears light (mean close to 1), invert so clothing becomes bright
        if mean_pixel > 0.6:
            arr = 1.0 - arr

    # Some uploaded photos might have dark foreground on light background; the threshold can be tuned.
    arr = arr.reshape(1, target_size[0], target_size[1], 1)
    return arr, Image.fromarray((arr.reshape(target_size) * 255).astype(np.uint8))

# read bytes
file_bytes = uploaded.read()
preprocessed_arr, preview_img = preprocess_image(file_bytes)

# show original & processed
col1, col2 = st.columns(2)
with col1:
    st.image(file_bytes, caption="Original upload", use_column_width=True)
with col2:
    st.image(preview_img.resize((150,150)), caption="Preprocessed 28×28 (what the model sees)", use_column_width=False)

# -------------------------
# 4) Predict and show top-3
# -------------------------
preds = model.predict(preprocessed_arr, verbose=0)[0]
top3_idx = preds.argsort()[-3:][::-1]
st.subheader("Top predictions")
for idx in top3_idx:
    st.write(f"- **{CLASS_NAMES[idx]}** — {preds[idx]*100:.2f}%")

predicted_index = int(np.argmax(preds))
st.success(f"Predicted class: **{CLASS_NAMES[predicted_index]}** ({preds[predicted_index]*100:.2f}%)")

