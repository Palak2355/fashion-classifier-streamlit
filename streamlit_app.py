import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from recommendation_system import classify_image, recommend_outfits, extract_color

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Shirt', 'Dress',
               'Coat', 'Sandal', 'Sneaker', 'Bag', 'Ankle boot']

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-top: 1rem;
    }
    .confidence-bar {
        width: 100%;
        background-color: #ddd;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    .confidence-fill {
        height: 20px;
        border-radius: 5px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI with enhanced UX
st.markdown('<h1 class="main-header">👗 Outfit Recommendation System (AI Stylist)</h1>', unsafe_allow_html=True)
st.markdown("### Upload a fashion item image to get outfit recommendations based on visual similarity.")

# Sidebar for filters
with st.sidebar:
    st.header("🎨 Filters")
    color_filter = st.checkbox("Match Color Palette")
    occasion = st.selectbox("Occasion", [None, "casual", "formal", "party"])
    season = st.selectbox("Season", [None, "summer", "winter", "spring", "fall"])

    st.header("ℹ️ About")
    st.write("This app uses CLIP (Contrastive Language-Image Pretraining) for classification and recommendations.")
    st.write("**Supported formats:** JPG, JPEG, PNG")
    st.write("**Database:** Fashion-MNIST test set")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Upload a clear image of a fashion item for recommendations.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True, output_format="auto")

        # Extract color if filter enabled
        extracted_color = None
        if color_filter:
            extracted_color = extract_color(image)
            st.write(f"**Dominant Color:** RGB{extracted_color}")

        if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Analyzing image and generating recommendations..."):
                predicted_label, confidence, recommendations = recommend_outfits(
                    image, color_filter=extracted_color, occasion=occasion, season=season
                )

            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.success(f"**Uploaded Item:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            # Confidence bar
            fill_color = "#4CAF50" if confidence >= 75 else "#FF9800" if confidence >= 50 else "#F44336"
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%; background-color: {fill_color};">
                    {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            if confidence < 75.0:
                st.warning("💡 **Tip:** Confidence is low. Try uploading a clearer image.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Recommendations
            if recommendations:
                st.header("👗 Recommended Outfit Items")
                cols = st.columns(len(recommendations))
                for i, rec in enumerate(recommendations):
                    with cols[i]:
                        st.image(rec['image'], caption=f"{rec['class']} (Sim: {rec['similarity']:.2f})", use_column_width=True)
            else:
                st.warning("No complementary items found with current filters.")
    else:
        st.info("👆 Upload an image to get started!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, CLIP, and PyTorch. [View Source Code](https://github.com/your-repo)")
