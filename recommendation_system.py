import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from colorthief import ColorThief
import io

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Shirt', 'Dress',
               'Coat', 'Sandal', 'Sneaker', 'Bag', 'Ankle boot']

# Load CLIP model
@torch.no_grad()
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_clip_model()

# Load Fashion-MNIST test data as database
def load_fashion_database():
    try:
        test_df = pd.read_csv('fashion-mnist_test.csv')
        images = test_df.iloc[:, 1:].values.astype('float32').reshape(-1, 28, 28) / 255.0
        labels = test_df.iloc[:, 0].values
        # Convert grayscale to RGB by duplicating channels
        images_rgb = np.stack([images] * 3, axis=-1)  # Shape: (N, 28, 28, 3)
        # Resize to 224x224 for CLIP
        images_resized = []
        for img in images_rgb:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            pil_img = pil_img.resize((224, 224), Image.BILINEAR)
            images_resized.append(np.array(pil_img) / 255.0)
        images_resized = np.array(images_resized)
        return images_resized, labels
    except FileNotFoundError:
        print("Fashion-MNIST test.csv not found. Using dummy data.")
        return np.random.rand(100, 224, 224, 3), np.random.randint(0, 10, 100)

database_images, database_labels = load_fashion_database()

# Precompute embeddings for database
@torch.no_grad()
def precompute_embeddings(images):
    embeddings = []
    batch_size = 32
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", do_rescale=False)
        outputs = model.get_image_features(**inputs)
        embeddings.append(outputs.cpu().numpy())
    return np.vstack(embeddings)

database_embeddings = precompute_embeddings(database_images)

# Classification function using CLIP
@torch.no_grad()
def classify_image(image):
    # Prepare image for CLIP
    inputs = processor(images=image, return_tensors="pt", do_rescale=False)
    image_features = model.get_image_features(**inputs)

    # Use text prompts for classification
    text_prompts = [f"a photo of a {cls}" for cls in class_names]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**text_inputs)

    # Compute similarities
    similarities = torch.nn.functional.cosine_similarity(image_features, text_features, dim=-1)
    predicted_class_index = similarities.argmax().item()
    confidence = similarities[0][predicted_class_index].item() * 100

    return class_names[predicted_class_index], confidence

# Extract dominant color
def extract_color(image):
    # Convert PIL to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    color_thief = ColorThief(io.BytesIO(img_byte_arr))
    dominant_color = color_thief.get_color(quality=1)
    return dominant_color

# Color similarity (simple Euclidean distance)
def color_similarity(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))

# Recommendation function
def recommend_outfits(uploaded_image, top_k=5, color_filter=None, occasion=None, season=None):
    # Classify uploaded image
    predicted_class, confidence = classify_image(uploaded_image)

    # Get embedding for uploaded image
    with torch.no_grad():
        inputs = processor(images=uploaded_image, return_tensors="pt", do_rescale=False)
        uploaded_embedding = model.get_image_features(**inputs).cpu().numpy()

    # Compute similarities
    similarities = cosine_similarity(uploaded_embedding, database_embeddings)[0]

    # Get top similar items
    top_indices = np.argsort(similarities)[::-1][:top_k*2]  # Get more to filter

    recommendations = []
    for idx in top_indices:
        item_class = class_names[database_labels[idx]]
        sim_score = similarities[idx]

        # Basic complementary logic (simple rules)
        is_complementary = True
        if predicted_class in ['T-shirt/top', 'Shirt', 'Pullover'] and item_class in ['T-shirt/top', 'Shirt', 'Pullover']:
            is_complementary = False  # Avoid multiple tops
        elif predicted_class in ['Trouser'] and item_class in ['Trouser']:
            is_complementary = False  # Avoid multiple bottoms
        elif predicted_class in ['Sandal', 'Sneaker', 'Ankle boot'] and item_class in ['Sandal', 'Sneaker', 'Ankle boot']:
            is_complementary = False  # Avoid multiple shoes

        # Occasion filter (basic)
        if occasion:
            if occasion == 'casual' and item_class in ['Dress', 'Coat']:
                is_complementary = False
            elif occasion == 'formal' and item_class in ['Sandal', 'Sneaker']:
                is_complementary = False

        # Season filter (basic)
        if season:
            if season == 'summer' and item_class in ['Coat']:
                is_complementary = False
            elif season == 'winter' and item_class in ['Sandal']:
                is_complementary = False

        # Color filter
        if color_filter:
            item_color = extract_color(Image.fromarray((database_images[idx] * 255).astype(np.uint8)))
            if color_similarity(color_filter, item_color) > 100:  # Threshold
                is_complementary = False

        if is_complementary and len(recommendations) < top_k:
            recommendations.append({
                'class': item_class,
                'similarity': sim_score,
                'image': database_images[idx]
            })

    return predicted_class, confidence, recommendations
