# TODO: Outfit Recommendation System (AI Stylist)

- [x] Update requirements.txt with CLIP dependencies (transformers, torch, colorthief for color extraction)
- [x] Modify streamlit_app.py to integrate CLIP for classification and embeddings
- [x] Implement outfit recommendation logic using cosine similarity on CLIP embeddings
- [x] Add filters: color palette (extract from image), occasion (user input), season (user input)
- [x] Use Fashion-MNIST test set as item database (convert to RGB for CLIP)
- [x] Test the updated app locally (app runs, but may need user interaction for full test)
- [x] Fix classification accuracy by replacing CNN with CLIP
