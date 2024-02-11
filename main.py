from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import torch
from sklearn.cluster import KMeans
import numpy as np

# Initialize the model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def extract_embeddings(image_folder):
    """
    Extract embeddings for each image in the folder using the CLIP model.
    """
    embeddings = []
    image_paths = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if not os.path.isfile(image_path):
            continue
        
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        
        embeddings.append(outputs.cpu().numpy())
        image_paths.append(image_path)
    
    return np.vstack(embeddings), image_paths

def get_semantic_interpretation(image_path, possible_labels):
    """
    Get a semantic interpretation of what the image contains by matching it against a set of possible labels.
    """
    image = Image.open(image_path)
    inputs = processor(text=possible_labels, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        best_label_idx = probs.argmax().item()

    return possible_labels[best_label_idx]

def cluster_images(embeddings, n_clusters):
    """
    Cluster the image embeddings into n_clusters using k-means.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_

def sort_images_by_semantic_label(image_paths, possible_labels, output_folder):
    """
    Sort images into folders based on their semantic interpretation and print the assignment.
    """
    for image_path in image_paths:
        semantic_label = get_semantic_interpretation(image_path, possible_labels)
        print(f"{os.path.basename(image_path)} is most likely about: {semantic_label}")
        
        semantic_folder = os.path.join(output_folder, semantic_label)
        if not os.path.exists(semantic_folder):
            os.makedirs(semantic_folder)
        
        image_name = os.path.basename(image_path)
        output_path = os.path.join(semantic_folder, image_name)
        os.rename(image_path, output_path)
        print(f"Moved {image_name} to {semantic_folder}")

# Define your set of possible semantic labels
possible_labels = ["animal", "vehicle", "kid", "food", "person", "technology"]

# Main process
image_folder = "sample_images"
output_folder = "sorted_images_by_label"

# Extract embeddings and image paths (unchanged)
embeddings, image_paths = extract_embeddings(image_folder)

print(embeddings)

# No need to cluster images based on embeddings for this task

# Sort images into folders by semantic labels
sort_images_by_semantic_label(image_paths, possible_labels, output_folder)

