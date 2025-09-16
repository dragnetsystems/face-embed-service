from fastapi import FastAPI, UploadFile, File, Form
from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine
import json
import os
import shutil

app = FastAPI()

# Paths
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "deepface_weights")
FACENET_WEIGHTS = os.path.join(WEIGHTS_DIR, "facenet_weights.h5")
DEEPFACE_DEFAULT_WEIGHTS = os.path.join(os.path.expanduser("~/.deepface/weights"), "facenet_weights.h5")

# Ensure DeepFace weights directory exists and copy local weights
os.makedirs(os.path.dirname(DEEPFACE_DEFAULT_WEIGHTS), exist_ok=True)
shutil.copy2(FACENET_WEIGHTS, DEEPFACE_DEFAULT_WEIGHTS)

# --- Preload model ---
FACENET_MODEL = DeepFace.build_model("Facenet")
print("Facenet model loaded successfully.")

# Generate embedding from image bytes
def generate_embedding(image_bytes: bytes):
    temp_path = os.path.join(BASE_DIR, "temp.jpg")
    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    embedding = DeepFace.represent(
        img_path=temp_path,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend="opencv",
    )[0]["embedding"]

    os.remove(temp_path)
    return np.array(embedding, dtype=np.float32)

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float32).flatten()
    vec2 = np.array(vec2, dtype=np.float32).flatten()
    return 1 - cosine(vec1, vec2)

# Endpoint 1: Create embedding from uploaded photo
@app.post("/embed")
async def embed_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    embedding = generate_embedding(image_bytes)
    return {"embedding": embedding.tolist()}

# Endpoint 2: Verify face against multiple reference embeddings
@app.post("/verify")
async def verify(file: UploadFile = File(...), references: str = Form(...)):
    """
    references: JSON string of list of embeddings, e.g.
    [
        [0.1, 0.2, ...],
        [0.3, 0.4, ...]
    ]
    """
    image_bytes = await file.read()
    embedding = generate_embedding(image_bytes)

    reference_list = json.loads(references)
    reference_embeddings = [np.array(ref, dtype=np.float32).flatten() for ref in reference_list]


    similarities = [cosine_similarity(embedding, ref) for ref in reference_embeddings]
    max_similarity = max(similarities) if similarities else 0.0
    verified = max_similarity > 0.7
    face_detected = len(embedding) > 0

    return {
        "verified": verified,
        "similarity": float(max_similarity),
        "face_detected": face_detected
    }
