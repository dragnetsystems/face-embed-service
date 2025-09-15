from fastapi import FastAPI, UploadFile, File, Form
from deepface import DeepFace
import numpy as np
from numpy.linalg import norm
import json
from scipy.spatial.distance import cosine

app = FastAPI()

# Generate embedding from image bytes
def generate_embedding(image_bytes: bytes):
    with open("temp.jpg", "wb") as f:
        f.write(image_bytes)
    embedding = DeepFace.represent("temp.jpg", model_name="Facenet", enforce_detection=False)[0]["embedding"]
    return np.array(embedding, dtype=np.float32)

def cosine_similarity(vec1, vec2):
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
    # Read uploaded file and generate embedding
    image_bytes = await file.read()
    embedding = generate_embedding(image_bytes)

    # Parse JSON list of reference embeddings
    reference_list = json.loads(references)  # list of lists
    reference_embeddings = [np.array(ref, dtype=np.float32) for ref in reference_list]

    # Compute similarity against all references
    similarities = [cosine_similarity(embedding, ref) for ref in reference_embeddings]
    max_similarity = max(similarities) if similarities else 0.0
    verified = bool(max_similarity > 0.7)

    # Check if face was detected (DeepFace returns non-zero embedding even if fallback)
    face_detected = bool(len(embedding) > 0)

    return {
        "verified": verified,
        "similarity": float(max_similarity),
        "face_detected": face_detected
    }
