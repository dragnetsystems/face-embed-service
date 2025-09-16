from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine
import json
import os
import shutil

app = FastAPI()

# Add this **before your endpoints**
origins = [
    "*"  # Allow all origins for testing; later you can restrict to your React frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # which origins are allowed
    allow_credentials=True,
    allow_methods=["*"],            # allow all HTTP methods (POST, OPTIONS, etc.)
    allow_headers=["*"],            # allow all headers
)

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
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        embedding = generate_embedding(image_bytes)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

# Endpoint 2: Verify face against multiple reference embeddings
@app.post("/verify")
async def verify(file: UploadFile = File(...), references: str = Form(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    if not references:
        raise HTTPException(status_code=400, detail="References are required.")
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        embedding = generate_embedding(image_bytes)

        # Load references as JSON
        try:
            reference_data = json.loads(references)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="References must be valid JSON.")

        # Wrap single embedding into list
        if isinstance(reference_data, list) and len(reference_data) > 0 and isinstance(reference_data[0], (float, int)):
            reference_data = [reference_data]

        # Convert all to np arrays and flatten
        reference_embeddings = [np.array(ref, dtype=np.float32).flatten() for ref in reference_data]

        similarities = [cosine_similarity(embedding, ref) for ref in reference_embeddings]
        max_similarity = float(max(similarities)) if similarities else 0.0
        verified = bool(max_similarity > 0.7)         # convert to Python bool
        face_detected = bool(len(embedding) > 0)     # convert to Python bool

        return {
            "verified": verified,
            "similarity": max_similarity,
            "face_detected": face_detected
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process verification: {str(e)}")
