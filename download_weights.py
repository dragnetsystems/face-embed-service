from deepface import DeepFace

# List of models you want to pre-download
models = ["Facenet", "VGG-Face", "OpenFace", "DeepFace", "DeepID"]

for model_name in models:
    print(f"Downloading weights for {model_name}...")
    DeepFace.build_model(model_name)  # This will download and cache the weights
print("All model weights downloaded!")
