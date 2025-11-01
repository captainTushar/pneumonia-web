from flask import Flask, render_template, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# Device
device = torch.device("cpu")

# 1️⃣ Create model architecture
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # binary classification

# 2️⃣ Load the saved state_dict
state_dict = torch.load("resnet18_pneumonia.pt", map_location=device)
model.load_state_dict(state_dict)

# 3️⃣ Move to device and set to evaluation mode
model.to(device)
model.eval()

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["image"]

    # Save uploaded image
    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)

    # Open and preprocess image
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    # Binary classification (0 = Normal, 1 = Pneumonia)
    prediction = torch.sigmoid(output).item()
    label = "Pneumonia" if prediction >= 0.5 else "Normal"

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
