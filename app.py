from flask import Flask, render_template, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)


os.makedirs("uploads", exist_ok=True)

device = torch.device("cpu")


model = None

def load_model():
    global model
    if model is None:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)

        state_dict = torch.load("resnet18_pneumonia.pt", map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

    return model



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

    
    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)


    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    
    model_loaded = load_model()

    with torch.no_grad():
        output = model_loaded(img)

    prediction = torch.sigmoid(output).item()
    label = "Pneumonia" if prediction >= 0.5 else "Normal"

    return jsonify({"prediction": label})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
