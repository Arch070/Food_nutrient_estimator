from flask import Flask, request, jsonify
import pickle
import pandas as pd
from scipy.sparse import hstack
from PIL import Image
import io
import torch
from torchvision import models, transforms

app = Flask(__name__)

with open("multi_rf.pkl", "rb") as f:
    multi_rf = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

numeric_features = ['Fat','Protein','Carbohydrates','Vitamin B12','Vitamin B6','Vitamin C',
                    'Vitamin D','Calcium','Iron','Magnesium','Zinc']
avg_values = {
    'Fat': 10, 'Protein': 10, 'Carbohydrates': 10, 'Vitamin B12': 0.5, 'Vitamin B6': 0.2,
    'Vitamin C': 5, 'Vitamin D': 0.1, 'Calcium': 50, 'Iron': 1, 'Magnesium': 20, 'Zinc': 1
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model = models.resnet18(pretrained=True)
image_model.eval()
image_model.to(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

food_classes = ["apple", "banana", "broccoli", "carrot", "chicken", "egg", "salmon", "almonds", "yogurt", "bread"]
@app.route("/predict", methods=["POST"])
def predict_text():
    food_name = request.form.get("food_name")
    if not food_name:
        return jsonify({"error": "No food name provided"}), 400

    new_food = pd.DataFrame({
        'Food': [food_name],
        **{feature: [avg_values[feature]] for feature in numeric_features}
    })
    new_X_text = vectorizer.transform(new_food['Food'])
    new_X_numeric = new_food[numeric_features]
    new_X = hstack([new_X_numeric, new_X_text])
    
    prediction = multi_rf.predict(new_X)[0]
    return jsonify({
        "Calories": float(prediction[0]),
        "Protein": float(prediction[1]),
        "Carbs": float(prediction[2]),
        "Fat": float(prediction[3])
    })


@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = image_model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        # Map predicted index to your food names (simplified)
        food_name = food_classes[predicted.item() % len(food_classes)]

    # Use same text model to get nutrition
    new_food = pd.DataFrame({
        'Food': [food_name],
        **{feature: [avg_values[feature]] for feature in numeric_features}
    })
    new_X_text = vectorizer.transform(new_food['Food'])
    new_X_numeric = new_food[numeric_features]
    new_X = hstack([new_X_numeric, new_X_text])
    
    prediction = multi_rf.predict(new_X)[0]
    return jsonify({
        "food_name": food_name,
        "Calories": float(prediction[0]),
        "Protein": float(prediction[1]),
        "Carbs": float(prediction[2]),
        "Fat": float(prediction[3])
    })

if __name__ == "__main__":
    app.run(debug=True)
