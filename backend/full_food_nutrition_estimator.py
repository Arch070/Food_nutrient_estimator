''' # expanded_food_nutrition_vitamins.py

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# -------------------------------
# 1. Expanded sample dataset with vitamins & minerals
# -------------------------------
data = {
    'Food': [
        "Grilled Chicken", "Boiled Rice", "Scrambled Eggs", "Oatmeal with Milk", "Apple",
        "Banana", "Fried Egg", "Salmon Fillet", "Greek Yogurt", "Almonds",
        "Broccoli", "Carrot", "Cheddar Cheese", "Peanut Butter", "Whole Wheat Bread"
    ],
    'Fat': [5, 0.5, 10, 3, 0.2, 0.3, 11, 13, 0.4, 50, 0.4, 0.1, 33, 50, 4],
    'Protein': [31, 2.5, 12, 6, 0.3, 1.3, 13, 20, 10, 21, 2.8, 0.9, 25, 25, 13],
    'Carbohydrates': [0, 28, 1, 27, 14, 27, 1, 0, 4, 22, 7, 10, 1.3, 20, 40],
    'Caloric Value': [200, 130, 150, 170, 52, 105, 155, 208, 59, 579, 35, 41, 403, 588, 247],
    'Vitamin B12': [0.3, 0, 0.5, 0, 0, 0, 0.5, 4.5, 1.3, 0, 0, 0, 1, 0, 0],
    'Vitamin B6': [0.5, 0.05, 0.2, 0.1, 0.05, 0.4, 0.2, 0.9, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
    'Vitamin C': [0, 0, 0, 0, 4.6, 8.7, 0, 0, 0, 0, 89, 5.9, 0, 0, 0],
    'Vitamin D': [0.01, 0, 0.01, 0.01, 0, 0, 0.01, 9, 0.03, 0, 0, 0, 0.02, 0, 0],
    'Calcium': [15, 10, 50, 120, 6, 5, 50, 12, 110, 269, 47, 33, 721, 43, 107],
    'Iron': [1, 0.3, 1.2, 1.5, 0.1, 0.3, 1.2, 0.5, 0.1, 3.7, 0.7, 0.3, 0.7, 1.9, 3.6],
    'Magnesium': [25, 12, 10, 40, 5, 27, 10, 30, 11, 268, 21, 12, 28, 168, 80],
    'Zinc': [1, 0.2, 1, 0.5, 0.1, 0.2, 1, 0.6, 0.5, 3.1, 0.4, 0.3, 3.1, 3.3, 1.2]
}

df = pd.DataFrame(data)

# -------------------------------
# 2. Prepare features and target
# -------------------------------
numeric_features = ['Fat', 'Protein', 'Carbohydrates', 'Vitamin B12', 'Vitamin B6', 'Vitamin C', 
                    'Vitamin D', 'Calcium', 'Iron', 'Magnesium', 'Zinc']
X_numeric = df[numeric_features]

# TF-IDF for food names
vectorizer = TfidfVectorizer(max_features=50)
X_text = vectorizer.fit_transform(df['Food'])

# Combine numeric + text features
X = hstack([X_numeric, X_text])

# Targets: Calories, Protein, Carbs, Fat
y = df[['Caloric Value', 'Protein', 'Carbohydrates', 'Fat']]

# -------------------------------
# 3. Train model
# -------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
multi_rf = MultiOutputRegressor(rf)
multi_rf.fit(X, y)

# -------------------------------
# 4. Interactive user input (only food name)
# -------------------------------
print("Welcome to the Advanced Food Nutrition Estimator!")
food_name = input("Enter food name: ")

# Use average values for numeric features as placeholders
avg_values = df[numeric_features].mean().to_dict()
new_food = pd.DataFrame({
    'Food': [food_name],
    **{feature: [avg_values[feature]] for feature in numeric_features}
})

new_X_text = vectorizer.transform(new_food['Food'])
new_X_numeric = new_food[numeric_features]
new_X = hstack([new_X_numeric, new_X_text])

# -------------------------------
# 5. Predict nutrition
# -------------------------------
predicted_nutrition = multi_rf.predict(new_X)
print("\nPredicted Nutrition (Calories, Protein, Carbs, Fat):")
print(predicted_nutrition[0])

import pickle

# Save trained model
with open('multi_rf.pkl', 'wb') as f:
    pickle.dump(multi_rf, f)

# Save TF-IDF vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved!")

'''
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from scipy.sparse import hstack
from PIL import Image
import io
import torch
from torchvision import models, transforms

app = Flask(__name__)

# -------------------------------
# Load your existing text-based model
# -------------------------------
with open("multi_rf.pkl", "rb") as f:
    multi_rf = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Sample numeric feature averages from your dataset
numeric_features = ['Fat','Protein','Carbohydrates','Vitamin B12','Vitamin B6','Vitamin C',
                    'Vitamin D','Calcium','Iron','Magnesium','Zinc']
avg_values = {
    'Fat': 10, 'Protein': 10, 'Carbohydrates': 10, 'Vitamin B12': 0.5, 'Vitamin B6': 0.2,
    'Vitamin C': 5, 'Vitamin D': 0.1, 'Calcium': 50, 'Iron': 1, 'Magnesium': 20, 'Zinc': 1
}

# -------------------------------
# Load a pretrained image classifier
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model = models.resnet18(pretrained=True)
image_model.eval()
image_model.to(device)

# Transform for image input
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Example food class mapping (replace with Food101 or custom classes)
food_classes = ["apple", "banana", "broccoli", "carrot", "chicken", "egg", "salmon", "almonds", "yogurt", "bread"]

# -------------------------------
# Text-based endpoint
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict_text():
    food_name = request.form.get("food_name")
    if not food_name:
        return jsonify({"error": "No food name provided"}), 400

    # Prepare numeric + text features
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

# -------------------------------
# Image-based endpoint
# -------------------------------
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
