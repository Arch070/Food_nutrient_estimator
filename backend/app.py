from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle
import os

# Image processing
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn


# 1. Hardcoded dataset

data = {
    'Food': ["Grilled Chicken", "Boiled Rice", "Scrambled Eggs", "Oatmeal with Milk", "Apple",
             "Banana", "Fried Egg", "Salmon Fillet", "Greek Yogurt", "Almonds",
             "Broccoli", "Carrot", "Cheddar Cheese", "Peanut Butter", "Whole Wheat Bread"],
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
numeric_features = [
    'Fat', 'Protein', 'Carbohydrates', 'Vitamin B12', 'Vitamin B6',
    'Vitamin C', 'Vitamin D', 'Calcium', 'Iron', 'Magnesium', 'Zinc'
]
avg_values = df[numeric_features].mean().to_dict()


# 2. Trained and loaded text model

if not os.path.exists("multi_rf.pkl") or not os.path.exists("vectorizer.pkl"):
    X_numeric = df[numeric_features]
    vectorizer = TfidfVectorizer(max_features=50)
    X_text = vectorizer.fit_transform(df['Food'])
    X = hstack([X_numeric, X_text])
    y = df[['Caloric Value', 'Protein', 'Carbohydrates', 'Fat']]

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    multi_rf = MultiOutputRegressor(rf)
    multi_rf.fit(X, y)

    with open('multi_rf.pkl', 'wb') as f:
        pickle.dump(multi_rf, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
else:
    with open('multi_rf.pkl', 'rb') as f:
        multi_rf = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)


# 3. Flask app

app = Flask(__name__, template_folder='../frontend')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# Text prediction

@app.route('/predict', methods=['POST'])
def predict_text():
    food_name = request.form['food_name']
    new_food = pd.DataFrame({'Food':[food_name], **{f:[avg_values[f]] for f in numeric_features}})
    new_X = hstack([new_food[numeric_features], vectorizer.transform(new_food['Food'])])
    pred = multi_rf.predict(new_X)[0]
    return jsonify({'Calories':float(pred[0]), 'Protein':float(pred[1]), 'Carbs':float(pred[2]), 'Fat':float(pred[3])})


# Image prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
food_model_path = "food_model.pth"
checkpoint = torch.load(food_model_path, map_location=device)

food_model = models.resnet18(weights=None)
num_features = food_model.fc.in_features
food_model.fc = nn.Linear(num_features, len(checkpoint['classes']))
food_model.load_state_dict(checkpoint['model_state_dict'])
food_model.eval().to(device)

food_classes = checkpoint['classes']

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error":"No image uploaded"}),400

    img_file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
    img_file.save(file_path)

    img = Image.open(file_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = food_model(tensor)
        _, pred_idx = torch.max(outputs, 1)
        food_name = food_classes[pred_idx.item()]

    # Nutrition prediction using text model
    new_food = pd.DataFrame({'Food':[food_name], **{f:[avg_values[f]] for f in numeric_features}})
    new_X = hstack([new_food[numeric_features], vectorizer.transform(new_food['Food'])])
    pred = multi_rf.predict(new_X)[0]

    return jsonify({
        "food_name": food_name,
        "Calories": float(pred[0]),
        "Protein": float(pred[1]),
        "Carbs": float(pred[2]),
        "Fat": float(pred[3]),
        "image_url": f"/uploads/{img_file.filename}"  # send back URL to frontend
    })


# Facts API

@app.route('/facts', methods=['GET'])
def facts():
    facts_data = {k:f"~{df[k].mean():.2f} {unit}" for k,unit in
                  zip(['Vitamin B12','Vitamin B6','Vitamin C','Vitamin D','Calcium','Iron','Magnesium','Zinc'],
                      ['µg/day','mg/day','mg/day','µg/day','mg/day','mg/day','mg/day','mg/day'])}
    facts_data.update({'Calories':'~2000 kcal/day','Protein':'~50 g/day','Carbohydrates':'~300 g/day','Fat':'~70 g/day'})
    return jsonify(facts_data)


# Run server

if __name__ == '__main__':
    app.run(debug=True)
