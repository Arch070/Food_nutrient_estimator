# 🍓 Food Nutrition Estimator — Health Companion  

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-Machine%20Learning-orange)
![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-Frontend-purple)
![Chart.js](https://img.shields.io/badge/Chart.js-Visualization-pink)
![Kaggle Dataset](https://img.shields.io/badge/Dataset-Food41-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-success)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-ResNet18-red)


---

## 🌟 Overview  

**Food Nutrition Estimator** is an  web application that predicts nutritional information — *Calories, Protein, Carbs, and Fat* — using both **Machine Learning (ML)** and **Deep Learning (DL)**.  

Users can:
- 📝 **Type a food name** to get instant nutrition values (ML-based)
- 📸 **Upload a food image** to detect food type & predict nutrition (CNN-based)

This project combines **Python, Flask, PyTorch, scikit-learn, and Tailwind CSS** for a smart, interactive, and aesthetic experience.

---

## 🧠 How It Works  

### 1️⃣ Text-based Prediction (Machine Learning)
- User enters a food name (e.g., “Apple”).
- Text is vectorized using **TF-IDF Vectorizer**.
- A **Random Forest Regressor** predicts Calories, Protein, Carbs, and Fat.
- Results are displayed in real time.

### 2️⃣ Image-based Prediction (Deep Learning)
- User uploads a food image.
- A **ResNet18 CNN** (trained on the [Kaggle Food-41 Dataset](https://www.kaggle.com/datasets/kmader/food41)) identifies the food.
- The ML model then predicts nutrition for that food.

### 3️⃣ Visualization
- Predicted values are shown with a **pie chart (Chart.js)**.
- Smooth pastel UI built with **Tailwind CSS** 🌷.

---

## 🧩 Machine Learning Model (Text-based)  

| Component | Description |
|------------|-------------|
| **Algorithm** | Random Forest Regressor |
| **Feature Extraction** | TF-IDF Vectorizer |
| **Training Data** | Food names & nutritional info |
| **Output** | Calories, Protein, Carbs, Fat |
| **Model File** | `multi_rf.pkl`, `vectorizer.pkl` |


## 🤖 Deep Learning Model (Image-based)

| Component           | Description         |
| ------------------- | ------------------- |
| **Architecture**    | ResNet18 (CNN)      |
| **Framework**       | PyTorch             |
| **Dataset**         | Food-41 from Kaggle |
| **Epochs**          | 5                   |
| **Loss Function**   | CrossEntropyLoss    |
| **Optimizer**       | Adam                |
| **Model File**      | `food_model.pth`    |
| **Accuracy**        |         80%         |

### 🧾 Dataset Import (via KaggleHub)
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/food41")

print("Path to dataset files:", path)
```

---

## 📈 Model Training & Accuracy  

### Train model:
```bash
python train_food_classifier.py
```

### Check accuracy:
```bash
python check_accuracy.py
```

📊 Model accuracy and loss metrics are printed in your terminal during training.

---

## 💻 Tech Stack  

| Layer | Technology |
|--------|-------------|
| **Frontend** | HTML, CSS, JavaScript, Tailwind CSS, Chart.js |
| **Backend** | Flask (Python) |
| **Machine Learning** | scikit-learn, joblib |
| **Deep Learning** | PyTorch, Torchvision |
| **Dataset** | Kaggle Food-41 |
| **Visualization** | Chart.js |
| **Extras** | Vanilla Tilt.js, JSON API |

---

## 🧾 Project Structure  

```
food-nutrition-estimator/
├── backend/
│   ├── app.py
│   ├── train_food_classifier.py
│   ├── check_accuracy.py
│   ├── food_model.pth
│   ├── multi_rf.pkl
│   ├── vectorizer.pkl
├── frontend/
│   ├── index.html
├── README.md
└── requirements.txt
```

---

## 🪄 Installation & Setup  

1️⃣ **Clone the repository**
```bash
git clone https://github.com/Arch070/Food_nutrient_estimator.git
cd food-nutrition-estimator/backend
```

2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Run Flask server**
```bash
python app.py
```

4️⃣ **Open your browser**
```
http://127.0.0.1:5000/
```

---

## 🌈 Features  

✅ Predict nutrition of food from text or image  
✅ Interactive nutrient visualization   
✅ Pastel-themed responsive UI  
✅ CNN-based image detection  
✅ Easy local model training  
✅ Modular backend for API integration  

---

### 💖 Made with Passion, Data, and Machine Learning by Archana 🌸
