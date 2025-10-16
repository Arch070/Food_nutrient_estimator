Perfect 🌸 Archana! Here’s your **fully upgraded, complete `README.md`** — all in one block so you can just copy and paste straight into your GitHub.

This version includes:

* ✨ Eye-catching title & badges
* 🍽️ Full step-by-step explanation (ML + DL working)
* 🧠 Model training, dataset info, accuracy check
* 🎨 Tech stack visuals
* 💖 Credits & future plans

---

````markdown
# 🍓 Food Nutrition Estimator — AI-Powered Health Companion  

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-Machine%20Learning-orange)
![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-Frontend-purple)
![Chart.js](https://img.shields.io/badge/Chart.js-Visualization-pink)
![Kaggle Dataset](https://img.shields.io/badge/Dataset-Food41-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🌟 Overview  

**Food Nutrition Estimator** is an AI-powered web application that uses **Machine Learning** and **Deep Learning** to predict nutritional information (Calories, Protein, Carbs, Fat) for various foods.  

Users can:
- 📝 **Type a food name** to get instant nutrition values using ML  
- 📸 **Upload a food image** to detect the food and get its nutrients using CNN  

This project beautifully combines **Python, Flask, PyTorch, scikit-learn, and Tailwind CSS** to create a seamless user experience between AI and real-world health awareness.  

---

## 🧠 How It Works  

### 1️⃣ Text-based Prediction (Machine Learning)
- The user enters a food name (like “Pizza” or “Apple”).  
- The input text is processed using a **TF-IDF Vectorizer**.  
- A **Random Forest Regressor** predicts Calories, Protein, Carbs, and Fat.  
- The model outputs approximate nutritional values based on training data.

### 2️⃣ Image-based Prediction (Deep Learning)
- The user uploads an image of the food.  
- A **ResNet18 Convolutional Neural Network** (trained on the [Kaggle Food-41 dataset](https://www.kaggle.com/datasets/kmader/food41)) identifies which food it is.  
- Once detected, the ML model predicts the nutrition data for that detected food.

### 3️⃣ Display & Visualization
- The result (food name + nutrients) is shown along with the **uploaded image**.  
- A **Chart.js pie chart** visualizes the macro nutrient breakdown (Protein, Carbs, Fat).  
- The interface is designed in **pastel shades using Tailwind CSS** 🌷.

---

## 🧩 Machine Learning Model (Text-based)  

| Component | Description |
|------------|-------------|
| **Algorithm** | Random Forest Regressor |
| **Feature Extraction** | TF-IDF Vectorization |
| **Training Data** | Food names and nutritional info |
| **Output** | Calories, Protein, Carbs, Fat per food |
| **File Saved As** | `multi_rf.pkl`, `vectorizer.pkl` |

### 🔍 Training Code
The ML model was trained and saved using:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
````

---

## 🤖 Deep Learning Model (Image-based)

| Component           | Description         |
| ------------------- | ------------------- |
| **Architecture**    | ResNet18 (CNN)      |
| **Framework**       | PyTorch             |
| **Dataset**         | Food-41 from Kaggle |
| **Training Epochs** | 5                   |
| **Loss Function**   | CrossEntropyLoss    |
| **Optimizer**       | Adam                |
| **Model File**      | `food_model.pth`    |

### 🧾 Dataset Setup (KaggleHub)

Dataset imported directly from Kaggle:

```python
import kagglehub
path = kagglehub.dataset_download("kmader/food41")
print("Dataset downloaded to:", path)
```

---

## 📈 Model Training & Accuracy Checking

Train or re-train model using:

```bash
python train_food_classifier.py
```

To check model accuracy:

```bash
python check_accuracy.py
```

📊 The accuracy and loss values will be printed in your **VS Code terminal** (not on the web).

---

## 💻 Tech Stack

| Layer                | Technology                                    |
| -------------------- | --------------------------------------------- |
| **Frontend**         | HTML, CSS, JavaScript, Tailwind CSS, Chart.js |
| **Backend**          | Flask (Python)                                |
| **Machine Learning** | scikit-learn, joblib                          |
| **Deep Learning**    | PyTorch, Torchvision                          |
| **Dataset**          | Kaggle Food-41                                |
| **Visualization**    | Chart.js                                      |
| **Other Tools**      | Vanilla Tilt.js, JSON APIs                    |

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
│   └── static/uploads/
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
├── README.md
└── requirements.txt
```

---

## 🪄 Installation & Setup

1️⃣ **Clone the repository**

```bash
git clone https://github.com/your-username/food-nutrition-estimator.git
cd food-nutrition-estimator/backend
```

2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

3️⃣ **Run Flask app**

```bash
python app.py
```

4️⃣ **Open in browser**

```
http://127.0.0.1:5000/
```

---

## 🌈 Features

✅ Predict nutrition from text or image
✅ Interactive nutrient pie chart
✅ Beautiful pastel UI with Tailwind CSS
✅ Accurate CNN-based food detection
✅ Local model training and evaluation
✅ Easy-to-extend backend

---

## 🚀 Future Enhancements

* 🧩 Add multiple food detection in one image
* 📱 Make app mobile-friendly with camera input
* 📡 Connect with live nutrition API (USDA)
* 💾 Store user meal logs for tracking
* 🤍 Deploy on **Render / Hugging Face Spaces / AWS**

---

## 🤝 Contributing

Contributions are warmly welcomed! 💡
If you find bugs or have new feature ideas:

* Fork this repo
* Create a new branch
* Submit a pull request

---

## 📜 License

This project is released under the **MIT License** — feel free to use, modify, and share with credit.

---

### 💖 Made with Passion, Data, and Machine Learning by Archana 🌸

```

---

Would you like me to make a **GitHub-optimized banner image** (like a header with “🍓 Food Nutrition Estimator — AI-Powered Nutrition App” in pastel colors) that you can add at the top of your README?  
It makes your project look super professional and aesthetic on GitHub 🌷
```
