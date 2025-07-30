# task3
# Task 3: End-to-End ML Project – Iris Classifier API

## 🚀 Project Summary
A complete ML pipeline using the Iris dataset, deployed as a Flask API.

## 🔧 Stack
- Python, Scikit-learn, Flask
- Pipeline (Scaler + Model)
- Deployment via Flask API

## 📈 Model
- Algorithm: Logistic Regression
- Accuracy: ~97% (on test split)

## 🌐 API Usage
- POST `/predict`
- Input: JSON with features
- Output: Class (0, 1, 2)

## 💻 To Run
```bash
python train_model.py
python app.py
