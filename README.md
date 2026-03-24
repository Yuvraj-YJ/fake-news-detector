# 📰 Fake News Detection System  

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-WebApp-black?logo=flask)
![ML](https://img.shields.io/badge/MachineLearning-Hybrid-green)
![Status](https://img.shields.io/badge/Status-Working-success)

---

## 🚀 Overview

This project is an **AI-powered Fake News Detection System** that classifies news as:

- 🟢 **REAL News**
- 🔴 **FAKE News**

using a **Hybrid Machine Learning approach** combined with **Natural Language Processing (NLP)**.

---

## ✨ Features

- 🔍 Detects fake vs real news  
- 🧠 Hybrid ML Model (Logistic Regression + Passive Aggressive)  
- 🎤 Voice input support (Speech Recognition)  
- ⚡ Real-time prediction  
- 📊 Confidence score visualization  
- 🔥 Loading spinner animation  
- 🎨 Premium modern UI (Flask-based)  

---

## 🧠 Hybrid Model Explanation

This project combines two machine learning models for improved robustness:

🔹 Logistic Regression
Probability-based predictions
Used as the primary decision model
Generates confidence scores
🔹 Passive Aggressive Classifier
Fast and efficient for text classification
Works as a secondary validation model
🔹 Final Decision Logic
If both models agree → prediction is accepted
If they differ → Logistic Regression result is prioritized

👉 This hybrid approach improves accuracy, stability, and reliability

---

## 🛠️ Tech Stack

| Category | Technology |
|--------|------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Flask |
| ML | Scikit-learn |
| NLP | TF-IDF |
| Models | Logistic Regression + Passive Aggressive |

---

## 📂 Project Structure

```
fake-news-detector/
│
├── app.py
├── model.pkl
├── pa_model.pkl
├── vectorizer.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repo

```bash
git clone https://github.com/Yuvraj-YJ/fake-news-detector.git
cd fake-news-detector
```

---

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run App

```bash
python app.py
```

---

### 4️⃣ Open Browser

```
http://127.0.0.1:5000/
```

---

⚡ Application Workflow

1.User Input → Text Preprocessing → TF-IDF Vectorization → Hybrid Model → Prediction Output
2.User inputs or speaks news content
3.Text is cleaned and preprocessed
4.Converted into numerical vectors (TF-IDF)
5.Both models generate predictions
6.Hybrid logic determines final output
7.Result + confidence score displayed

---

## 📊 Accuracy

- Logistic Regression: ~92–96%  
- Passive Aggressive: ~93–97%  
- Hybrid Model: ~94–97%  

---

🌐 Deployment Status
⚙️ Configured for Render deployment
🐍 Uses runtime.txt for Python version control
🚀 Live deployment in progress

---

## 🔥 Future Enhancements

- 🌐 Deploy online (Render / Railway)  
- 🤖 Use Deep Learning (BERT / LSTM)  
- 📊 Display accuracy graphs  
- 🧾 Add explainable AI (why fake/real)  
- 🌙 Dark mode toggle  

---

## 👨‍💻 Author

**Yuvraj**

---

## ⭐ Support

If you like this project:

👉 Star ⭐ the repo  
👉 Share with others  
👉 Follow for more projects  

---

## 🧾 License

This project is licensed under the **MIT License**