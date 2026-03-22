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

This project uses a **hybrid machine learning approach** by combining two algorithms:

### 🔹 1. Logistic Regression
- Provides probability-based predictions  
- Used as the **primary model**  
- Helps generate confidence scores  

### 🔹 2. Passive Aggressive Classifier
- Fast and efficient for text classification  
- Acts as a **supporting model**  

### 🔹 Final Prediction Logic

- If both models agree → result is accepted  
- If they differ → Logistic Regression prediction is prioritized  

👉 This improves **robustness and reliability** of predictions.

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

## ⚡ How It Works

```
User Input → Text Cleaning → TF-IDF → Hybrid Model → Prediction → Result
```

1. User enters or speaks news  
2. Text is preprocessed  
3. Converted into numerical vectors using TF-IDF  
4. Both models make predictions  
5. Final result is generated using hybrid logic  
6. Confidence score is displayed  

---

## 📊 Accuracy

- Logistic Regression: ~92–96%  
- Passive Aggressive: ~93–97%  
- Hybrid Model: ~94–97%  

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