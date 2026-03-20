import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import re

print("Training model...")

# Load data
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])

# Combine text
data["content"] = data["title"] + " " + data["text"]

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

data["content"] = data["content"].apply(clean_text)

# Features & labels
X = data["content"]
y = data["label"]

# TF-IDF (text → numbers)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

print("Text converted to vectors")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split done")

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)