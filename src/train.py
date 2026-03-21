import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import re
import pickle



print("\n==============================")
print(" FAKE NEWS DETECTION SYSTEM")
print(" Developed by: Yuvraj")
print(" Model: TF-IDF + Logistic Regression")
print("==============================\n")

# Load data
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

print(f"Fake news samples: {len(fake)}")
print(f"Real news samples: {len(real)}")


fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])

# Shuffle dataset to avoid bias (important for training)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine text
data["content"] = data["title"] + " " + data["text"]

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text)         # remove extra spaces
    return text.strip()

data["content"] = data["content"].apply(clean_text)

# Features & labels
X = data["content"]
y = data["label"]

# TF-IDF (text → numbers)
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1,2),
    stop_words='english'
)
X = vectorizer.fit_transform(X)

print("Text converted to vectors")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split done")

# Train model
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Model training completed successfully!")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("------------------------------")

#Model saving

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")