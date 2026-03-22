import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load dataset
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])
data = data.sample(frac=1).reset_index(drop=True)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

data["text"] = data["text"].apply(clean_text)

# Split
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_vec, y_train)

# Passive Aggressive
pa_model = PassiveAggressiveClassifier(max_iter=200)
pa_model.fit(X_train_vec, y_train)

# Predictions
lr_pred = lr_model.predict(X_test_vec)
pa_pred = pa_model.predict(X_test_vec)

# Accuracy
print("Logistic Accuracy:", accuracy_score(y_test, lr_pred))
print("Passive Aggressive Accuracy:", accuracy_score(y_test, pa_pred))

# Save files
pickle.dump(lr_model, open("model.pkl", "wb"))
pickle.dump(pa_model, open("pa_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ All models saved successfully!")