import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("news_dataset.csv")

# Map labels
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Drop nulls
df = df.dropna()

# 🔥 REMOVE BAD DATA (KEY FIX)
df = df[df['text'].str.len() > 50]   # remove short junk rows

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df['content'] = df['text'].apply(clean_text)

# Balance dataset
df_fake = df[df.label == 0]
df_real = df[df.label == 1]

df_fake_down = resample(df_fake,
                       replace=False,
                       n_samples=len(df_real),
                       random_state=42)

df = pd.concat([df_fake_down, df_real])
df = df.sample(frac=1, random_state=42)

# Split
X = df['content']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.8,
    min_df=2,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
lr_model = LogisticRegression(max_iter=1000)
sgd_model = SGDClassifier(loss='hinge', max_iter=1000)

lr_model.fit(X_train_vec, y_train)
sgd_model.fit(X_train_vec, y_train)

# Evaluate
print("LR Accuracy:", accuracy_score(y_test, lr_model.predict(X_test_vec)))
print("SGD Accuracy:", accuracy_score(y_test, sgd_model.predict(X_test_vec)))

# Save
pickle.dump(lr_model, open("model.pkl", "wb"))
pickle.dump(sgd_model, open("pa_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Clean model trained successfully")