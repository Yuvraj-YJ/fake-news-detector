import pandas as pd
import re

print("Running preprocess script...")

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])

# 👉 Combine text
data["content"] = data["title"] + " " + data["text"]

# 👉 Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# 👉 Apply cleaning
data["content"] = data["content"].apply(clean_text)

print("Dataset shape:", data.shape)
print("\nFirst 5 cleaned rows:\n")
print(data["content"].head())