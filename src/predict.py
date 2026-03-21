import pickle
import re

print("\n======================================")
print(" 📰 FAKE NEWS DETECTION SYSTEM")
print(" 👨‍💻 Developed by: Yuvraj")
print(" 🤖 Model: TF-IDF + Logistic Regression")
print("======================================\n")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully!\n")
print("System ready for prediction 🚀\n")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

while True:
    user_input = input("Enter news text (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting... 👋")
        break

    if user_input.strip() == "":
        print("⚠ Please enter valid text\n")
        continue

    # Clean + process
    cleaned_text = clean_text(user_input)
    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)
    prob = model.predict_proba(vector)
    confidence = max(prob[0]) * 100

   

    # ✅ OUTPUT (inside loop)
    print("\n------------------------------")
    print(f"📝 Words analyzed: {len(cleaned_text.split())}")

    if prediction[0] == 1:
        print(f"🟢 REAL News ({confidence:.2f}%)")
        print("Reason: Language resembles authentic reporting")
    else:
        print(f"🔴 FAKE News ({confidence:.2f}%)")
        print("Reason: Pattern matches misleading content")

    print("------------------------------\n")