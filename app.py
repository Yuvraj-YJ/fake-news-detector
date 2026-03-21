from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = 0

    if request.method == "POST":
        text = request.form["news"]

        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)
        prob = model.predict_proba(vector)

        confidence = round(max(prob[0]) * 100, 2)

        if prediction[0] == 1:
            result = "REAL News"
        else:
            result = "FAKE News"

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)