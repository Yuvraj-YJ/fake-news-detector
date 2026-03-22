from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load models
with open("model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("pa_model.pkl", "rb") as f:
    pa_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    model_used = None

    if request.method == "POST":
        text = request.form["news"]

        # Clean + vectorize
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])

        # Predictions
        lr_pred = lr_model.predict(vector)[0]
        pa_pred = pa_model.predict(vector)[0]

        lr_prob = lr_model.predict_proba(vector)[0]
        confidence = round(max(lr_prob) * 100, 2)

        # Hybrid logic
        if lr_pred == pa_pred:
            final_pred = lr_pred
            model_used = "Both Models Agreed"
        else:
            final_pred = lr_pred
            model_used = "Logistic Model Selected"

        # Final result
        if final_pred == 1:
            result = "REAL News"
        else:
            result = "FAKE News"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        model_used=model_used
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)