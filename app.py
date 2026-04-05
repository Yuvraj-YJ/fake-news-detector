from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load models
lr_model = pickle.load(open("model.pkl", "rb"))
sgd_model = pickle.load(open("pa_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔧 Same cleaning as training
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

        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])

        # Predictions
        lr_pred = lr_model.predict(vector)[0]
        sgd_pred = sgd_model.predict(vector)[0]

        # Confidence (only LR supports probability)
        lr_prob = lr_model.predict_proba(vector)[0]
        confidence = round(max(lr_prob) * 100, 2)

        print("LR:", lr_pred, "SGD:", sgd_pred, "Prob:", lr_prob)

        # 🔥 Improved hybrid logic
        if lr_pred == sgd_pred:
            final_pred = lr_pred
            model_used = "Both Models Agreed"
        else:
            # choose higher confidence class
            final_pred = 1 if lr_prob[1] > lr_prob[0] else 0
            model_used = "Confidence-Based Selection"

        # Result mapping
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

if __name__ == "__main__":
    app.run(debug=True)