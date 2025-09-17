from flask import Flask, request, jsonify
import pickle
import re
import pandas as pd

# Load trained model
try:
    with open("rf_model.pkl", "rb") as f:
        clf = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ ERROR: Could not load rf_model.pkl:", e)
    clf = None

app = Flask(__name__)

# Feature extraction
def extract_features(url: str) -> dict:
    url = str(url).strip()
    url_lower = url.lower().strip()
    features = {}
    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["has_ip"] = 1 if re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", url) else 0
    suspicious_words = [
        "login","signin","sign-in","authenticate","account","secure","verification",
        "verify","confirm","confirmation","update","reset","password","credential",
        "bank","banking","paypal","appleid","mastercard","visa","stripe","payment",
        "billing","invoice","transaction","urgent","alert","locked","suspended",
        "otp","token","authorize","claim","refund","unclaimed","verify-email"
    ]
    features["suspicious_keywords"] = sum(1 for kw in suspicious_words if kw in url_lower)
    return features

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is running"})

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    print("---- /predict called ----")
    try:
        data = request.get_json()
        url = data.get("url")
        print("Incoming URL:", url)

        if not url:
            return jsonify({"error": "URL missing"}), 400

        # Extract features
        feats = extract_features(url)
        print("Extracted features:", feats)

        df = pd.DataFrame([feats])

        # If model not loaded
        if clf is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Ensure same feature order as training
        df = df[clf.feature_names_in_]
        print("DataFrame for model:", df.to_dict(orient="records"))

        # Predict
        prediction = clf.predict(df)[0]
        proba = clf.predict_proba(df)[0][prediction]

        print("Prediction done:", prediction, "Confidence:", proba)

        result = {
            "url": url,
            "prediction": "Phishing" if prediction == 1 else "Benign",
            "confidence": round(float(proba), 2)
        }
        return jsonify(result)

    except Exception as e:
        import traceback
        print("❌ ERROR in /predict:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

