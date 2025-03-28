import numpy as np
import shap
import pandas as pd
import pickle
import os
from flask import Flask, request, jsonify
from scraper import scrape_instagram

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/fake_profile_detector.pkl")
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Define feature columns
FEATURE_COLUMNS = [
    "profile pic", "nums/length username", "fullname words",
    "nums/length fullname", "name==username", "description length",
    "external URL", "private", "#posts", "#followers", "#follows"
]

@app.route("/predict", methods=["GET"])
def predict():
    try:
        username = request.args.get("username")
        if not username:
            return jsonify({"error": "Username parameter is required"}), 400

        # Scrape profile data
        profile_data = scrape_instagram(username)
        if "error" in profile_data:
            return jsonify({"error": profile_data["error"]}), 400

        # Convert to DataFrame
        df = pd.DataFrame([profile_data])
        df = df[FEATURE_COLUMNS]

        # Get fake probability
        if hasattr(model, "predict_proba"):
            fake_score = model.predict_proba(df)[:, 1][0] * 100
        else:
            fake_score = model.predict(df)[0] * 100

        # SHAP for feature importance
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        # Extract SHAP values for the "fake profile" class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get SHAP values for class 1 (fake profile)

        # Convert 2D array to 1D
        shap_values = shap_values.flatten()  # 🔥 Fixes the error

        # Normalize feature contributions on a scale of 0-100
        abs_shap_values = np.abs(shap_values)
        max_shap = np.max(abs_shap_values)
        normalized_shap_values = (abs_shap_values / max_shap) * 100 if max_shap != 0 else abs_shap_values

        # Format feature contributions correctly
        feature_contributions = {
            feature: round(float(value), 2) for feature, value in zip(FEATURE_COLUMNS, normalized_shap_values)
        }

        return jsonify({
            "username": username,
            "fake_score": round(float(fake_score), 2),
            "feature_contributions": feature_contributions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)






