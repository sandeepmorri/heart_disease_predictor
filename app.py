from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load saved model
MODEL_PATH = "models_out/best_SF-3_MI_NaiveBayes.pkl"
with open(MODEL_PATH, "rb") as f:
    saved_data = pickle.load(f)

model = saved_data["pipeline"]
selected_features = saved_data["features"]

# Optional: define reasonable ranges for features (example: min-max from dataset)
feature_ranges = {feat: (0, 300) for feat in selected_features}  # adjust based on your data

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    
    if request.method == "POST":
        user_data = {}
        for feature in selected_features:
            try:
                value = float(request.form.get(feature, 0))
            except ValueError:
                value = 0
            # Clamp value to a reasonable range
            min_val, max_val = feature_ranges.get(feature, (None, None))
            if min_val is not None:
                value = max(min_val, min(max_val, value))
            user_data[feature] = value
        
        # Create DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Predict
        prediction = model.predict(user_df)[0]
        probability = model.predict_proba(user_df)[0][1] if hasattr(model, "predict_proba") else None
        
        result = "HEART DISEASE DETECTED 🚨" if prediction == 1 else "NO HEART DISEASE DETECTED ✅"

    return render_template("index.html", features=selected_features, result=result, probability=probability)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)