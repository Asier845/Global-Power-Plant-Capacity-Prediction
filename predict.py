import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

with open("final_rf_pipeline.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

app = Flask("power-plant-capacity-service")

COLS_TO_DROP = [
    "gppd_idnr", "name", "source", "url",
    "wepp_id", "geolocation_source", "owner", "capacity_mw"
]

EXPECTED_COLS = model_pipeline.named_steps["preprocessor"].feature_names_in_.tolist()


def prepare_features(plant_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([plant_data])
    df = df.drop(columns=COLS_TO_DROP, errors="ignore").copy()

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[EXPECTED_COLS]

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    return df


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    plant_data = request.get_json()
    if not plant_data:
        return jsonify({"error": "No JSON payload provided"}), 400

    try:
        X = prepare_features(plant_data)
        prediction = model_pipeline.predict(X)[0]
        return jsonify({
            "predicted_capacity_mw": round(float(prediction), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=9696)