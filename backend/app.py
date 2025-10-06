import logging
import sys
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure we can import shared modeling utilities
ROOT_DIR = BASE_DIR = Path(__file__).parent
DM_DIR = ROOT_DIR / "data-modeling"
if str(DM_DIR) not in sys.path:
    sys.path.insert(0, str(DM_DIR))

from inference import predict_precip_mm  # type: ignore
from power_data import get_dataset, EXTRA_PARAMS, TARGET_PARAM  # type: ignore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "https://probilisky.vercel.app",
        ],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# Load the model bundle and datasets once at startup
model_bundle = None
ds = None
ds_climo = None
try:
    # Prefer MLP model bundle; fallback to baseline if needed
    pickle = DM_DIR / 'artifacts/mlp_rain_model.pkl'
    
    model_path = pickle if pickle.exists() else None
    if model_path is None:
        logging.error("No model bundle found in artifacts directory. Train and save a bundle first.")
    else:
        model_bundle = joblib.load(model_path)
        logging.info(f"Model bundle loaded from {model_path.name}.")

        # Open datasets via shared data loader (uses local cache if available)
        bbox = model_bundle.get('bbox')
        target_param = model_bundle.get('target_param', TARGET_PARAM)
        bundle = get_dataset(years=5, bbox=bbox, target_param=target_param, extra_params=EXTRA_PARAMS)
        ds = bundle.ds
        ds_climo = bundle.ds_climo
        logging.info("Datasets loaded for inference.")
except Exception as e:
    logging.error(f"Error loading model bundle or datasets: {e}")


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict precipitation metrics for a given lat/lon/date.

    Request JSON:
      { "lat": 43.65, "lon": -79.38, "date": "2025-11-15" }

    Response JSON:
      { "p_rain": float, "amount_if_rain_mm": float, "expected_mm": float }
    """
    if model_bundle is None or ds is None or ds_climo is None:
        return jsonify({"error": "Model or datasets not loaded. Check server logs."}), 500
    try:
        payload = request.get_json(force=True) or {}
        lat = float(payload.get("lat"))
        lon = float(payload.get("lon"))
        date_str = str(payload.get("date"))
        if not date_str:
            raise ValueError("'date' is required in ISO format, e.g., YYYY-MM-DD")

        logging.info(f"Predict request: lat={lat}, lon={lon}, date={date_str}")
        result = predict_precip_mm(model_bundle, ds, ds_climo, lat, lon, date_str)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 400

@app.route('/')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Server is running"})

if __name__ == '__main__':
    # Set debug=False for production environments
    app.run(host='0.0.0.0', port=5000, debug=False)
