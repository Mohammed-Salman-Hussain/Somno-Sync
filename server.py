import os
import json
import joblib
import pandas as pd
from http.server import SimpleHTTPRequestHandler, HTTPServer

# Load the 3 trained models
print("Loading production ML models into memory...")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
rf_cls = joblib.load(os.path.join(MODEL_DIR, "sleep_debt_classifier.joblib"))
rf_lat = joblib.load(os.path.join(MODEL_DIR, "sleep_latency_regressor.joblib"))
gb_fat = joblib.load(os.path.join(MODEL_DIR, "fatigue_regressor.joblib"))
print("All 3 models loaded successfully!")

class MLServerHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "OK")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                payload = json.loads(post_data.decode("utf-8"))

                # Convert to DataFrame
                df_input = pd.DataFrame([payload])

                # 1. Model 1: Classifier
                pred_debt = str(rf_cls.predict(df_input)[0])
                probs = rf_cls.predict_proba(df_input)[0]
                class_probs = {
                    cls_name: round(float(prob) * 100, 1)
                    for cls_name, prob in zip(rf_cls.classes_, probs)
                }

                # 2. Model 2: Latency Regressor
                pred_latency = round(float(rf_lat.predict(df_input)[0]), 1)

                # 3. Model 3: Fatigue Regressor
                pred_fatigue = round(float(gb_fat.predict(df_input)[0]), 1)

                response_data = {
                    "status": "success",
                    "source": "Python Scikit-Learn Production Models",
                    "predictions": {
                        "model_1_sleep_debt_category": pred_debt,
                        "model_1_class_probabilities": class_probs,
                        "model_2_sleep_latency_min": pred_latency,
                        "model_3_next_day_fatigue_score": pred_fatigue
                    }
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                err_resp = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(err_resp).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

def run(port=8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, MLServerHandler)
    print(f"SomnoSync ML Server running at http://localhost:{port}/index.html")
    httpd.serve_forever()

if __name__ == "__main__":
    run(8000)
