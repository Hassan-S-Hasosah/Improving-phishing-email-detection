from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from phishing_detector import PhishingDetector



app = Flask(__name__)
CORS(app)


detector = PhishingDetector()


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        email_text = data.get("email_text", "")
        mode = data.get("mode", "balanced")

        if not email_text.strip():
            return jsonify({"error": "Email text is empty"}), 400

        if mode not in ["balanced", "high_security"]:
            return jsonify({
                "error": "Invalid mode. Use 'balanced' or 'high_security'."
            }), 400

        result = detector.predict_raw_email(
            raw_email=email_text,
            mode=mode
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed. Check server logs."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )