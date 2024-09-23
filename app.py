from flask import Flask, render_template, request, jsonify
from gpr_forecast import generate_forecast

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/forecast", methods=["POST"])
def forecast():
    # Generate forecast and get plot as base64
    plot_url, predictions = generate_forecast()
    return jsonify({"plot_url": plot_url, "predictions": predictions})


if __name__ == "__main__":
    app.run(debug=True)
