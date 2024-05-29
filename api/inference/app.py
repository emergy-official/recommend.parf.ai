from flask import Flask, request, jsonify  
from werkzeug.middleware.proxy_fix import ProxyFix  
import os
from keras.models import load_model

from helper import inference

# USAGE FLASK_APP=app.py flask run --port=8080)

app = Flask(__name__)

# which is passed to the container by SageMaker (usually /opt/ml/model).
# model = load_model()

# Since the web application runs behind a proxy (nginx), we need to
# add this setting to our app.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    print("Ping received")
    
    return "pong"

@app.route("/invocations", methods=["POST"])  
def invocations():
    data = request.get_json()
    user_id = data.get("userId") if data else None
    predictions = inference(user_id)  # Update predict to handle base64 image data  
    # predictions = inference(content["user_id"])  # Update predict to handle base64 image data  
    
    return jsonify(predictions=predictions)  


