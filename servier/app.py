
from flask import Flask, jsonify, request
from .nodes.featurizer import get_mpnn_dataset

import os
from pathlib import Path
from tensorflow import keras

from .config import (
    MODEL_DATA_PATH,
    MODEL_DATA_PATH_NAME,
    PREDICTION_THRESHOLD,
)

from flask_caching import Cache

def host(debug=False):
    app = Flask(__name__)
    cache = Cache(config={
        "CACHE_TYPE":"RedisCache",
        "CACHE_REDIS_HOST": "0.0.0.0",
        "CACHE_REDIS_PORT": 6379,
        })
    cache.init_app(app)

    @app.route('/', methods=["GET"])
    def hello_world():
        return "Hello! Go to /predict", 200

    @cache.cached(timeout = 3600)
    @app.route('/predict', methods=["GET"])
    def app_predict():
        smiles = request.args.get("smiles", "")
        print(f"smiles: {smiles}")

        experiment = request.args.get("experiment", "")
        print(f"experiment: {experiment}")

        if experiment != "":
            model_path = MODEL_DATA_PATH / experiment / MODEL_DATA_PATH_NAME
        else:
            # get the last experiment if none is given
            model_path = max([d for d in Path(MODEL_DATA_PATH).glob('*/') if d.is_dir()], key=os.path.getmtime) / MODEL_DATA_PATH_NAME

        
        if smiles == "":
            result = {
                "message": "No SMILES input was given!",
                "use_example1": "/predict?smiles=NC(=O)NC(Cc1ccccc1)C(=O)O",
                "use_example2": "/predict?smiles=NC(=O)NC(Cc1ccccc1)C(=O)O&experiment=GKH14H",}
            return jsonify(result), 500
        print(f"model: {model_path}")
        dataset = get_mpnn_dataset([smiles])
        model = keras.models.load_model(model_path)
        model_output = round(model.predict(dataset)[0][0], 2)
        prediction = int(model_output > PREDICTION_THRESHOLD)
        result = {
            "model": str(model_path),
            "smiles": smiles,
            "prediction": str(prediction),
            "model_output": str(model_output),}

        return jsonify(result), 200
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)