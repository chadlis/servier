
from flask import Flask, jsonify, request
from .nodes.featurizer import get_mpnn_dataset

import os
from pathlib import Path
from tensorflow import keras

from .config import (
    MODEL_DATA_PATH,
    MODEL_DATA_PATH_NAME,
)

def host(debug=False):
    app = Flask(__name__)

    @app.route('/', methods=["GET"])
    def hello_world():
        return "Hello World!", 200

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
            result = {"message": "No SMILES input was given!"}
            return jsonify(result), 500
        print(f"model: {model_path}")
        dataset = get_mpnn_dataset([smiles])
        model = keras.models.load_model(model_path)
        model_output = round(model.predict(dataset)[0][0], 2)
        prediction = int(model_output > 0.5)
        result = {
            "model": str(model_path),
            "smiles": smiles,
            "prediction": str(prediction),
            "model_output": str(model_output),}

        return jsonify(result), 200

    app.run(debug=debug)