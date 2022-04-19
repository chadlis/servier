
from flask import Flask, jsonify, request
from servier.pipelines.prediction import predict
from servier.nodes.featurizer import get_mpnn_dataset
from servier.config import (
MODEL_DATA_PATH,
MODEL_DATA_PATH_NAME,
)

import os
from pathlib import Path
from tensorflow import keras, multiply



app = Flask(__name__)

#Example: 'CC1=C(C(=O)Nc2cc(-c3cccc(F)c3)[nH]n2)C2(CCCCC2)OC1=O'
    
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

if __name__ == "__main__":
    app.run(debug=True) # TODO set debug to False before deployment!