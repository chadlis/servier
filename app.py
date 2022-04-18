from .servier.config import (
MODEL_DATA_PATH,
MODEL_DATA_PATH_NAME,
)

from flask import Flask
from .servier.pipelines.prediction import predict
from .servier.nodes.featurizer import get_mpnn_dataset

import os
from pathlib import Path
from tensorflow import keras, multiply


experiment = None
if experiment is not None:
    model_path = MODEL_DATA_PATH / experiment / MODEL_DATA_PATH_NAME
else:
    model_path = max([d for d in Path(MODEL_DATA_PATH).glob('*/') if d.is_dir()], key=os.path.getmtime) / MODEL_DATA_PATH_NAME


app = Flask(__name__)



@app.route('/')
def hello_world():
    print(model_path)
    prediction = multiply(2, 3).numpy()
    smiles=['CC1=C(C(=O)Nc2cc(-c3cccc(F)c3)[nH]n2)C2(CCCCC2)OC1=O',]
    dataset = get_mpnn_dataset(smiles)
    print(dataset)

    model = keras.models.load_model(model_path)
    prediction = model.predict(dataset)
    return f"{model_path} : {str(round(prediction[0][0], 2))}"