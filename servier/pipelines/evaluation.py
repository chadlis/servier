from ..config import (
    DATA_TEST_PATTERN,
    MODEL_DATA_PATH_NAME,
    TEST_RESULTS_FILE_NAME,
)
from ..nodes.data_ingestion import ingest_data
from ..nodes.data_validation import validate_dataframe
from ..nodes.featurizer import get_mpnn_dataset
from tensorflow import keras
import json
from pathlib import Path

def evaluate(experiment, input_path, model_path, reporting_path,):
    if type(input_path) == str:
        input_path = Path(input_path)
    input_path = input_path/experiment
    print(input_path)
    if type(model_path) == str:
        model_path = Path(model_path)
    model_path = model_path/experiment/MODEL_DATA_PATH_NAME
    print(model_path)
    df_data = ingest_data(
        input_path, files_pattern=DATA_TEST_PATTERN, msg_type="Testing Data"
    )
    df_data = validate_dataframe(df_data, predict=False, msg_type="Testing Data")
    dataset = get_mpnn_dataset(df_data)
    model = keras.models.load_model(model_path)
    loss, auc = model.evaluate(dataset)
    results = {"loss": loss, "auc": auc,}
    reporting_path = reporting_path/experiment
    reporting_path.mkdir(parents=True, exist_ok=True)
    reporting_path = reporting_path/TEST_RESULTS_FILE_NAME
    with open(reporting_path, 'w') as outfile:
        json.dump(results, outfile)
    return loss, auc
