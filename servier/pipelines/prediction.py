from ..config import (
    DATA_TEST_PATTERN,
    MODEL_DATA_PATH_NAME,
    PREDICTIONS_FILE_NAME,
)
from ..nodes.data_ingestion import ingest_data
from ..nodes.data_validation import validate_dataframe
from ..nodes.featurizer import get_mpnn_dataset
from tensorflow import keras
from pathlib import Path

def predict(experiment, model_path, reporting_path="", data_path=None, smiles=None,):
    if type(model_path) == str:
        model_path = Path(model_path)
    model_path = model_path/experiment/MODEL_DATA_PATH_NAME
    print(model_path)
    model = keras.models.load_model(model_path)
    if (data_path is None) and (smiles is None):
        return "Error!"
    if data_path is not None:
        if type(data_path) == str:
            data_path = Path(data_path)
        data_path = data_path/experiment
        df_data = ingest_data(
        data_path, files_pattern=DATA_TEST_PATTERN, msg_type="Testing Data"
        )
        df_data = validate_dataframe(df_data, predict=True, msg_type="Testing Data")
        dataset = get_mpnn_dataset(df_data)
        predictions = model.predict(dataset)
        df_data["prediction"] = predictions
        
        reporting_path = reporting_path/experiment
        reporting_path.mkdir(parents=True, exist_ok=True)
        #savetxt(f'{reporting_path}_predictions.csv', predictions, delimiter=',')
        df_data.to_csv(reporting_path/PREDICTIONS_FILE_NAME)
        return  predictions
    if smiles is not None:
        dataset = get_mpnn_dataset(smiles)
        return model.predict(dataset)