from ..config import (
    DATA_TRAIN_PATTERN,
    DATA_VALID_PATTERN,
    MODEL_DATA_PATH_NAME,
    TRAINING_HISTORY_FILE_NAME,
    MAX_EPOCHS,
    LEARNING_RATE,
)
from ..nodes.data_ingestion import ingest_data
from ..nodes.data_validation import validate_dataframe
from ..nodes.featurizer import get_mpnn_dataset
from ..nodes.modeling import get_imbalance_params, MPNNModel
from tensorflow import keras
import warnings
import json
from pathlib import Path

def train(experiment, input_path, valid_path, model_path, reporting_path, learning_rate, epochs, handle_imbalance=True,):
    """
    Create and train the model and save the artifacts and the results
    """
    print(f"learning rate: {learning_rate}")
    print(f"epochs: {epochs}")

    if type(input_path) == str:
        input_path = Path(input_path)

    input_path = input_path/experiment
    input_path.mkdir(parents=True, exist_ok=True)

    print(f"input path: {input_path}")
    df_train = ingest_data(
        input_path, files_pattern=DATA_TRAIN_PATTERN, msg_type="Training Data"
    )
    df_train = validate_dataframe(df_train, predict=False, msg_type="Training Data")

    train_dataset, atom_dim, bond_dim = get_mpnn_dataset(df_train, return_dims=True)

    valid_dataset = None
    if valid_path is not None:
        if type(valid_path) == str:
            valid_path = Path(valid_path)
        valid_path = valid_path/experiment
        valid_path.mkdir(parents=True, exist_ok=True)

        df_valid = ingest_data(
            valid_path, files_pattern=DATA_VALID_PATTERN, msg_type="Validation Data"
        )
        df_valid = validate_dataframe(
            df_valid, predict=False, msg_type="Validation Data"
        )
        valid_dataset = get_mpnn_dataset(
            df_valid,
        )

    initial_bias = None
    class_weight = None
    if handle_imbalance == True:
        initial_bias, class_weight = get_imbalance_params(df_train)

    model = MPNNModel(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        output_bias=initial_bias,
    )

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[keras.metrics.AUC(name="AUC")],
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=[reduce_lr, early_stopping],
            class_weight=class_weight,
        )

    model_path = model_path/experiment
    model_path.mkdir(parents=True, exist_ok=True)
    model_path = model_path/MODEL_DATA_PATH_NAME
    model.save(model_path)

    reporting_path = reporting_path/experiment
    reporting_path.mkdir(parents=True, exist_ok=True)

    with open(reporting_path/TRAINING_HISTORY_FILE_NAME, 'w') as outfile:
        json.dump(str(history.history), outfile)


    return model, history

