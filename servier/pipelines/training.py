from ..config import (
    MAX_EPOCHS,
    DATA_TRAIN_PATTERN,
    DATA_VALID_PATTERN,
    MODEL_DATA_PATH_NAME,
    TRAINING_HISTORY_FILE_NAME,
)
from ..nodes.data_ingestion import ingest_data
from ..nodes.data_validation import validate_dataframe
from ..nodes.featurizer import get_mpnn_dataset
from ..nodes.modeling import get_imbalance_params, MPNNModel
import matplotlib.pyplot as plt
from numpy import savetxt
from tensorflow import keras
import warnings
import json
from pathlib import Path

def train(experiment, input_path, valid_path, model_path, reporting_path, handle_imbalance=False,):
    """
    Create and train the model and save the artifacts and the results
    """

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
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
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
            epochs=MAX_EPOCHS,
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

    #with warnings.catch_warnings():
    #    fig = plt.figure(figsize=(10, 6))
    #    plt.plot(history.history["AUC"], label="train AUC")
    #    plt.plot(history.history["val_AUC"], label="valid AUC")
    #    plt.xlabel("Epochs", fontsize=16)
    #    plt.ylabel("AUC", fontsize=16)
    #    plt.legend(fontsize=16)
    #    plt.close(fig)
    #    plt.savefig(reporting_path / "training_history.png")

    return model, history

