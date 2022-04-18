
from ..nodes.data_ingestion import ingest_data
from ..nodes.data_validation import validate_dataframe
from .data import get_mpnn_dataset
from .modeling import get_imbalance_params, MPNNModel
import matplotlib.pyplot as plt
from tensorflow import keras
import warnings
from ..config import (
    MAX_EPOCHS,
)

def train(input_path, valid_path, model_path, reporting_path, handle_imbalance=False):
    df_train = ingest_data(input_path, msg_type="Training Data")
    df_train = validate_dataframe(df_train, predict=False, msg_type="Training Data")

    train_dataset, atom_dim, bond_dim = get_mpnn_dataset(df_train, return_dims=True)

    if valid_path is not None:
        df_valid = ingest_data(valid_path, msg_type="Validation Data")
        df_valid = validate_dataframe(df_valid, predict=False, msg_type="Validation Data")
        valid_dataset = get_mpnn_dataset(df_valid,)


    initial_bias=None
    class_weight=None
    if handle_imbalance==True:
        initial_bias, class_weight = get_imbalance_params(df_train)
    
    model = MPNNModel(
        atom_dim=atom_dim, bond_dim=bond_dim, output_bias=initial_bias,
    )
    
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        metrics=[keras.metrics.AUC(name="AUC")],
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=5, min_lr=1e-7)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
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
    model.save(model_path)

    with warnings.catch_warnings():
        fig = plt.figure(figsize=(10, 6))
        plt.plot(history.history["AUC"], label="train AUC")
        plt.plot(history.history["val_AUC"], label="valid AUC")
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("AUC", fontsize=16)
        plt.legend(fontsize=16)
        plt.close(fig)
        plt.savefig(reporting_path/"training_history.png")

    return model, history