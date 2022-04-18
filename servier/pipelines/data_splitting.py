from pathlib import Path
from ..config import (
    COL_TARGET,
    RANDOM_STATE,
    DATA_TRAIN_FILENAME,
    DATA_TEST_FILENAME,
    DATA_VALID_FILENAME,
    TRAINING_SIZE,
)
from ..config import logging
from ..nodes.data_ingestion import ingest_data
from ..nodes.data_validation import validate_dataframe
from sklearn.model_selection import train_test_split


def split_data(
    experiment,
    data_path,
    output_path,
    train_size=TRAINING_SIZE,
    test_only=True,
    col_target=COL_TARGET,
    random_state=RANDOM_STATE,
    filename_train=DATA_TRAIN_FILENAME,
    filename_test=DATA_TEST_FILENAME,
    filename_valid=DATA_VALID_FILENAME,
):
    """
    Split data into train, (validation) and test sets
    """
    if type(output_path) == str:
        output_path = Path(output_path)
    output_path = output_path/experiment
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = ingest_data(data_path, msg_type="Full Data")
    df = validate_dataframe(df, predict=False, msg_type="Full Data")
    

    if test_only:
        logging.info(f" Data Splitting | Train: {train_size}, Test: {1-train_size}")
        df_train, df_test = train_test_split(
            df,
            test_size=(1 - train_size),
            random_state=RANDOM_STATE,
            stratify=df[COL_TARGET],
        )
        df_train.reset_index(drop=True).to_csv(
            output_path / filename_train,
            index=False,
        )
        df_test.reset_index(drop=True).to_csv(
            output_path / filename_test,
            index=False,
        )
        logging.info(f" Data Splitting | Train: {len(df_train)}, Test: {len(df_test)}")
        logging.info(f" Data Splitting | Finished!")
        return str(output_path / filename_train), str(output_path / filename_test)

    logging.info(
        f" Data Splitting | Train: {train_size}, Valid: {round((1-train_size)/2 ,2)}, Test: {round((1-train_size)/2,2)}"
    )
    df_train, df_not_train = train_test_split(
        df,
        test_size=(1 - train_size),
        random_state=RANDOM_STATE,
        stratify=df[COL_TARGET],
    )
    df_valid, df_test = train_test_split(
        df_not_train,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=df_not_train[COL_TARGET],
    )
    logging.info(
        f" Data Splitting | Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}"
    )

    df_train.reset_index(drop=True).to_csv(
        output_path / filename_train,
        index=False,
    )
    df_valid.reset_index(drop=True).to_csv(
        output_path / filename_valid,
        index=False,
    )
    df_test.reset_index(drop=True).to_csv(
        output_path / filename_test,
        index=False,
    )
    logging.info(f" Data Splitting | Finished!")
    return (
        str(output_path / filename_train),
        str(output_path / filename_valid),
        str(output_path / filename_test),
    )
