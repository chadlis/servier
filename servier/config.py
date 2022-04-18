from pathlib import Path
import logging
from pandera import Column, DataFrameSchema, Check, Index


logging.getLogger().setLevel(logging.INFO)

RANDOM_STATE = 42

COL_TARGET = "P1"
COL_SMILES = "smiles"
COL_ID = "mol_id"

DATA_PATH = Path("./data")
RAW_DATA_PATH = DATA_PATH / "0_raw"
PRIMARY_DATA_PATH = DATA_PATH / "1_primary"
FEATURE_DATA_PATH = DATA_PATH / "2_feature"
MODEL_DATA_PATH = DATA_PATH / "3_model"
REPORTING_DATA_PATH = DATA_PATH / "4_reporting"

INPUT_FILENAME = "data.csv"

MODEL_DATA_PATH_NAME = "model"
TRAINING_HISTORY_FILE_NAME = "training_history.json"
TEST_RESULTS_FILE_NAME = "test_results.json"
PREDICTIONS_FILE_NAME = "predictions.csv"


DATA_TRAIN_FILENAME = "data_train.csv"
DATA_VALID_FILENAME = "data_valid.csv"
DATA_TEST_FILENAME = "data_test.csv"

DATA_TRAIN_PATTERN = "*train.csv"
DATA_VALID_PATTERN = "*valid.csv"
DATA_TEST_PATTERN = "*test.csv"

TRAINING_SIZE = 0.7
MAX_EPOCHS = 1


DATA_SCHEMA = DataFrameSchema(
    {
        COL_ID: Column(
            str,
            Check(lambda s: s.str.startswith("CID")),
            required=False,
            nullable=False,
        ),
        COL_SMILES: Column(
            str,
            Check(
                lambda s: len(s) >= 1,
            ),
            nullable=False,
        ),
        COL_TARGET: Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
        ),
    },
    index=Index(int),
    strict="filter",
    coerce=True,
)

DATA_SCHEMA_PREDICTION = DataFrameSchema(
    {
        COL_ID: Column(
            str,
            Check(lambda s: s.str.startswith("CID")),
            required=False,
            nullable=False,
        ),
        COL_SMILES: Column(
            str,
            Check(
                lambda s: len(s) >= 1,
            ),
            nullable=False,
        ),
        COL_TARGET: Column(
            int,
            Check.isin([0, 1]),
            required=False,
            nullable=False,
        ),
    },
    index=Index(int),
    strict="filter",
    coerce=True,
)
