from pathlib import Path
import logging

logging.getLogger().setLevel(logging.DEBUG)


DATA_PATH = Path("data")
RAW_DATA_PATH = DATA_PATH/"0_raw"
PRIMARY_DATA_PATH = DATA_PATH/"1_primary"
FEATURE_DATA_PATH = DATA_PATH/"2_feature"
MODEL_DATA_PATH = DATA_PATH/"3_model"
REPORTING_DATA_PATH = DATA_PATH/"4_reporting"

INPUT_FILENAME = "data.csv"

MODEL_DATA_PATH_NAME = "model"

DATA_TRAIN_FILENAME = "data_train.csv"
DATA_VALID_FILENAME = "data_valid.csv"
DATA_TEST_FILENAME = "data_test.csv"

TRAINING_SIZE = 0.15
MAX_EPOCHS = 2