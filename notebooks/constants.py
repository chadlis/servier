from pathlib import Path


from pandera import Column, DataFrameSchema, Check, Index



COL_TARGET = "P1"
COL_SMILES = "smiles"
COL_ID = "mol_id"
RANDOM_STATE = 42

DATA_PATH = Path("data")
RAW_DATA_PATH = DATA_PATH/"0_raw"
INTERMEDIATE_DATA_PATH = DATA_PATH/"1_intermediate"
INPUT_FILENAME = "dataset_single.csv"

DATA_TRAIN_FILENAME = "data_train.csv"
DATA_VALID_FILENAME = "data_valid.csv"
DATA_TEST_FILENAME = "data_test.csv"

INPUT_DATA_PATH = RAW_DATA_PATH/INPUT_FILENAME


VALIDATION_SIZE = 0.15
MAX_EPOCHS = 2


DATA_SCHEMA =  DataFrameSchema(
    {
        COL_ID: Column(str,
                        Check(lambda s: s.str.startswith("CID")),
                        required=False,
                        nullable=False,),
        COL_SMILES: Column(str,
                        Check(lambda s: len(s)>=1,),
                        nullable=False,),
        COL_TARGET: Column(int, Check.isin([0, 1]),
                    nullable=False,),
    },
    index=Index(int),
    strict='filter',
    coerce=True,
)

DATA_SCHEMA_PREDICTION =  DataFrameSchema(
    {
        COL_ID: Column(str,
                        Check(lambda s: s.str.startswith("CID")),
                        required=False,
                        nullable=False,),
        COL_SMILES: Column(str,
                        Check(lambda s: len(s)>=1,),
                        nullable=False,),
        COL_TARGET: Column(int, Check.isin([0, 1]),
                     required=False,
                     nullable=False,),
    },
    index=Index(int),
    strict='filter',
    coerce=True,
)