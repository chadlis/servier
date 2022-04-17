from pathlib import Path
import pandas as pd
from ..config import logging
from pathlib import Path


def ingest_data(
    data_path,
    msg_type=None,
):
    if msg_type is not None:
        msg_type = f"| {msg_type} "
    else:
        msg_type = ""
    logging.info(f" Data Ingestion {msg_type}| Start!")
    if type(data_path) == str:
        data_path = Path(data_path)
    df = pd.read_csv(data_path)
    logging.info(f" Data Ingestion {msg_type}| {len(df)} rows ingested")
    logging.info(f" Data Ingestion {msg_type}| Finished!")
    return df
