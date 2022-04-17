from ..config import DATA_SCHEMA, DATA_SCHEMA_PREDICTION
from ..config import COL_SMILES, COL_TARGET
from ..config import logging


def validate_dataframe(data, predict=False, check_balance=True, msg_type=""):
    if msg_type is not None:
        msg_type = f"| {msg_type} "
    else:
        msg_type = ""
    logging.info(f" Data Validation {msg_type}| Start!")
    if predict:
        data = DATA_SCHEMA_PREDICTION.validate(data).drop_duplicates(subset=[COL_SMILES])
        logging.info(f" Data Validation {msg_type}| Finished!")
        return data
    
    data = DATA_SCHEMA.validate(data).drop_duplicates(subset=[COL_SMILES, COL_TARGET])
    
    if check_balance:
        balance_proportions = (data[COL_TARGET].value_counts()/len(data)).round(2).to_dict()
        if max(balance_proportions[1]/balance_proportions[0], balance_proportions[0]/balance_proportions[1]) > 2:
            logging.warning(f" Data Validation {msg_type}| Dataset imbalance | {balance_proportions} -> Severe imbalance!")
        else:
            logging.info(f" Data Validation {msg_type}| Dataset imbalance | Proportions: {balance_proportions}")
            
    logging.info(f" Data Validation {msg_type}| Finished!")
    return data
    