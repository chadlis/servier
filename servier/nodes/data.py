from pathlib import Path
import tensorflow as tf
from graphizer import graphs_from_smiles, molecule_from_smiles, graph_from_molecule

from ..config import COL_ID, COL_SMILES, COL_TARGET, DATA_SCHEMA, RANDOM_STATE, DATA_TRAIN_FILENAME, DATA_VALID_FILENAME, DATA_TEST_FILENAME
from ..config import DATA_SCHEMA, DATA_SCHEMA_PREDICTION


from featurizer import atom_featurizer, bond_featurizer

import pandas as pd

from sklearn.model_selection import train_test_split



import logging
logging.getLogger().setLevel(logging.DEBUG)

def validate_dataframe(data, predict=False, check_balance=True):
    if predict:
        data = DATA_SCHEMA_PREDICTION.validate(data).drop_duplicates(subset=[COL_SMILES])
        logging.info(f" Data Validation | Finished!")
        return data
    
    data = DATA_SCHEMA.validate(data).drop_duplicates(subset=[COL_SMILES, COL_TARGET])
    
    if check_balance:
        balance_proportions = (data[COL_TARGET].value_counts()/len(data)).round(2).to_dict()
        logging.info(f" Data Validation | Dataset imbalance | Proportions: {balance_proportions}")
        if max(balance_proportions[1]/balance_proportions[0], balance_proportions[0]/balance_proportions[1]) > 2:
            logging.warning(f" Data Validation | Dataset imbalance | Severe imbalance!")
            
    logging.info(f" Data Validation | Finished!")
    return data
    
    
def split_data(data_path, output_path, train_size=0.7, test_only=True, col_target=COL_TARGET, random_state=RANDOM_STATE,
              filename_train=DATA_TRAIN_FILENAME, filename_test=DATA_TEST_FILENAME, filename_valid=DATA_VALID_FILENAME):
    data_path = Path(data_path)
    output_path = Path(output_path)
    df = pd.read_csv(data_path).reset_index(drop=True)
    df = validate_dataframe(df)
    
    if test_only:
        logging.info(f" Data Splitting | Train: {train_size}, Test: {1-train_size}")
        df_train, df_test = train_test_split(df, test_size=(1-train_size), random_state=RANDOM_STATE, stratify=df[COL_TARGET])
        df_train.reset_index(drop=True).to_csv(output_path/filename_train, index=False,)
        df_test.reset_index(drop=True).to_csv(output_path/filename_test, index=False,)
        logging.info(f" Data Splitting | Finished!")
        return str(output_path/filename_train), str(output_path/filename_test)
    
    logging.info(f" Data Splitting | Train: {train_size}, Valid: {round((1-train_size)/2 ,2)}, Test: {round((1-train_size)/2,2)}")
    df_train, df_not_train = train_test_split(df, test_size=(1-train_size), random_state=RANDOM_STATE, stratify=df[COL_TARGET])
    df_valid, df_test = train_test_split(df_not_train, test_size=0.5, random_state=RANDOM_STATE, stratify=df_not_train[COL_TARGET])
    df_train.reset_index(drop=True).to_csv(output_path/filename_train, index=False,)
    df_valid.reset_index(drop=True).to_csv(output_path/filename_valid, index=False,)
    df_test.reset_index(drop=True).to_csv(output_path/filename_test, index=False,)
    logging.info(f" Data Splitting | Finished!")
    return str(output_path/filename_train), str(output_path/filename_valid), str(output_path/filename_test)



def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)

def get_mpnn_dataset(data, atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer,
                     col_smiles=COL_SMILES, col_target=COL_TARGET, return_dims=False):
    
    if type(data) == pd.core.frame.DataFrame:
        x = graphs_from_smiles(data[col_smiles], atom_featurizer, bond_featurizer)
        y = data[col_target]
    else:
        x = graphs_from_smiles(data, atom_featurizer, bond_featurizer)
        y = None
    
    if return_dims:
        return MPNNDataset(x, y), x[0][0][0].shape[0], x[1][0][0].shape[0]
    return MPNNDataset(x, y)