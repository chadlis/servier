"""Console script for servier."""
import argparse
import sys

from .config import (
    RAW_DATA_PATH,
    PRIMARY_DATA_PATH,
    MODEL_DATA_PATH,
    MODEL_DATA_PATH_NAME,
    REPORTING_DATA_PATH,
    MAX_EPOCHS,
    LEARNING_RATE,
)

from .pipelines.data_splitting import split_data
from .pipelines.training import train
from .pipelines.evaluation import evaluate
from .pipelines.prediction import predict
from .app import host
import string
import random


def main():
    """Console script for servier."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        default="split;train;evaluate;predict;host",
        type=str,
        # choices=["split", "train", "evaluate", "predict",],
        help="split data, train, evaluate or predict (default: %(default)s)",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        help="Input data path",
    )
    parser.add_argument("--valid_path", type=str, help="Validation data path")
    parser.add_argument("--output_path", type=str, help="Output path")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--reporting_path", type=str, help="Reporting path")
    parser.add_argument("--test_only", action="store_true", help="Output path")
    parser.add_argument("--experiment", type=str, help="Experiment Name")
    parser.add_argument("--lr", type=float, help="Learning Rate for training")
    parser.add_argument("--epochs", type=int, help="Epochs training")

    args = parser.parse_args()
    mode = args.mode
    input_path = args.input_path
    valid_path = args.valid_path
    output_path = args.output_path
    model_path = args.model_path
    reporting_path = args.reporting_path
    test_only = args.test_only
    experiment = args.experiment
    train_lr = args.lr
    train_epochs = args.epochs


    print(f"Experiment: {experiment}")
    if "split" in mode:
        if experiment is None:
            experiment = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=7)
            )
        print("\nSplitting data..")
        if input_path is None:
            input_path = RAW_DATA_PATH
        if output_path is None:
            output_path = PRIMARY_DATA_PATH
        split_data(
            experiment,
            input_path,
            output_path,
            test_only=test_only,
        )
        input_path = output_path
        valid_path = output_path
    if "train" in mode:
        print("\nTraining..")
        if experiment is None:
            raise Exception('No experiment was given!')
        if input_path is None:
            input_path = PRIMARY_DATA_PATH
            if valid_path is None:
                valid_path = PRIMARY_DATA_PATH
        if model_path is None:
            model_path = MODEL_DATA_PATH
        if reporting_path is None:
            reporting_path = REPORTING_DATA_PATH
        if train_lr is None:
            train_lr = LEARNING_RATE
        if train_epochs is None:
            train_epochs = MAX_EPOCHS
        train(
            experiment,
            input_path,
            valid_path=valid_path,
            model_path=model_path,
            reporting_path=reporting_path,
            learning_rate=train_lr,
            epochs=train_epochs,
        )
    if "evaluate" in mode:
        print("\nEvaluation..")
        if experiment is None:
            raise Exception('No experiment was given!')
        if input_path is None:
            input_path = PRIMARY_DATA_PATH
        if model_path is None:
            model_path = MODEL_DATA_PATH
        if reporting_path is None:
            reporting_path = REPORTING_DATA_PATH
        evaluate(
            experiment,
            input_path,
            model_path,
            reporting_path,
        )
    if "predict" in mode:
        print("\nPrediction..")
            if experiment is None:
            raise Exception('No experiment was given!')
        if input_path is None:
            input_path = PRIMARY_DATA_PATH
        if model_path is None:
            model_path = MODEL_DATA_PATH
        if reporting_path is None:
            reporting_path = REPORTING_DATA_PATH
        predict(
            experiment,
            model_path,
            reporting_path,
            data_path=input_path,
        )
    if "host" in mode:
        print("\nHosting app..")
        host()


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
