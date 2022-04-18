"""Console script for servier."""
import argparse
import sys

from .config import (
    RAW_DATA_PATH,
    PRIMARY_DATA_PATH,
    MODEL_DATA_PATH,
    REPORTING_DATA_PATH,
    MODEL_DATA_PATH_NAME,
)

from .pipelines.data_splitting import split_data
from .pipelines.training import train
from .pipelines.evaluation import evaluate
from .pipelines.prediction import predict
import string    
import random

from flask import Flask




def main():
    """Console script for servier."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        default="split;train;evaluate;predict",
        type=str,
        #choices=["split", "train", "evaluate", "predict",],
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

    args = parser.parse_args()
    mode = args.mode
    input_path = args.input_path
    valid_path = args.valid_path
    output_path = args.output_path
    model_path = args.model_path
    reporting_path = args.reporting_path
    test_only = args.test_only
    experiment=args.experiment

    if experiment is None:
        experiment = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))

    print(f"Experiment: {experiment}")

    if "split" in mode:
        print("\nSplitting data..")
        if input_path is None:
            input_path = RAW_DATA_PATH
        if output_path is None:
            output_path = PRIMARY_DATA_PATH
        split_data(experiment,
        input_path,
        output_path, test_only=test_only,)
        input_path = output_path
        valid_path = output_path
    if "train" in mode:
        print("\nTraining..")
        if input_path is None:
            input_path = PRIMARY_DATA_PATH
            if valid_path is None:
                valid_path = PRIMARY_DATA_PATH
        if model_path is None:
            model_path = MODEL_DATA_PATH
        if reporting_path is None:
            reporting_path = REPORTING_DATA_PATH
        train(
            experiment,
            input_path,
            valid_path=valid_path,
            model_path=model_path,
            reporting_path=reporting_path,
        )
    if "evaluate" in mode:
        print("\nEvaluation..")
        if input_path is None:
            input_path = PRIMARY_DATA_PATH
        if model_path is None:
            model_path = MODEL_DATA_PATH
        if reporting_path is None:
            reporting_path = REPORTING_DATA_PATH
        evaluate(experiment, input_path, model_path, reporting_path,)
    if "predict" in mode:
        print("\nPrediction..")
        if input_path is None:
            input_path = PRIMARY_DATA_PATH
        if model_path is None:
            model_path = MODEL_DATA_PATH
        if reporting_path is None:
            reporting_path = REPORTING_DATA_PATH
        predict(experiment, model_path, reporting_path, data_path=input_path,)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
