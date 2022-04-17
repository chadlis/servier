"""Console script for servier."""
import argparse
import sys

from .config import (
    RAW_DATA_PATH,
    PRIMARY_DATA_PATH,
    MODEL_DATA_PATH,
    REPORTING_DATA_PATH,
)
from .config import (
    INPUT_FILENAME,
    DATA_TRAIN_FILENAME,
    DATA_VALID_FILENAME,
    DATA_TEST_FILENAME,
    MODEL_DATA_PATH_NAME,
)

from .pipelines.data_splitting import split_data


def main():
    """Console script for servier."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "_",
        default="train",
        choices=["split", "train", "evaluate", "predict"],
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

    args = parser.parse_args()
    mode = args._
    input_path = args.input_path
    valid_path = args.valid_path
    output_path = args.output_path
    model_path = args.model_path
    reporting_path = args.reporting_path
    test_only = args.test_only

    if mode == "split":
        if input_path is None:
            input_path = RAW_DATA_PATH / INPUT_FILENAME
        if output_path is None:
            output_path = PRIMARY_DATA_PATH
        split_data(input_path, output_path, test_only=test_only)
    elif mode == "train":
        if input_path is None:
            input_path = PRIMARY_DATA_PATH / DATA_TRAIN_FILENAME
            if valid_path is None:
                valid_path = PRIMARY_DATA_PATH / DATA_VALID_FILENAME
        if model_path is None:
            model_path = MODEL_DATA_PATH / MODEL_DATA_PATH_NAME
        if reporting_path is None:
            reporting_path = REPORTING_DATA_PATH
        train(
            input_path,
            valid_path=valid_path,
            model_path=model_path,
            reporting_path=reporting_path,
        )
    elif mode == "evaluate":
        if input_path is None:
            input_path = PRIMARY_DATA_PATH / DATA_TEST_FILENAME
        if model_path is None:
            model_path = MODEL_DATA_PATH / MODEL_DATA_PATH_NAME
        if reporting_path is None:
            reporting_path = REPORTING_DATA_PATH
        evaluate(input_path, model_path, reporting_path)
    elif mode == "predict":
        if input_path is None:
            input_path = PRIMARY_DATA_PATH / DATA_TEST_FILENAME
        if model_path is None:
            model_path = MODEL_DATA_PATH / MODEL_DATA_PATH_NAME
        if reporting_path is None:
            reporting_path = REPORTING_DATA_PATH
        predict(input_path, model_path, reporting_path)
    return 0


def train(input_path, valid_path, model_path, reporting_path):
    print("Train..")
    print(f"input_path: {input_path}")
    print(f"valid_path: {valid_path}")
    print(f"model_path: {model_path}")
    print(f"reporting_path: {reporting_path}")


def evaluate(input_path, model_path, reporting_path):
    print("Evaluate..")
    print(f"input_path: {input_path}")
    print(f"model_path: {model_path}")
    print(f"reporting_path: {reporting_path}")


def predict(input_path, model_path, reporting_path):
    print("Predict..")
    print(f"input_path: {input_path}")
    print(f"model_path: {model_path}")
    print(f"reporting_path: {reporting_path}")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
