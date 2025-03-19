import argparse
import os
from scripts.train_hyperparameters_search import main_raytune_search_train
from scripts.parallel_prediction_ensamble import parallel_prediction_ensamble
from utils.configurations import configs
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Train model with hyperparameter search or run parallel prediction ensemble.")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Choose between training or prediction mode.")

    # -------------------- TRAIN SUBCOMMAND --------------------
    train_parser = subparsers.add_parser("train", help="Train model with hyperparameter search.")

    train_parser.add_argument(
        "--data_train_val_test",
        type=str,
        required=True,
        help="Path to the CSV file containing training, validation, and test data.",
    )
    train_parser.add_argument(
        "--target_name",
        type=str,
        default=f"trypanodeepscreen_experiment_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Name of the experiment to be used for logging and identification.",
    )
    train_parser.add_argument(
        "--experiment_result_path",
        type=str,
        default="/root/trypanodeepscreen/trained_models",
        help="Path to the directory where trained model folder will be saved.",
    )

    # -------------------- PREDICT SUBCOMMAND --------------------
    predict_parser = subparsers.add_parser("predict", help="Run parallel prediction ensemble.")

    predict_parser.add_argument(
        "--model_folder_path",
        type=str,
        required=True,
        help="Path to the directory containing trained model checkpoints.",
    )
    predict_parser.add_argument(
        "--data_input_prediction",
        type=str,
        required=True,
        help="Path to the input data for prediction.",
    )
    predict_parser.add_argument(
        "--metric_ensambled_prediction",
        type=str,
        default="val_auroc",
        help="Metric to evaluate to ensemble top n models for predictions (e.g., val_auroc, val_mcc).",
    )
    predict_parser.add_argument(
        "--result_path_prediction_csv",
        type=str,
        default=f"/root/trypanodeepscreen/predictions/prediction_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        help="Result path of csv prediction",
    )
    predict_parser.add_argument(
        "--n_checkpoints",
        type=int,
        default=26,
        help="Number of top n models to use for each ensemble model (default: 26).",
    )

    args = parser.parse_args()

    if args.mode == "train":
        main_raytune_search_train(
            data_train_val_test=args.data_train_val_test,
            target_name_experiment=args.target_name,
            data_split_mode="non_random_split",
            experiment_result_path=os.path.abspath(args.experiment_result_path),
        )

    elif args.mode == "predict":
        parallel_prediction_ensamble(
            model_experiments_path=os.path.abspath(args.model_folder_path),
            data_input=os.path.abspath(args.data_input_prediction),
            metric=args.metric_ensambled_prediction,
            result_path=os.path.abspath(args.result_path_prediction_csv),
            n_checkpoints=args.n_checkpoints,
        )

if __name__ == "__main__":
    main()
