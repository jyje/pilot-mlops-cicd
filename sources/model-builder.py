import argparse, logging, time
from services import load_data, train_model, save_model

# Logging presets
logging.Formatter.converter = time.gmtime
logging.addLevelName(logging.DEBUG, "\033[94mDEBUG\033[0m")
logging.addLevelName(logging.INFO, "\033[92mINFO\033[0m")
logging.addLevelName(logging.WARNING, "\033[93mWARNING\033[0m")
logging.addLevelName(logging.ERROR, "\033[91mERROR\033[0m")
logging.addLevelName(logging.CRITICAL, "\033[95mCRITICAL\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    # dataset-root-path
    parser.add_argument(
        "--dataset-root-path",
        type = str,
        required = True,
        help = "The root path of the dataset"
    )

    # model-root-path
    parser.add_argument(
        "--model-root-path",
        type = str,
        required = True,
        help = "The root path of the model"
    )

    # model-name
    parser.add_argument(
        "--model-name",
        type = str,
        required = True,
        help = "The name of the model"
    )

    # model-version
    parser.add_argument(
        "--model-version",
        type = str,
        required = True,
        help = "The version of the model"
    )

    # learning-rate
    parser.add_argument(
        "--learning-rate",
        type = float,
        required = False,
        default = 0.001,
        help = "The learning rate of the model"
    )

    # momentum
    parser.add_argument(
        "--momentum",
        type = float,
        required = False,
        default = 0.9,
        help = "The momentum of the model"
    )

    # num-epochs
    parser.add_argument(
        "--num-epochs",
        type = int,
        required = False,
        default = 3,
        help = "The number of epochs of the model"
    )

    # batch-size
    parser.add_argument(
        "--batch-size",
        type = int,
        required = False,
        default = 100,
        help = "The batch size of the model training"
    )

    # t-max
    parser.add_argument(
        "--t-max",
        type = int,
        required = False,
        default = 200,
        help = "Maximum iteration for cosine annealing"
    )

    # num-workers
    parser.add_argument(
        "--num-workers",
        type = int,
        required = False,
        default = 0,
        help = "The number of workers for the model training"
    )

    # input-layer-name
    parser.add_argument(
        "--input-layer-name",
        type = str,
        required = False,
        default = "x.1",
        help = "The name of the input layer"
    )

    # input-layer-data-type
    parser.add_argument(
        "--input-layer-data-type",
        type = str,
        required = False,
        default = "TYPE_FP32",
        help = "The data type of the input layer"
    )

    # output-layer-name
    parser.add_argument(
        "--output-layer-name",
        type = str,
        required = False,
        default = "4",
        help = "The name of the output layer"
    )

    # output-layer-data-type
    parser.add_argument(
        "--output-layer-data-type",
        type = str,
        required = False,
        default = "TYPE_FP32",
        help = "The data type of the output layer"
    )

    # download-dataset
    parser.add_argument(
        "--skip-download-dataset",
        action = "store_true",
        help = "Whether to skip downloading the dataset"
    )

    # train-fc-layer-only
    parser.add_argument(
        "--train-fc-layer-only",
        action = "store_true",
        help = "Whether to train only the fully connected layer"
    )

    # log-level
    parser.add_argument(
        "--log-level",
        type = str,
        required = False,
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help = "The log level of the model training"
    )

    # dry-run
    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help = "Whether to dry run the model training"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level = args.log_level.upper(),
        format = "%(levelname)s [%(asctime)sZ] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # Display Arguments
    logging.debug(f"Arguments: {dict(args._get_kwargs())}")

    ## Pipeline Start
    logging.info("model-builder.py")
    logging.info("Pipeline Start")

    # 1. Load Data
    logging.info(" === Pipeline 1/3. Load Data === ")
    dataset, dataloader = load_data(
        dataset_root_path = args.dataset_root_path,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        skip_download_dataset = args.skip_download_dataset,
    )

    # 2. Train Model
    logging.info(" === Pipeline 2/3. Train Model === ")
    model = train_model(
        dataset = dataset,
        dataloader = dataloader,
        learning_rate = args.learning_rate,
        momentum = args.momentum,
        t_max = args.t_max,
        num_epochs = args.num_epochs,
        batch_size = args.batch_size,
        train_fc_layer_only = args.train_fc_layer_only,
        dry_run = args.dry_run,
    )

    # 3. Save Model
    logging.info(" === Pipeline 3/3. Save Model === ")
    save_model(
        class_count = len(dataset.classes),
        model = model,
        model_root_path = args.model_root_path,
        model_name = args.model_name,
        model_version = args.model_version,
        input_layer_name = args.input_layer_name,
        input_layer_data_type = args.input_layer_data_type,
        output_layer_name = args.output_layer_name,
        output_layer_data_type = args.output_layer_data_type,
    )

    # Pipeline End
    logging.info("Pipeline End")

# COMMANDS
# python model-builder.py --help
#
# python model-builder.py \
#   --dataset-root-path data \
#   --model-root-path models \
#   --model-name "pilot" \
#   --model-version "1" \
#   --learning-rate 0.001 \
#   --momentum 0.9 \
#   --batch-size 100 \
#   --t-max 200 \
#   --num-workers 0 \
#   --num-epochs 3 \
#   --train-fc-layer-only \
#   --log-level DEBUG \
#   --dry-run

