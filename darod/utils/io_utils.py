import argparse
import json
import os
from datetime import datetime


def get_log_path(model_type, backbone="vgg16", custom_postfix=""):
    """
    Generating log path from model_type value for tensorboard.
    :param model_type:  "rpn", "faster_rcnn"
    :param backbone: backbone used
    :param custom_postfix: any custom string for log folder name
    :return: tensorboard log path, for example: "logs/rpn_mobilenet_v2/{date}"
    """
    return "logs/{}_{}{}/{}".format(model_type, backbone, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))


def get_model_path(model_type, backbone="vgg16"):
    """
    Generating model path from model_type value for save/load model weights.
    :param model_type: "rpn", "faster_rcnn"
    :param backbone: backbone used
    :return: os model path, for example: "trained/rpn_vgg16_model_weights.h5"
    """
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "{}_{}_model_weights.h5".format(model_type, backbone))
    return model_path


def handle_args():
    """
    Parse arguments from command line
    :return: args
    """
    parser = argparse.ArgumentParser(description="DAROD implementation")
    parser.add_argument("--config", help="Path of the config file")
    parser.add_argument("--backup-dir", help="Path to backup dir")
    parser.add_argument("--sequence-length", default=None, type=int,
                        help="The length of the sequence for temporal information")
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--use-bn", action="store_true", default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--rpn-pth", default=None, type=str)
    parser.add_argument("--frcnn-pth", default=None, type=str)
    parser.add_argument("--use-aug", action="store_true", default=None)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--use-doppler", action="store_true", default=None)
    parser.add_argument("--use-dropout", action="store_true", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--use-scheduler", action="store_true", default=None)
    parser.add_argument("--exp")
    parser.add_argument("--init", type=str)
    parser.add_argument("--pt", action="store_true", default=None)
    args = parser.parse_args()
    return args


def args2config(args):
    """
    Parse arguments to a new config file
    :param args: arguments
    :return: updated configuration dictionary
    """
    with open(args.config) as file:
        config = json.load(file)

    config["log"]["exp"] = args.exp

    # Model hyper parameters
    config["model"]["backbone"] = args.backbone if args.backbone is not None else config["model"]["backbone"]
    config["model"]["layout"] = args.layout if args.layout is not None else config["model"]["layout"]
    config["model"]["sequence_len"] = args.sequence_length if args.sequence_length is not None else config["model"][
        "sequence_len"]

    # Training hyper parameters
    config["training"]["batch_size"] = args.batch_size if args.batch_size is not None else config["training"][
        "batch_size"]
    config["training"]["epochs"] = args.n_epochs if args.n_epochs is not None else config["training"]["epochs"]
    config["training"]["use_bn"] = args.use_bn if args.use_bn is not None else config["training"]["use_bn"]
    config["training"]["use_aug"] = args.use_aug if args.use_aug is not None else config["training"]["use_aug"]
    config["training"]["use_doppler"] = args.use_doppler if args.use_doppler is not None else config["training"][
        "use_doppler"]
    config["training"]["use_dropout"] = args.use_dropout if args.use_dropout is not None else config["training"][
        "use_dropout"]
    config["training"]["optimizer"] = args.optimizer if args.optimizer is not None else config["training"]["optimizer"]
    config["training"]["lr"] = args.lr if args.lr is not None else config["training"]["lr"]
    config["training"]["scheduler"] = args.use_scheduler if args.use_scheduler is not None else config["training"][
        "scheduler"]
    config["training"]["pretraining"] = "imagenet" if args.pt is not None else "None"

    # Dataset hyper parameters
    config["data"]["dataset"] = args.dataset.split(":")[0] if args.dataset is not None else config["data"]["dataset"]
    config["data"]["dataset_version"] = args.dataset.split(":")[1] if args.dataset is not None else config["data"][
        "dataset_version"]
    return config


def handle_args_eval():
    """
    Parse command line arguments for evaluation script
    :return: args
    """
    parser = argparse.ArgumentParser(description="R2D2 evaluation and inference")
    parser.add_argument("--path", help="Path to the logs")
    parser.add_argument("--show-res", action="store_true",
                        help="Print predictions on the range-Doppler spectrum and upload it in Tensorboard")
    parser.add_argument("--iou-th", action="append", type=float,
                        help="Store a list of IoU threshold to evaluate the model")
    parser.add_argument("--eval-best", action="store_true",
                        help="Eval both the model on the model saved at the best val loss and on the last ckpt.")
    args = parser.parse_args()
    return args


def handle_args_viz():
    parser = argparse.ArgumentParser(description="R2D2 evaluation and inference")
    parser.add_argument("--path", help="Path to the logs")
    args = parser.parse_args()
    return args
