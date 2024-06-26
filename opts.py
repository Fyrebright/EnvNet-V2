import os
import argparse
import time


def parse():
    parser = argparse.ArgumentParser(description="BC learning for sounds")

    # General settings
    parser.add_argument(
        "--dataset", required=False, default="esc50", choices=["esc05", "esc50"]
    )
    parser.add_argument(
        "--netType", required=False, default="envnetv2", choices=["envnet", "envnetv2"]
    )
    parser.add_argument(
        "--data",
        default="/home/mohaimen/Desktop/EXPERIMENTS/datasets/",
        required=False,
        help="Path to dataset",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=-1,
        help="esc: 1-5, urbansound: 1-10 (-1: run on all splits)",
    )
    parser.add_argument(
        "--save",
        default=f"results/run_{time.time()}",
        help="Directory to save the results",
    )
    parser.add_argument("--testOnly", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)

    # Learning settings (default settings are defined below)
    parser.add_argument("--BC", default=True, action="store_true", help="BC learning")
    parser.add_argument(
        "--strongAugment",
        default=True,
        action="store_true",
        help="Add scale and gain augmentation",
    )
    parser.add_argument("--nEpochs", type=int, default=-1)
    parser.add_argument("--LR", type=float, default=-1, help="Initial learning rate")
    parser.add_argument(
        "--schedule", type=float, nargs="*", default=-1, help="When to divide the LR"
    )
    parser.add_argument(
        "--warmup", type=int, default=-1, help="Number of epochs to warm up"
    )
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--weightDecay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    # Testing settings
    parser.add_argument("--nCrops", type=int, default=10)

    parser.add_argument("--fs", type=int, default=44100)
    parser.add_argument(
        "--noBC", default=False, action="store_true", help="BC learning"
    )

    opt = parser.parse_args()

    if opt.noBC:
        opt.BC = False

    # Dataset details
    if opt.dataset == "esc50":
        opt.nClasses = 50
        opt.nFolds = 5
    elif opt.dataset == "esc05":
        opt.nClasses = 5
        opt.nFolds = 5
    else:  # urbansound8k
        opt.nClasses = 10
        opt.nFolds = 10

    if opt.split == -1:
        opt.splits = range(1, opt.nFolds + 1)
    else:
        opt.splits = [opt.split]

    # Model details
    if opt.netType == "envnet":
        # opt.fs = 16000
        opt.inputLength = 24014
    else:  # envnetv2
        # opt.fs = 44100
        opt.inputLength = 66650

    # Default settings (nEpochs will be doubled if opt.BC)
    default_settings = dict()
    default_settings["esc50"] = {
        "envnet": {"nEpochs": 600, "LR": 0.01, "schedule": [0.5, 0.75], "warmup": 0},
        "envnetv2": {
            "nEpochs": 1000,
            "LR": 0.1,
            "schedule": [0.3, 0.6, 0.9],
            "warmup": 10,
        },
    }
    default_settings["esc05"] = {
        "envnet": {"nEpochs": 600, "LR": 0.01, "schedule": [0.5, 0.75], "warmup": 0},
        "envnetv2": {"nEpochs": 600, "LR": 0.01, "schedule": [0.5, 0.75], "warmup": 0},
    }
    default_settings["urbansound8k"] = {
        "envnet": {"nEpochs": 400, "LR": 0.01, "schedule": [0.5, 0.75], "warmup": 0},
        "envnetv2": {
            "nEpochs": 600,
            "LR": 0.1,
            "schedule": [0.3, 0.6, 0.9],
            "warmup": 10,
        },
    }
    for key in ["nEpochs", "LR", "schedule", "warmup"]:
        if eval("opt.{}".format(key)) == -1:
            setattr(opt, key, default_settings[opt.dataset][opt.netType][key])
            if key == "nEpochs" and opt.BC:
                opt.nEpochs *= 2

    if opt.save != "None" and not os.path.isdir(opt.save):
        os.makedirs(opt.save)

    display_info(opt)

    return opt


def display_info(opt):
    if opt.BC:
        learning = "BC"
    else:
        learning = "standard"

    maxwidth = 15
    line = f"+{'-'*(2*maxwidth+4)}+"

    optstr = (
        f"{line}\n"
        "| Sound classification\n"
        f"{line}\n"
        f"| {'learning':{maxwidth}} : {learning:>{maxwidth}}\n"
    )

    for k, v in vars(opt).items():
        if k == "save":
            continue

        try:
            optstr += f"| {k:{maxwidth}} : {v:>{maxwidth}}\n"
        except TypeError:
            optstr += f"| {k:{maxwidth}} : {str(v):>{maxwidth}}\n"

    optstr += f"{line}\n"

    print(optstr)

    with open(f"{opt.save}/opts", "w") as f:
        f.write(optstr)

    print(f"saving to {opt.save}...\n")
