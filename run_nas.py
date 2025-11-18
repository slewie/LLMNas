import os
import json
import datetime

import torch
from dotenv import load_dotenv

from utils.logger import logger
from main import parser as main_parser
from nas.nas_manager import NASManager


load_dotenv()


existing_args = {action.dest for action in main_parser._actions}

if "api_key" not in existing_args:
    main_parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="Ключ API OpenRouter",
    )

if "nas_iter" not in existing_args:
    main_parser.add_argument(
        "--nas_iter", type=int, default=5, help="Количество итераций NAS"
    )

for action in main_parser._actions:
    if action.dest in ["model", "data"] and action.required:
        action.required = False


def run_nas(args=None, **kwargs):
    if args is None:
        args = main_parser.parse_args([])
    elif isinstance(args, dict):
        defaults = main_parser.parse_args([])
        for k, v in args.items():
            setattr(defaults, k, v)
        args = defaults

    for k, v in kwargs.items():
        setattr(args, k, v)

    if not args.api_key:
        logger.error(
            "Ошибка: Ключ API должен быть задан через --api_key или переменную окружения OPENROUTER_API_KEY"
        )
        return None, None

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        "ETTh1": {
            "data": "ETTh1.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTh2": {
            "data": "ETTh2.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info["data"]
        args.target = data_info["T"]
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(" ", "").split(",")]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    nas_manager = NASManager(api_key=args.api_key)

    best_arch, best_mse = nas_manager.run_search(args, iterations=args.nas_iter)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"nas_results_{timestamp}.json"

    results = {
        "best_arch": best_arch,
        "best_mse": best_mse,
        "history": nas_manager.history,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    return best_arch, best_mse


if __name__ == "__main__":
    args = main_parser.parse_args()
    run_nas(args)
