import copy
import datetime

import torch
import optuna

from utils.logger import logger
from main import parser as main_parser
from exp.exp_informer import Exp_Informer

existing_args = {action.dest for action in main_parser._actions}

if "n_trials" not in existing_args:
    main_parser.add_argument(
        "--n_trials", type=int, default=20, help="Количество итераций Optuna"
    )

for action in main_parser._actions:
    if action.dest in ["model", "data"] and action.required:
        action.required = False


def copy_args(args):
    return copy.deepcopy(args)


def run_optuna(args=None, **kwargs):
    if args is None:
        args = main_parser.parse_args([])
    elif isinstance(args, dict):
        defaults = main_parser.parse_args([])
        for k, v in args.items():
            setattr(defaults, k, v)
        args = defaults

    for k, v in kwargs.items():
        setattr(args, k, v)

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

    def objective(trial):
        current_args = copy_args(args)

        # Search space
        d_model = trial.suggest_categorical("d_model", [128, 256, 512, 768])

        n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])
        e_layers = trial.suggest_int("e_layers", 1, 5)
        d_layers = trial.suggest_int("d_layers", 1, 3)
        d_ff = trial.suggest_categorical("d_ff", [512, 1024, 2048])
        factor = trial.suggest_categorical("factor", [3, 5])
        learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3])

        current_args.d_model = d_model
        current_args.n_heads = n_heads
        current_args.e_layers = e_layers
        current_args.d_layers = d_layers
        current_args.d_ff = d_ff
        current_args.factor = factor
        current_args.learning_rate = learning_rate

        logger.info(f"\nTrial {trial.number} параметры: {trial.params}")

        try:
            exp = Exp_Informer(current_args)
            setting = "optuna_trial_{}".format(trial.number)
            exp.train(setting)

            vali_data, vali_loader = exp._get_data(flag="val")
            criterion = exp._select_criterion()
            mse = exp.vali(vali_data, vali_loader, criterion)

            return mse
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("\n=== Оптимизация через Optuna завершена ===")
    logger.info("Лучшая итерация:")
    trial = study.best_trial
    logger.info(f"  MSE: {trial.value}")
    logger.info("  Параметры: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"optuna_results_{timestamp}.csv"

    df = study.trials_dataframe()
    df.to_csv(results_file, index=False)

    return trial.params, trial.value


if __name__ == "__main__":
    args = main_parser.parse_args()
    run_optuna(args)
