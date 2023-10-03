import os
import os.path as path
import pickle
from typing import Optional
import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import utils as tune_utils
import script.utility as util
from script.callback import BestValLossReporter, SlackBot
from script.data import DataModule
from script.model import CNN, BiLSTM, DualCNNLSTM, DualCNNXformer

CPU = 64
GPU_PER_TRIAL = 1
MAX_PEND_TRIAL_NUM = 1
VISIBLE_GPU = (0, 1, 2, 3, 4, 5, 6, 7)

def _get_grid_param_space(param_list: dict[str, list[util.Param]]) -> dict[str, dict[str, list[util.Param]]]:
    param_space = {}
    for k, l in param_list.items():
        param_space[k] = tune.grid_search(l)

    return param_space

def _try(model_name: str, param: dict[str, util.Param], split_file: str) -> None:
    torch.set_float32_matmul_precision("high")
    # tune_utils.wait_for_gpu()

    match model_name:
        case "bilstm":
            model_cls = BiLSTM
        case "cnn":
            model_cls = CNN
        case "dualcnnlstm":
            model_cls = DualCNNLSTM
        case "dualcnnxformer":
            model_cls = DualCNNXformer

    trainer = pl.Trainer(
        logger=TensorBoardLogger(path.join(tune.get_trial_dir(), "log/"), name=None, default_hp_metric=False),
        callbacks=[BestValLossReporter(), ModelCheckpoint(monitor="validation_loss", save_last=True)],
        devices=1,
        enable_progress_bar=False,
        max_epochs=param["epoch"],
        accelerator="gpu",
        enable_model_summary=False
    )

    trainer.fit(model_cls(param), datamodule=DataModule(param, split_file))

def tune_params(model_name: str, param_list_file: str, split_file: str, bot_conf_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in VISIBLE_GPU])
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(MAX_PEND_TRIAL_NUM)
    ray.init(num_cpus=CPU)

    param_list = util.load_param(param_list_file)
    result_dir = util.get_result_dir(result_dir_name)

    tuner = tune.Tuner(
        trainable=tune.with_resources(lambda param: _try(model_name, param, split_file), {"gpu": GPU_PER_TRIAL}),
        param_space=_get_grid_param_space(param_list),
        tune_config=tune.TuneConfig(mode="min", metric="best_validation_loss", chdir_to_trial_dir=False),
        run_config=air.RunConfig(name=path.basename(result_dir), local_dir=path.dirname(result_dir), callbacks=None if bot_conf_file is None else [SlackBot(bot_conf_file)])
    )

    results = tuner.fit()

    with open(path.join(result_dir, "tune_results.pkl"), mode="wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=("bilstm", "cnn", "dual", "xformer"), required=True, help="specify model", metavar="MODEL_NAME")
    parser.add_argument("-p", "--param_list_file", required=True, help="specify parameter list file", metavar="PATH_TO_PARAM_LIST_FILE")
    parser.add_argument("-s", "--split_file", required=True, help="specify split file", metavar="PATH_TO_SPLIT_FILE")
    parser.add_argument("-b", "--bot_conf_file", help="enable slack bot", metavar="PATH_TO_BOT_CONF_FILE")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")
    args = parser.parse_args()

    tune_params(args.model, args.param_list_file, args.split_file, args.bot_conf_file, args.result_dir_name)
