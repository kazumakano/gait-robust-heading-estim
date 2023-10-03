import os.path as path
from glob import glob
from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import script.utility as util
from script.data import DataModule
from script.model import CNN, BiLSTM, DualCNNLSTM, DualCNNXformer


def run(gpu_id: int, model_name: str, param: dict[str, util.Param] | str, split_file: str, ckpt_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    match model_name:
        case "bilstm":
            model_cls = BiLSTM
        case "cnn":
            model_cls = CNN
        case "dualcnnlstm":
            model_cls = DualCNNLSTM
        case "dualcnnxformer":
            model_cls = DualCNNXformer

    if isinstance(param, str):
        param = util.load_param(param)

    datamodule = DataModule(param, split_file)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        callbacks=ModelCheckpoint(monitor="validation_loss", save_last=True),
        devices=[gpu_id],
        max_epochs=param["epoch"],
        accelerator="gpu"
    )

    if ckpt_file is None:
        model = model_cls(param)
        trainer.fit(model, datamodule=datamodule)
        model.load_from_checkpoint(glob(path.join(trainer.log_dir, "checkpoints/", "epoch=*-step=*.ckpt"))[0])
    else:
        model = model_cls.load_from_checkpoint(ckpt_file, param=param)

    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=("bilstm", "cnn", "dualcnnlstm", "dualcnnxformer"), required=True, help="specify model", metavar="MODEL_NAME")
    parser.add_argument("-s", "--split_file", required=True, help="specify split file", metavar="PATH_TO_SPLIT_FILE")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")

    if sys.stdin.isatty():
        parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
        parser.add_argument("-c", "--ckpt_file", help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
        args = parser.parse_args()

        run(args.gpu_id, args.model, args.param_file, args.split_file, args.ckpt_file, args.result_dir_name)

    else:
        args = parser.parse_args()
        lines = sys.stdin.readlines()

        run(args.gpu_id, args.model, json.loads(lines[1]), args.split_file, lines[3].rstrip(), args.result_dir_name)
