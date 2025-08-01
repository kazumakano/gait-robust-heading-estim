import os.path as path
from glob import glob
from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import script.model as M
import script.utility as util
from script.data import DataModule


def run(gpu_id: int, param: dict[str, util.Param] | str, split_file: str, ckpt_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    if isinstance(param, str):
        param = util.load_param(param)
    model_cls = M.get_model_cls(param["arch"])

    datamodule = DataModule(param, split_file)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu_id],
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        callbacks=ModelCheckpoint(monitor="validation_loss", save_last=True),
        max_epochs=param["epoch"]
    )

    if ckpt_file is None:
        model = model_cls(param)
        trainer.fit(model, datamodule=datamodule)
        ckpt_file = glob(path.join(trainer.log_dir, "checkpoints/", "epoch=*-step=*.ckpt"))[0]

    trainer.test(model=model_cls.load_from_checkpoint(ckpt_file), datamodule=datamodule)

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split_file", required=True, help="specify split file", metavar="PATH_TO_SPLIT_FILE")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")

    if sys.stdin.isatty():
        parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
        parser.add_argument("-c", "--ckpt_file", help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
        args = parser.parse_args()

        run(args.gpu_id, args.param_file, args.split_file, args.ckpt_file, args.result_dir_name)

    else:
        args = parser.parse_args()
        lines = sys.stdin.readlines()

        run(args.gpu_id, json.loads(lines[1]), args.split_file, lines[3].rstrip(), args.result_dir_name)
