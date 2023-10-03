import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from . import utility as util


class TopconHorAccDataset(Dataset):
    def __init__(self, files: list[str], freq: int, gf_radius: int, gf_sigma: float, heading_calc_range: int, win_len: int, win_stride: int) -> None:
        heading_calc_range_half = heading_calc_range // 2
        self.win_len, self.win_stride = win_len, win_stride

        self.ts_list, self.hor_acc_list, self.heading_list = [], [], []
        self.map = []
        for i, f in enumerate(files):
            data = np.loadtxt(f, dtype=np.float64, delimiter=",")
            self.ts_list.append(util.get_win_center_ts(data[heading_calc_range_half:-heading_calc_range_half, 0], self.win_len, self.win_stride))
            self.hor_acc_list.append(util.gaussian_filter(util.calc_gcs_hor_acc(data[heading_calc_range_half:-heading_calc_range_half, 1:4], data[heading_calc_range_half:-heading_calc_range_half, 13:17]), freq, gf_radius, gf_sigma).astype(np.float32))
            self.heading_list.append(util.calc_mean_heading(data[:, 17:], heading_calc_range, self.win_len, self.win_stride))

            for j in range(len(self.ts_list[i])):
                self.map.append((i, j))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ts = self.ts_list[self.map[idx][0]][self.map[idx][1]]
        hor_acc = self.hor_acc_list[self.map[idx][0]][self.map[idx][1] * self.win_stride:self.map[idx][1] * self.win_stride + self.win_len]
        heading = self.heading_list[self.map[idx][0]][self.map[idx][1]]

        return torch.DoubleTensor((ts, )), torch.from_numpy(hor_acc), torch.from_numpy(heading)

    def __len__(self) -> int:
        return len(self.map)

class DataModule(pl.LightningDataModule):
    def __init__(self, param: dict[str, int], split_file: str) -> None:
        super().__init__()

        self.dataset = {}
        self.save_hyperparameters(param)
        self.split = util.load_param(split_file)

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.dataset["train"] = TopconHorAccDataset(self.split["train"], 100, 3, 0.011, 10, 400, 10)
                self.dataset["validate"] = TopconHorAccDataset(self.split["validate"], 100, 3, 0.011, 10, 400, 10)
            case "test":
                self.dataset["test"] = TopconHorAccDataset(self.split["test"], 100, 3, 0.011, 10, 400, 1)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"], shuffle=True, num_workers=self.hparams["num_workers"])

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["validate"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])
