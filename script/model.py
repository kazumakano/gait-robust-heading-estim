import os.path as path
import pickle
from typing import Literal
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from . import utility as util


class _BaseModule(pl.LightningModule):
    def __init__(self, param: dict[str, util.Param]) -> None:
        super().__init__()

        self.criterion = nn.MSELoss()
        self.save_hyperparameters(param)

    def configure_optimizers(self) -> optim.Adam:
        return optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        estim = self(batch[1])
        loss = self.criterion(estim, batch[2])
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: list[torch.Tensor], _: int) -> None:
        estim = self(batch[1])
        loss = self.criterion(estim, batch[2])
        self.log("validation_loss", loss)

    def on_test_start(self) -> None:
        self.test_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def test_step(self, batch: list[torch.Tensor], _: int) -> None:
        estim = self(batch[1])
        self.test_outputs.append((batch[0], estim, batch[2]))

    def on_test_end(self) -> None:
        ts = np.empty(0, dtype=np.float64)
        estim = np.empty((0, 2), dtype=np.float32)
        truth = np.empty((0, 2), dtype=np.float32)
        for o in self.test_outputs:
            ts = np.hstack((ts, o[0].squeeze().cpu().numpy()))
            estim = np.vstack((estim, o[1].cpu().numpy()))
            truth = np.vstack((truth, o[2].cpu().numpy()))

        last_split_idx = 0
        self.test_outputs = []
        for i in range(len(ts) - 1):
            if abs(ts[i + 1] - ts[i]) > 0.015:
                self.test_outputs.append((ts[last_split_idx:i + 1], estim[last_split_idx:i + 1], truth[last_split_idx:i + 1]))
                last_split_idx = i + 1
        self.test_outputs.append((ts[last_split_idx:], estim[last_split_idx:], truth[last_split_idx:]))

        with open(path.join(self.logger.log_dir, "test_outputs.pkl"), mode="wb") as f:
            pickle.dump(self.test_outputs, f)

class TimeNorm(nn.Module):
    def __init__(self, enable_on_training: bool) -> None:
        super().__init__()
        self.enable_on_training = enable_on_training

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and not self.enable_on_training:
            return input
        else:
            return input / input.norm(dim=1, keepdim=True)

class BiLSTM(_BaseModule):
    def __init__(self, param: dict[str, util.Param]) -> None:
        super().__init__(param)

        self.lstm = nn.LSTM(2, param["lstm_hs"], num_layers=param["lstm_n_layers"], batch_first=True, dropout=param["lstm_dp"], bidirectional=True)
        self.fc = nn.Linear(2 * param["lstm_hs"], 2)
        self.tn = TimeNorm(False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, time, channel) -> (batch, channel)
        hidden = self.lstm(input)[0]
        hidden = torch.cat((hidden[:, -1, :self.hparams["lstm_hs"]], hidden[:, 0, self.hparams["lstm_hs"]:]), dim=1)    # (batch, time, channel) -> (batch, channel)
        hidden = F.relu(hidden)
        output = self.tn(self.fc(hidden))

        return output

class CNN(_BaseModule):
    def __init__(self, param: dict[str, util.Param]) -> None:
        super().__init__(param)

        self.conv_1 = nn.Conv1d(2, param["conv_ch_1"], param["conv_ks_1"])
        self.bn_1 = nn.BatchNorm1d(param["conv_ch_1"])
        self.conv_2 = nn.Conv1d(param["conv_ch_1"], param["conv_ch_2"], param["conv_ks_2"])
        self.bn_2 = nn.BatchNorm1d(param["conv_ch_2"])
        self.conv_3 = nn.Conv1d(param["conv_ch_2"], param["conv_ch_3"], param["conv_ks_3"])
        self.bn_3 = nn.BatchNorm1d(param["conv_ch_3"])

        self.fc_1 = nn.Linear((403 - param["conv_ks_1"] - param["conv_ks_2"] - param["conv_ks_3"]) * param["conv_ch_3"], param["fc_ch"])
        self.fc_2 = nn.Linear(param["fc_ch"], 2)
        self.tn = TimeNorm(True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, time, channel) -> (batch, channel)
        hidden = F.relu(self.bn_1(self.conv_1(input.transpose(1, 2))))
        hidden = F.relu(self.bn_2(self.conv_2(hidden)))
        hidden = F.relu(self.bn_3(self.conv_3(hidden)))
        hidden = hidden.flatten(start_dim=1)    # (batch, channel, time) -> (batch, channel)
        hidden = F.relu(self.fc_1(hidden))
        output = self.tn(self.fc_2(hidden))

        return output

class DualCNNLSTM(_BaseModule):
    def __init__(self, param: dict[str, util.Param]) -> None:
        super().__init__(param)

        self.conv_1 = nn.Conv1d(2, param["conv_ch_s"], param["conv_ks_s"])
        self.dropout_1 = nn.Dropout(p=param["conv_dp"])
        self.conv_2_l = nn.Conv1d(param["conv_ch_s"], param["conv_ch_l"], 2 * param["conv_ks_s"] - 1)
        self.dropout_2_l = nn.Dropout(p=param["conv_dp"])
        self.conv_3_l = nn.Conv1d(param["conv_ch_l"], param["conv_ch_l"], 2 * param["conv_ks_s"] - 1)
        self.dropout_3_l = nn.Dropout(p=param["conv_dp"])
        self.conv_2_s = nn.Conv1d(param["conv_ch_s"], param["conv_ch_s"], param["conv_ks_s"])
        self.dropout_2_s = nn.Dropout(p=param["conv_dp"])
        self.conv_3_s = nn.Conv1d(param["conv_ch_s"], param["conv_ch_s"], param["conv_ks_s"])
        self.dropout_3_s = nn.Dropout(p=param["conv_dp"])

        self.lstm = nn.LSTM(param["conv_ch_s"] + param["conv_ch_l"], param["lstm_hs"], batch_first=True)

        self.output_vel = nn.Sequential(
            nn.Linear(param["lstm_hs"], param["fc_ch"]),
            nn.BatchNorm1d(param["fc_ch"]),
            nn.ReLU(),
            nn.Linear(param["fc_ch"], 2),
            TimeNorm(True)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, time, channel) -> (batch, channel)
        hidden = F.relu(self.dropout_1(self.conv_1(input.transpose(1, 2))))
        hidden_l = F.relu(self.dropout_2_l(self.conv_2_l(hidden)))
        hidden_l = F.relu(self.dropout_3_l(self.conv_3_l(hidden_l)))
        hidden_s = F.relu(self.dropout_2_s(self.conv_2_s(hidden)))
        hidden_s = F.relu(self.dropout_3_s(self.conv_3_s(hidden_s)))

        head_pad = (hidden_s.shape[2] - hidden_l.shape[2]) // 2
        tail_pad = hidden_s.shape[2] - hidden_l.shape[2] - head_pad
        hidden = torch.cat((hidden_s.transpose(1, 2)[:, head_pad:-tail_pad], hidden_l.transpose(1, 2)), dim=2)    # (batch, channel, time) -> (batch, time, channel)
        hidden = self.lstm(hidden)[0][:, -1]    # (batch, time, channel) -> (batch, channel)

        output = self.output_vel(hidden)

        return output

class DualCNNXformer(_BaseModule):
    def __init__(self, param: dict[str, util.Param]) -> None:
        super().__init__(param)

        self.conv_1 = nn.Conv1d(2, param["xformer_d_model"] // 2, param["conv_ks_s"])
        self.conv_2_l = nn.Conv1d(param["xformer_d_model"] // 2, param["xformer_d_model"] // 2, 2 * param["conv_ks_s"] - 1)
        self.conv_3_l = nn.Conv1d(param["xformer_d_model"] // 2, param["xformer_d_model"] // 2, 2 * param["conv_ks_s"] - 1)
        self.conv_2_s = nn.Conv1d(param["xformer_d_model"] // 2, param["xformer_d_model"] // 2, param["conv_ks_s"])
        self.conv_3_s = nn.Conv1d(param["xformer_d_model"] // 2, param["xformer_d_model"] // 2, param["conv_ks_s"])

        self.xformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(param["xformer_d_model"], param["xformer_nhead"], dim_feedforward=param["xformer_d_ff"], activation=F.gelu),
            param["xformer_n_layers"]
        )
        self.cls_token = nn.Parameter(data=torch.zeros(1, param["xformer_d_model"]))
        self.position_embed = nn.Parameter(data=torch.randn(406 - 5 * param["conv_ks_s"], 1, param["xformer_d_model"]))

        self.imu_head = nn.Sequential(
            nn.Linear(param["xformer_d_model"], param["xformer_d_model"] // 4),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(param["xformer_d_model"] // 4, 2),
            TimeNorm(False)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, time, channel) -> (batch, channel)
        hidden: torch.Tensor = F.gelu(self.conv_1(input.transpose(1, 2)))
        hidden_l: torch.Tensor = F.gelu(self.conv_2_l(hidden))
        hidden_l = F.gelu(self.conv_3_l(hidden_l))
        hidden_s: torch.Tensor = F.gelu(self.conv_2_s(hidden))
        hidden_s = F.gelu(self.conv_3_s(hidden_s))

        head_pad = (hidden_s.shape[2] - hidden_l.shape[2]) // 2
        tail_pad = hidden_s.shape[2] - hidden_l.shape[2] - head_pad
        hidden = torch.cat((hidden_s[:, :, head_pad:-tail_pad], hidden_l), dim=1).permute(2, 0, 1)    # (batch, channel, time) -> (time, batch, channel)

        cls_token = self.cls_token.unsqueeze(1).repeat(1, hidden.shape[1], 1)
        hidden = torch.cat((cls_token, hidden)) + self.position_embed
        hidden = self.xformer_encoder(hidden)
        hidden = hidden.mean(axis=0)
        output = self.imu_head(hidden)

        return output

def get_model_cls(name: Literal["bilstm", "cnn", "dualcnnlstm", "dualcnnxformer"]) -> type[BiLSTM | CNN | DualCNNLSTM | DualCNNXformer]:
    match name:
        case "bilstm":
            return BiLSTM
        case "cnn":
            return CNN
        case "dualcnnlstm":
            return DualCNNLSTM
        case "dualcnnxformer":
            return DualCNNXformer
        case _:
            raise Exception(f"unknown model {name} was specified")
