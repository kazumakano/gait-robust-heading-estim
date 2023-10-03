import math
import os.path as path
import pickle
from datetime import datetime
from typing import Optional
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt


Param = bool | float | int | str

def calc_gcs_hor_acc(scs_acc: np.ndarray, quat: np.ndarray, scs_grav: Optional[np.ndarray] = None) -> np.ndarray:
    if scs_grav is None:
        return Rotation.from_quat(quat).apply(scs_acc)[:, :2]
    else:
        scs_vert_acc = ((scs_acc * scs_grav).sum(axis=1) / (scs_grav ** 2).sum(axis=1))[:, np.newaxis] * scs_grav
        return Rotation.from_quat(quat).apply(scs_acc - scs_vert_acc)[:, :2]

def calc_mean_heading(pos: np.ndarray, calc_range: int, win_len: int, win_stride: int) -> np.ndarray:
    heading = np.empty((len(pos) - calc_range, 2), dtype=np.float64)
    for i in range(len(pos) - calc_range):
        norm = np.linalg.norm(pos[i + calc_range] - pos[i])
        heading[i] = ((pos[i + calc_range, 0] - pos[i, 0]) / norm, (pos[i + calc_range, 1] - pos[i, 1]) / norm)

    win_num = (len(heading) - win_len) // win_stride + 1
    mean_heading = np.empty((win_num, 2), dtype=np.float32)
    for i in range(win_num):
        mean_heading[i] = heading[i * win_stride:i * win_stride + win_len].mean(axis=0)

    return mean_heading

def calc_spd(pos: np.ndarray, calc_range: int) -> np.ndarray:
    spd = np.empty(len(pos) - calc_range, dtype=np.float64)
    for i in range(len(pos) - calc_range):
        spd[i] = 100 * np.linalg.norm(pos[i + calc_range] - pos[i]) / calc_range

    return spd

def derive_pos_with_every_win_heading(heading: np.ndarray, spd: np.ndarray, win_stride: int, init_pos: Optional[np.ndarray] = None) -> np.ndarray:
    pos = np.empty((len(spd) + 1, 2), dtype=np.float32)

    current_pos = np.zeros(2, dtype=np.float64) if init_pos is None else init_pos.copy()
    pos[0] = current_pos

    for i in range(len(spd)):
        current_pos += spd[i] * win_stride / 100 * np.array((math.cos(heading[i]), math.sin(heading[i])), dtype=np.float64)
        pos[i + 1] = current_pos

    return pos

def gaussian_filter(ar: np.ndarray, freq: int, radius: int, sigma: float) -> np.ndarray:
    sigma *= freq
    kernel = np.empty(2 * radius + 1, dtype=np.float64)
    for i in range(2 * radius + 1):
        kernel[i] = math.exp((i - radius) ** 2 / (-2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    kernel /= kernel.sum()

    for i in range(ar.shape[1]):
        ar[:, i] = np.convolve(np.pad(ar[:, i], (radius, radius), mode="edge"), kernel, mode="valid")

    return ar

def get_result_dir(dir_name: str | None) -> str:
    if dir_name is None:
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return path.join(path.dirname(__file__), "../result/", dir_name)

def get_win_center_ts(ts: np.ndarray, win_len: int, win_stride: int) -> np.ndarray:
    win_num = (len(ts) - win_len) // win_stride + 1
    center_ts = np.empty(win_num, dtype=np.float64)
    for i in range(win_num):
        center_ts[i] = (ts[i * win_stride] + ts[i * win_stride + win_len - 1]) / 2

    return center_ts

def load_param(file: str) -> dict[str, Param | list[Param]]:
    with open(file) as f:
        return yaml.safe_load(f)

def load_pos(file: str) -> np.ndarray:
    return np.loadtxt(file, dtype=np.float64, delimiter=",")[:, 17:19]

def load_test_result(result_dir: str, ver: int = 0) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Param]]:
    with open(path.join(result_dir, f"version_{ver}/", "test_outputs.pkl"), mode="rb") as f:
        return pickle.load(f), load_param(path.join(result_dir, f"version_{ver}/", "hparams.yaml"))

def _unix2datetime(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype(object)

    for i, t in enumerate(ts):
        ts[i] = datetime.fromtimestamp(t)

    return ts.astype(datetime)

def plot_outputs(outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(nrows=len(outputs), sharey=True, figsize=(16, 4 * len(outputs)))
    for i, o in enumerate(outputs):
        ts = _unix2datetime(o[0])
        axes[i].scatter(ts, np.rad2deg(np.arctan2(o[1][:, 1], o[1][:, 0])), s=2, marker=".")
        axes[i].scatter(ts, np.rad2deg(np.arctan2(o[2][:, 1], o[2][:, 0])), s=2, marker=".")
        axes[i].set_ylabel("heading [°]")
    axes[-1].set_xlabel("time")

    fig.show()

def vis_tj(estim: np.ndarray, truth: np.ndarray) -> None:
    plt.axis("equal")
    plt.plot(estim[:, 0], estim[:, 1])
    plt.plot(truth[:, 0], truth[:, 1])
    plt.xlabel("position x [m]")
    plt.ylabel("position y [m]")
    plt.show()
