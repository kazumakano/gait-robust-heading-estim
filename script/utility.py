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
