import itertools
import sys
from functools import reduce
from typing import Dict, List
from PIL import Image
import numpy as np
import glob
from pathlib import Path
from torch.nn.modules.module import _addindent
import torchvision.transforms as T
import torch


def resize(im: np.ndarray, target_size=[256, 256]):
    assert im.ndim <= 3, ""
    _im = torch.Tensor(im)
    _im = _im[None, ...] if im.ndim == 2 else _im

    [w, h] = target_size
    return T.Resize(size=(w, h))(_im)


def make_nested_dir(directory: str) -> str:
    """Make nested Directory

    Args:
        directory (str): Path to directory

    Returns:
        str: Path to that directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


GROUP1 = "FLARE22_Tr_0001_0000_abdomen-soft tissues_abdomen-liver"
GROUP2 = "FLARE22_Tr_0001_0000_chest-lungs_chest-mediastinum"
GROUP3 = "FLARE22_Tr_0001_0000_spine-bone"


def get_data_paths(GROUP):
    data = list(glob.glob(f"../dataset/FLARE-small/{GROUP}/*"))
    mask = list(glob.glob("../dataset/FLARE-small/FLARE22_Tr_0001_0000-mask/*"))
    data = sorted(data)
    mask = sorted(mask)
    return data, mask


def load_img(path):
    return np.asarray(Image.open(path).convert("RGB"))


def load_file_npz(npz_path) -> np.ndarray:
    return np.load(npz_path)


def argmax_dist(coors, x, y):
    return np.argmax(np.sqrt(np.square(coors[:, 0] - x) + np.square(coors[:, 0] - y)))


def generate_grid(w, h, est_n_point=16):
    n_axis_sampling = int(np.sqrt(est_n_point))
    w_axis = np.linspace(0, w, n_axis_sampling)
    h_axis = np.linspace(0, h, n_axis_sampling)
    samples = np.array(list(itertools.product(w_axis, h_axis))).astype(np.int32)
    labels = np.ones(n_axis_sampling**2)
    return samples, labels


def mask_out(mask, xmin, xmax, ymin, ymax, to_value):
    _mask = np.ones(mask.shape) == 1.0
    _mask[xmin:xmax, ymin:ymax] = False
    mask[_mask] = to_value
    return mask


def pick(d: Dict[object, object], keys: List[object]):
    return {k: v for k, v in d.items() if k in keys}


def omit(d: Dict[object, object], keys: List[object]):
    return {k: v for k, v in d.items() if k not in keys}


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, "shape"):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        if file is sys.stdout:
            main_str += ", \033[92m{:,}\033[0m params".format(total_params)
        else:
            main_str += ", {:,} params".format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, "w")
        print(string, file=file)
        file.flush()

    return count
