from __future__ import annotations

import numpy as np


DEFAULT_BOX_SIZE_XYZ = (0.3, 0.3, 0.3)


def parse_box_size_xyz(value) -> np.ndarray:
    if isinstance(value, str):
        text = value.replace("[", " ").replace("]", " ").replace(",", " ")
        parts = text.split()
        arr = np.asarray([float(x) for x in parts], dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)

    if arr.size != 3:
        raise ValueError(f"box_size_xyz must contain 3 values, got {arr.tolist()}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"box_size_xyz must contain positive finite values, got {arr.tolist()}")
    return arr
