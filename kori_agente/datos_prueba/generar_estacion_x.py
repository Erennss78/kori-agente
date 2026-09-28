"""Genera estacion_x.csv: dataset sintético con nombres ambiguos (t, h, v, p) para probar la regla de preguntar."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 144 * 3
t = np.arange(n) * 600
hora = (t / 3600) % 24
pd.DataFrame({
    "t": t,
    "h": np.clip(80 - 15 * np.sin(2 * np.pi * (hora - 9) / 24) + rng.normal(0, 2, n), 40, 100).round(1),
    "v": (12.6 - 0.002 * np.arange(n) + rng.normal(0, 0.02, n)).round(2),
    "p": (1013 + rng.normal(0, 0.8, n).cumsum() * 0.1).round(1),
}).to_csv(Path(__file__).with_name("estacion_x.csv"), index=False)
