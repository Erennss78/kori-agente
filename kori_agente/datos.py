"""Adaptadores a formato canónico (replicabilidad, puntos 1 y 3).

Formato canónico: DataFrame con índice de tiempo `fecha_hora` (si la fuente declara columna de tiempo)
y una columna por variable. Cada fuente se describe en `fuentes.json`: no hay código por dataset.
Los metadatos (periodo, cadencia) se calculan de los datos, nunca se escriben a mano.
"""
import json
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
FUENTES = AQUI / "fuentes.json"


def _ruta(p):
    p = Path(p)
    return p if p.is_absolute() else RAIZ / p


def leer_fuente(cfg):
    df = pd.read_csv(_ruta(cfg["archivo"]), sep=cfg.get("sep", ","))
    if cfg["formato"] == "largo":
        df[cfg["tiempo"]] = pd.to_datetime(df[cfg["tiempo"]])
        w = df.pivot_table(index=cfg["tiempo"], columns=cfg["variable"], values=cfg["valor"])
    elif cfg["formato"] == "ancho":
        w = df
        if cfg.get("tiempo"):
            w[cfg["tiempo"]] = pd.to_datetime(w[cfg["tiempo"]])
            w = w.set_index(cfg["tiempo"])
    else:
        raise ValueError(f"formato desconocido: {cfg['formato']}")
    w = w.sort_index()
    if isinstance(w.index, pd.DatetimeIndex):
        w.index.name = "fecha_hora"
    w.columns = [str(c) for c in w.columns]
    return w


def leer_diccionario(cfg):
    if not cfg:
        return None
    d = pd.read_csv(_ruta(cfg["archivo"]), sep=cfg.get("sep", ","))
    return {str(f[cfg["id"]]): {"significado": f[cfg["significado"]], "unidad": f[cfg["unidad"]],
                                "minv": float(f[cfg["min"]]), "maxv": float(f[cfg["max"]])}
            for _, f in d.iterrows()}


def describir(cfg, df):
    m = {"descripcion": cfg["descripcion"], "filas": len(df)}
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        paso = df.index.to_series().diff().dt.total_seconds().median() / 60
        m.update(inicio=f"{df.index.min():%Y-%m-%d %H:%M}", fin=f"{df.index.max():%Y-%m-%d %H:%M}",
                 cadencia_min=round(float(paso), 1))
        m["texto"] = (f"{cfg['descripcion']}; datos del {m['inicio']} al {m['fin']}, "
                      f"una lectura cada ~{m['cadencia_min']:g} min")
    else:
        m["texto"] = f"{cfg['descripcion']}; {len(df)} filas, sin columna de tiempo identificada"
    return m


def cargar_todo(solo=None):
    """Devuelve (dfs, diccionarios, metadatos) para todas las fuentes de fuentes.json."""
    cfgs = json.loads(FUENTES.read_text(encoding="utf-8"))
    dfs, dics, metas = {}, {}, {}
    for nombre, cfg in cfgs.items():
        if solo and nombre not in solo:
            continue
        if not _ruta(cfg["archivo"]).exists():  # p. ej. los datos del laboratorio no vienen en el repositorio
            print(f"[datos] se omite '{nombre}': no existe {cfg['archivo']}")
            continue
        dfs[nombre] = leer_fuente(cfg)
        dics[nombre] = leer_diccionario(cfg.get("diccionario"))
        metas[nombre] = describir(cfg, dfs[nombre])
    return dfs, dics, metas
