"""Junta en una sola tabla los resultados ya guardados de todos los modelos evaluados.

Uso: python tabla_modelos.py
"""
import json
from pathlib import Path

import pandas as pd

from comparar_modelos import COLUMNAS, CONJUNTOS

AQUI = Path(__file__).resolve().parent


def main():
    filas = []
    for gold, part in CONJUNTOS:
        for f in sorted((AQUI / "resultados").glob(f"eval_{Path(gold).stem}_{part}_*.json")):
            m = json.loads(f.read_text(encoding="utf-8"))["metricas"]
            filas.append({"modelo": m["modelo"], "conjunto": Path(gold).stem, **{c: m.get(c) for c in COLUMNAS}})
    tabla = pd.DataFrame(filas).sort_values(["conjunto", "modelo"])
    print(tabla.to_string(index=False))
    tabla.to_csv(AQUI / "resultados" / "comparacion_modelos.csv", index=False)


if __name__ == "__main__":
    main()
