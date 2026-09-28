"""Corre los mismos conjuntos de referencia con varios modelos y resume los resultados.

Uso: python comparar_modelos.py qwen3:8b qwen3.5:9b nemotron-3.5-lightning:30b
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
CONJUNTOS = [("gold_auto_psha_s1.json", "prueba"), ("gold_ambiguedad_psha.json", "todas")]
COLUMNAS = ["resultado_correcto_%", "texto_correcto_%", "aclaracion_cuando_corresponde_%",
            "aclaracion_innecesaria_%", "segundos_mediana"]


def main(modelos):
    filas = []
    for modelo in modelos:
        for gold, part in CONJUNTOS:
            subprocess.run([sys.executable, str(AQUI / "evaluar.py"), "--gold", gold, "--particion", part,
                            "--modelo", modelo], cwd=AQUI, check=True)
            nombre = f"{Path(gold).stem}_{part}_" + modelo.replace(":", "_").replace("/", "_")
            m = json.loads((AQUI / "resultados" / f"eval_{nombre}.json").read_text(encoding="utf-8"))["metricas"]
            filas.append({"modelo": modelo, "conjunto": Path(gold).stem, **{c: m.get(c) for c in COLUMNAS}})
    tabla = pd.DataFrame(filas)
    print("\n" + tabla.to_string(index=False))
    tabla.to_csv(AQUI / "resultados" / "comparacion_modelos.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen3:8b"])
