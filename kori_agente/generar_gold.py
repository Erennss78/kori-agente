"""Genera automáticamente un conjunto de referencia para cualquier dataset de fuentes.json (replicabilidad, punto 4).

Preguntas = plantillas × columnas confirmadas × agregaciones × periodos al azar. La respuesta correcta la calculan
las herramientas deterministas. Se agregan trampas: columnas sin confirmar, periodo sin datos, fuera del periodo,
pronóstico y calidad. Cada pregunta queda en la partición "desarrollo" o "prueba".

Uso: python generar_gold.py psha [--semilla 0] [--prueba 0.6]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datos import cargar_todo
from herramientas import ejecutar_consulta, reporte_calidad
from tarjetas import construir

AQUI = Path(__file__).resolve().parent
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
TEXTO_AGG = {"promedio": "el promedio", "maximo": "el valor máximo", "minimo": "el valor mínimo"}
TEXTO_DIARIO = {"promedio": "el promedio diario", "maximo": "el máximo diario", "minimo": "el mínimo diario"}


def fecha(ts):
    return f"{ts.day} de {MESES[ts.month - 1]} de {ts.year}"


def generar(ds, semilla=0, frac_prueba=0.6):
    dfs, dics, metas = cargar_todo([ds])
    df, meta = dfs[ds], metas[ds]
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"'{ds}' no tiene columna de tiempo confirmada: primero hay que incorporarlo")
    tj = construir(ds, df, dics[ds])
    rng = np.random.default_rng(semilla)
    dias = pd.DatetimeIndex(sorted(df.index.normalize().unique()))
    cols = [c for c, t in tj.items() if t["estado"] == "confirmado" and t["tipo"] == "medicion"
            and df[c].notna().mean() > 0.5]
    items = []

    def agregar(pregunta, clase, **extra):
        plan = extra.get("plan")
        if plan is not None and clase == "respuesta":
            tabla, _ = ejecutar_consulta(plan, df, tj)
            if tabla.empty:
                clase = "sin_datos"
        items.append({"id": f"{ds}-{len(items) + 1:03d}", "pregunta": pregunta, "clase": clase, **extra})

    def plan(col, d1, d2, agg, por):
        return {"dataset": ds, "columnas": [col], "desde": f"{d1:%Y-%m-%d} 00:00" if d1 is not None else None,
                "hasta": f"{d2 + pd.Timedelta(days=1):%Y-%m-%d} 00:00" if d2 is not None else None,
                "agregacion": agg, "por": por}

    for c in cols:
        sig = tj[c]["significado"].lower()
        d = dias[rng.integers(len(dias))]
        agg = rng.choice(list(TEXTO_AGG))
        agregar(f"¿Cuál fue {TEXTO_AGG[agg]} de {sig} el {fecha(d)}?", "respuesta", plan=plan(c, d, d, agg, "total"))
        i = int(rng.integers(0, len(dias) - 4))
        d1, d2 = dias[i], dias[i + int(rng.integers(1, 4))]
        agg = rng.choice(list(TEXTO_AGG))
        agregar(f"¿Cuál fue {TEXTO_AGG[agg]} de {sig} entre el {fecha(d1)} y el {fecha(d2)}?", "respuesta",
                plan=plan(c, d1, d2, agg, "total"))
        agg = rng.choice(list(TEXTO_DIARIO))
        agregar(f"Dame {TEXTO_DIARIO[agg]} de {sig} del {fecha(d1)} al {fecha(d2)}", "respuesta",
                plan=plan(c, d1, d2, agg, "dia"))
    for c in rng.choice(cols, size=min(2, len(cols)), replace=False):
        sig = tj[c]["significado"].lower()
        agregar(f"¿Cuántas lecturas de {sig} hay en todo el periodo de datos?", "respuesta",
                plan=plan(c, None, None, "conteo", "total"))
        d = dias[rng.integers(len(dias))]
        agregar(f"Dame el promedio por hora de {sig} el {fecha(d)}", "respuesta", plan=plan(c, d, d, "promedio", "hora"))

    # Trampas
    for c, t in tj.items():
        if t["estado"] != "confirmado":
            agregar(f"¿Cuál fue el promedio de {c} en todo el periodo de datos?", "aclaracion", columna=c)
    sig0 = tj[cols[0]]["significado"].lower()
    huecos = df.index.to_series().diff()
    if meta.get("cadencia_min") and huecos.max() > pd.Timedelta(minutes=6 * meta["cadencia_min"]):
        fin = huecos.idxmax()
        medio = (fin - huecos.max() / 2).floor("h")
        agregar(f"¿Qué {sig0} había el {fecha(medio)} a las {medio:%H}:00?", "sin_datos",
                nota="momento dentro de la caída más larga del nodo")
    fuera = df.index.max() + pd.Timedelta(days=45)
    agregar(f"¿Cuál fue el promedio de {sig0} el {fecha(fuera)}?", "sin_datos", nota="fecha fuera del periodo de datos")
    agregar(f"¿Qué {sig0} habrá mañana?", "fuera_de_alcance")
    _, hallazgos = reporte_calidad(df, tj, meta.get("cadencia_min"))
    grupos = []
    if any("dejó de enviar" in h for h in hallazgos):
        grupos.append(["dejó de enviar", "caída", "caídas", "sin datos"])
    for c, t in tj.items():
        if any(h.startswith(c) for h in hallazgos) and len(grupos) < 4:
            grupos.append([c, (t["significado"] or c).lower()])
    agregar("¿Cómo está la calidad de los datos?", "calidad", debe_mencionar=grupos)

    orden = rng.permutation(len(items))
    for k, idx in enumerate(orden):
        items[idx]["particion"] = "prueba" if k < round(frac_prueba * len(items)) else "desarrollo"
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--prueba", type=float, default=0.6)
    a = ap.parse_args()
    items = generar(a.dataset, a.semilla, a.prueba)
    ruta = AQUI / f"gold_auto_{a.dataset}.json"
    ruta.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    clases = pd.Series([i["clase"] for i in items]).value_counts().to_dict()
    parts = pd.Series([i["particion"] for i in items]).value_counts().to_dict()
    print(f"{len(items)} preguntas → {ruta.name} | clases {clases} | particiones {parts}")


if __name__ == "__main__":
    main()
