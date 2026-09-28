"""Evalúa el agente contra el conjunto de referencia (Fase 5 del roadmap).

Uso: python evaluar.py [--modelo qwen3:8b]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from agente import MODELO, responder
from datos import cargar_todo
from herramientas import ejecutar_consulta
from tarjetas import construir

AQUI = Path(__file__).resolve().parent


def resultado_referencia(item, cache):
    dfs, dics, _ = cache
    plan = item["plan"]
    tj = construir(plan["dataset"], dfs[plan["dataset"]], dics[plan["dataset"]], item.get("confirmar", {}))
    return ejecutar_consulta(plan, dfs[plan["dataset"]], tj)[0]


def mismo_resultado(a, b):
    if a is None or b is None or a.shape != b.shape:
        return False
    return bool(np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-6, equal_nan=True))


INTERNOS = re.compile(r"\b(meta|resultado|json)\b", re.I)
DICE_VACIO = re.compile(r"no (hay|se registr|exist|se encontr|tengo)|sin datos", re.I)


def texto_ok(item, s, ref):
    """Control del texto (no solo de la tabla): sin contradicciones, sin términos internos, con la cifra correcta."""
    t = s["texto"]
    if INTERNOS.search(t):
        return False
    vacio = bool(DICE_VACIO.search(t))
    if item["clase"] == "sin_datos":
        return vacio
    if item["clase"] == "respuesta":
        if vacio:
            return False
        if ref is not None and len(ref) == 1:
            v = float(ref.iloc[0, 0])
            candidatos = {str(int(v))} if v.is_integer() else {f"{round(v, k):g}" for k in (1, 2, 3)}
            return any(c in t for c in candidatos)
    return True


def calificar(item, s, cache):
    ok_clase = s["clase"] == item["clase"]
    detalle = {"clase_ok": ok_clase}
    ref = resultado_referencia(item, cache) if item.get("plan") else None
    detalle["texto_ok"] = texto_ok(item, s, ref)
    if item["clase"] == "respuesta" and ok_clase:
        detalle["resultado_ok"] = mismo_resultado(s["tabla"], ref)
    elif item["clase"] == "aclaracion":
        if item.get("columna"):  # aclaración semántica: debe nombrar la columna dudosa
            detalle["resultado_ok"] = ok_clase and f"`{item['columna']}`" in s["texto"]
        else:  # aclaración decidida por el LLM: debe ser una pregunta
            detalle["resultado_ok"] = ok_clase and "?" in s["texto"]
    elif item["clase"] == "calidad":
        texto = s["texto"].lower()
        cubiertos = [any(k.lower() in texto for k in grupo) for grupo in item["debe_mencionar"]]
        detalle["resultado_ok"] = ok_clase and all(cubiertos)
        detalle["cobertura"] = f"{sum(cubiertos)}/{len(cubiertos)}"
    elif item["clase"] == "sin_datos":
        detalle["resultado_ok"] = ok_clase and bool(re.search(r"no (hay|se registr|exist|se encontr|tengo)|sin datos",
                                                              s["texto"].lower()))
    else:
        detalle["resultado_ok"] = ok_clase
    return detalle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=MODELO)
    ap.add_argument("--gold", default="gold_set_v0.json")
    ap.add_argument("--particion", default="todas", choices=["todas", "desarrollo", "prueba"])
    args = ap.parse_args()
    modelo = args.modelo
    items = json.loads((AQUI / args.gold).read_text(encoding="utf-8"))
    if args.particion != "todas":
        items = [i for i in items if i.get("particion") == args.particion]
    cache = cargar_todo()
    filas = []
    for it in items:
        s = responder(it["pregunta"], modelo, confirmaciones=it.get("confirmar", {}), cache=cache)
        d = calificar(it, s, cache)
        r = s["registro"]
        filas.append({"id": it["id"], "esperado": it["clase"], "obtenido": s["clase"], **d,
                      "anclaje_inicial": r["anclaje_inicial_ok"], "anclaje_final": r["anclaje_final_ok"],
                      "reparaciones": r["reparaciones_plan"] + r["reparaciones_texto"],
                      "respaldo": r["respaldo_determinista"], "segundos": s["segundos"],
                      "plan": s["plan"], "texto": s["texto"]})
        marca = "OK " if d["resultado_ok"] and d["texto_ok"] else "MAL"
        print(f"[{marca}] {it['id']} {it['clase']:>16} → {s['clase']:<16} {s['segundos']:>6} s | {s['texto'][:110]!r}")

    n = len(filas)
    con_texto = [f for f in filas if f["anclaje_final"] is not None]
    metricas = {
        "modelo": modelo,
        "preguntas": n,
        "resultado_correcto_%": round(100 * sum(f["resultado_ok"] for f in filas) / n, 1),
        "clase_correcta_%": round(100 * sum(f["clase_ok"] for f in filas) / n, 1),
        "texto_correcto_%": round(100 * sum(f["texto_ok"] for f in filas) / n, 1),
        "aclaracion_cuando_corresponde_%": round(100 * np.mean([f["obtenido"] == "aclaracion" for f in filas if f["esperado"] == "aclaracion"]), 1),
        "aclaracion_innecesaria_%": round(100 * np.mean([f["obtenido"] == "aclaracion" for f in filas if f["esperado"] != "aclaracion"]), 1),
        "anclaje_inicial_%": round(100 * np.mean([f["anclaje_inicial"] for f in con_texto]), 1) if con_texto else None,
        "anclaje_final_%": round(100 * np.mean([f["anclaje_final"] for f in con_texto]), 1) if con_texto else None,
        "reparaciones_totales": int(sum(f["reparaciones"] for f in filas)),
        "segundos_mediana": float(np.median([f["segundos"] for f in filas])),
    }
    print("\n" + json.dumps(metricas, indent=2, ensure_ascii=False))
    (AQUI / "resultados").mkdir(exist_ok=True)
    nombre = f"{Path(args.gold).stem}_{args.particion}_" + modelo.replace(":", "_").replace("/", "_")
    (AQUI / "resultados" / f"eval_{nombre}.json").write_text(
        json.dumps({"metricas": metricas, "detalle": filas}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
