"""Fase 1 del roadmap: tarjetas semánticas y detector de ambigüedad.

Regla: ninguna columna se analiza si su significado no está confirmado.
El detector es determinista; las hipótesis salen de la evidencia de los datos.
"""
import json
from pathlib import Path

import pandas as pd

ARCHIVO = Path(__file__).resolve().parent / "tarjetas_confirmadas.json"
GENERICOS = {"t", "h", "v", "p", "x", "y", "val", "valor", "data", "dato", "col", "c"}


def cargar_confirmaciones():
    return json.loads(ARCHIVO.read_text(encoding="utf-8")) if ARCHIVO.exists() else {}


def confirmar(dataset, columna, significado, unidad, tipo, por):
    """Guarda lo que confirmó una persona para no volver a preguntar.
    tipo: 'medicion' | 'acumulado' | 'tiempo' | 'calibracion' | 'falla'."""
    c = cargar_confirmaciones()
    c[f"{dataset}:{columna}"] = {"significado": significado, "unidad": unidad, "tipo": tipo,
                                 "confirmado_por": por, "fecha": pd.Timestamp.now().strftime("%Y-%m-%d")}
    ARCHIVO.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")


def evidencia(x):
    x = x.dropna()
    d = x.diff().dropna()
    cambios = d.abs()[d.abs() > 0]
    return {
        "min": float(x.min()), "max": float(x.max()), "unicos": int(x.nunique()),
        "sube": float((d > 0).mean()), "baja": float((d < 0).mean()), "igual": float((d == 0).mean()),
        "paso": float(cambios.median()) if len(cambios) else 0.0,
        "paso_constante": bool(len(cambios) and cambios.nunique() == 1),
        "ceros": float((x == 0).mean()),
        "reinicios": int(((d < 0) & (x.iloc[1:].to_numpy() <= x.max() * 0.05)).sum()),
    }


def hipotesis(ev):
    h = []
    if ev["unicos"] == 1:
        h.append("valor constante: probable parámetro de calibración o configuración, no una medición")
    elif ev["baja"] == 0 and ev["sube"] > 0.9:
        paso = f" (sube siempre de {ev['paso']:g} en {ev['paso']:g})" if ev["paso_constante"] else ""
        h.append(f"siempre crece{paso}: probable tiempo o contador")
    elif ev["reinicios"] >= 2 and ev["igual"] > 0.5 and ev["min"] >= 0 and ev["ceros"] < 0.9:
        h.append(f"acumulado que se reinicia a 0 ({ev['reinicios']} reinicios): el total NO es la suma de las lecturas")
    elif ev["ceros"] > 0.9:
        h.append("casi siempre marca 0: sensor apagado, desconectado o sin señal")
    if ev["unicos"] > 1 and not h:
        lo, hi = ev["min"], ev["max"]
        if -40 <= lo and hi <= 60:
            h.append("temperatura en °C")
        if 0 <= lo and hi <= 100:
            h.append("porcentaje (p. ej. humedad relativa %)")
        if 0 <= lo and hi <= 23:
            h.append("hora del día")
        if 900 <= lo and hi <= 1100:
            h.append("presión en hPa")
        if 10 <= lo and hi <= 15:
            h.append("voltaje de una batería de 12 V")
    return h


def construir(dataset, df, diccionario, confirmaciones=None):
    confirmaciones = cargar_confirmaciones() if confirmaciones is None else confirmaciones
    tarjetas = {}
    for col in df.columns:
        ev = evidencia(df[col])
        hip = hipotesis(ev)
        entrada = (diccionario or {}).get(col)
        motivos = []
        if entrada is None:
            motivos.append("no está en el diccionario de datos")
            if len(col) <= 3 or col.lower() in GENERICOS:
                motivos.append("nombre corto o genérico")
            if len(hip) != 1:
                motivos.append("los valores no bastan para saber qué es" if not hip else "los valores encajan con varios significados")
        else:
            fuera = ((df[col] < entrada["minv"]) | (df[col] > entrada["maxv"])).mean()
            if fuera > 0.5:
                motivos.append(f"el diccionario dice rango {entrada['minv']:g}–{entrada['maxv']:g} {entrada['unidad']}, "
                               f"pero el {fuera:.0%} de los valores está fuera (¿sensor muerto o unidad distinta?)")
            if any(k in " ".join(hip) for k in ("constante", "acumulado", "marca 0")):
                motivos.append("el diccionario no aclara cómo se registra este valor")
        confirmada = confirmaciones.get(f"{dataset}:{col}")
        base = {"significado": entrada["significado"], "unidad": entrada["unidad"], "tipo": "medicion",
                "fuente": "diccionario oficial"} if entrada else {"significado": None, "unidad": None, "tipo": None,
                                                                  "fuente": None}
        if confirmada:
            base.update({k: confirmada[k] for k in ("significado", "unidad", "tipo")})
            base["fuente"] = f"confirmado por {confirmada['confirmado_por']} ({confirmada['fecha']})"
        tarjetas[col] = {**base, "estado": "confirmado" if (confirmada or not motivos) else "requiere_confirmacion",
                         "motivos": [] if confirmada else motivos, "hipotesis": hip, "evidencia": ev}
    return tarjetas


def pregunta_aclaracion(descripcion, columnas, tarjetas):
    lineas = [f"Antes de responder necesito confirmar qué significan estas columnas ({descripcion}):"]
    for c in columnas:
        t, ev = tarjetas[c], tarjetas[c]["evidencia"]
        dato = f"valores de {ev['min']:g} a {ev['max']:g}"
        hip = "; ".join(t["hipotesis"]) or "sin hipótesis clara"
        lineas.append(f"- `{c}`: {'; '.join(t['motivos'])}. Datos: {dato}. Posible: {hip}.")
    lineas.append("¿Qué significa cada una, en qué unidad está y cómo debe agregarse (promedio, suma, último valor)?")
    return "\n".join(lineas)
