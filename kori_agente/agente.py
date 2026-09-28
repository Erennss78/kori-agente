"""Prototipo v0 del agente analista de Kori (cadena B del roadmap).

B1+B3 plan JSON (Qwen con esquema forzado) → B2 verificación semántica → B4 código calcula
→ B5 reparación → B6 anclaje numérico → respuesta (texto y/o tabla).

Uso: python agente.py "¿Cuál fue la temperatura promedio el 10 de marzo?" [--modelo qwen3:8b]
"""
import argparse
import json
import re
import time
import urllib.request

import pandas as pd

from datos import cargar_todo
from herramientas import AGREGACIONES, ejecutar_consulta, reporte_calidad
from tarjetas import cargar_confirmaciones, construir, pregunta_aclaracion

OLLAMA = "http://localhost:11434/api/chat"
MODELO = "qwen3:8b"


def llm(mensajes, modelo, esquema=None):
    cuerpo = {"model": modelo, "messages": mensajes, "stream": False, "think": False,
              "options": {"temperature": 0, "num_ctx": 8192}}
    if esquema:
        cuerpo["format"] = esquema
    req = urllib.request.Request(OLLAMA, data=json.dumps(cuerpo).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["message"]["content"]


# ---------------------------------------------------------------- B1+B3: plan
def esquema_plan(dfs):
    columnas = sorted({c for df in dfs.values() for c in df.columns})
    return {"type": "object",
            "required": ["tipo", "pregunta_al_usuario", "dataset", "columnas", "desde", "hasta", "agregacion", "por", "formato"],
            "properties": {
                "tipo": {"type": "string", "enum": ["consulta", "calidad", "preguntar", "pronostico", "otro"]},
                "pregunta_al_usuario": {"type": ["string", "null"]},
                "dataset": {"type": "string", "enum": list(dfs)},
                "columnas": {"type": "array", "items": {"type": "string", "enum": columnas}},
                "desde": {"type": ["string", "null"]},
                "hasta": {"type": ["string", "null"]},
                "agregacion": {"type": "string", "enum": list(AGREGACIONES) + ["ninguna"]},
                "por": {"type": "string", "enum": ["total", "hora", "dia"]},
                "formato": {"type": "string", "enum": ["valor", "tabla", "texto+tabla"]}}}


def catalogo(dfs, tarjetas, metas):
    lineas = []
    for ds in dfs:
        lineas.append(f"Dataset '{ds}': {metas[ds]['texto']}")
        for c, t in tarjetas[ds].items():
            sig = t["significado"] or "significado desconocido"
            lineas.append(f"  - {c}: {sig}" + (f" ({t['unidad']})" if t["unidad"] else ""))
    return "\n".join(lineas)


SISTEMA_PLAN = """Eres el planificador de un sistema de análisis de datos de sensores. No respondes la pregunta:
solo produces un plan JSON que un programa ejecutará.

Columnas disponibles:
{catalogo}

Fecha de hoy: {hoy}.

Herramienta "preguntar": si la petición es ambigua y una interpretación equivocada daría otra respuesta, NO adivines.
Usa tipo "preguntar" y escribe en "pregunta_al_usuario" UNA pregunta corta y concreta, con las opciones posibles
tomadas de la lista de columnas. Es ambigua si: no dice qué variable y hay varias candidatas (p. ej. "el panel"
puede ser voltaje o corriente); no dice qué quiere saber de ella ("¿cómo estuvo X?" sin periodo ni operación);
usa palabras vagas ("el sensor", "eso", "lo raro", "el martes" sin fecha). NO preguntes si la pregunta ya dice variable,
periodo y operación, o si hay una sola interpretación razonable. Si no preguntas, "pregunta_al_usuario" es null.

Reglas:
- tipo: "consulta" si pide valores o estadísticas; "calidad" si pregunta por la calidad de los datos, errores,
  problemas o fallas de sensores; "preguntar" si la petición es ambigua (ver arriba); "pronostico" si pide valores
  futuros; "otro" si no se puede responder con estos datos.
- columnas: solo nombres exactos de la lista. Usa las columnas que corresponden a lo que se nombra en la pregunta.
- desde / hasta: formato "YYYY-MM-DD HH:MM". "hasta" es EXCLUSIVO: un día completo D va de "D 00:00" a las
  00:00 del día siguiente. "Entre el día A y el día B" incluye el día B completo. Si la pregunta no limita fechas
  o abarca todo el periodo de datos ("en todo el periodo", "en total", el mes completo de los datos), usa null en ambos.
  Si la pregunta no dice el año, usa el año del periodo de datos del dataset.
- agregacion: promedio, maximo, minimo, suma, conteo, ultimo, o "ninguna" para ver lecturas sueltas.
- por: "total" para un solo resultado; "dia" o "hora" si pide el valor de cada día u hora.
- formato: "valor" si la respuesta es un solo número; "tabla" si son varias filas; "texto+tabla" para calidad.

Ejemplo con un dataset ficticio 'ejemplo' (columna HUM = humedad relativa, datos de junio de 2025):
"¿Cuál fue la humedad máxima de cada día del 2 al 3 de junio?" →
{{"tipo":"consulta","pregunta_al_usuario":null,"dataset":"ejemplo","columnas":["HUM"],"desde":"2025-06-02 00:00",
"hasta":"2025-06-04 00:00","agregacion":"maximo","por":"dia","formato":"tabla"}}"""


def validar_plan(plan, dfs, pregunta=""):
    errores = []
    for k in ("desde", "hasta"):  # límites inventados (p. ej. 23:50) recortan lecturas sin que nadie lo note
        if plan.get(k) and not plan[k].endswith(":00") and ":" not in pregunta:
            errores.append(f"'{k}' = {plan[k]} tiene minutos que la pregunta no menciona. Usa horas exactas (HH:00); "
                           "si la pregunta abarca días completos, 'hasta' es las 00:00 del día siguiente, y si abarca "
                           "todo el periodo de datos, usa null en 'desde' y 'hasta'")
    df = dfs[plan["dataset"]]
    faltan = [c for c in plan["columnas"] if c not in df.columns]
    if faltan:
        errores.append(f"las columnas {faltan} no existen en el dataset '{plan['dataset']}'")
    if plan["tipo"] == "consulta" and not plan["columnas"]:
        errores.append("una consulta necesita al menos una columna")
    if plan["tipo"] == "preguntar" and not (plan.get("pregunta_al_usuario") or "").strip():
        errores.append("tipo 'preguntar' necesita el texto de la pregunta en 'pregunta_al_usuario'")
    for k in ("desde", "hasta"):
        if plan[k]:
            try:
                pd.Timestamp(plan[k])
            except ValueError:
                errores.append(f"'{k}' = {plan[k]!r} no es una fecha válida (usa YYYY-MM-DD HH:MM)")
    return errores


def planificar(pregunta, dfs, tarjetas, metas, modelo, registro, historial=()):
    """historial: turnos previos [(rol, texto)], p. ej. la aclaración que pidió el agente y la respuesta del usuario."""
    sistema = SISTEMA_PLAN.format(catalogo=catalogo(dfs, tarjetas, metas), hoy=f"{pd.Timestamp.now():%Y-%m-%d}")
    mensajes = [{"role": "system", "content": sistema}]
    mensajes += [{"role": "user" if r == "usuario" else "assistant", "content": t} for r, t in historial]
    mensajes.append({"role": "user", "content": pregunta})
    for intento in range(3):
        crudo = llm(mensajes, modelo, esquema_plan(dfs))
        plan = json.loads(crudo)
        if plan.get("hasta") and plan["hasta"].endswith("23:59"):  # "hasta" es exclusivo: 23:59 → 00:00 del día siguiente
            plan["hasta"] = str(pd.Timestamp(plan["hasta"]) + pd.Timedelta(minutes=1))[:16]
        errores = validar_plan(plan, dfs, pregunta)
        if not errores:
            return plan
        registro["reparaciones_plan"] += 1
        mensajes += [{"role": "assistant", "content": crudo},
                     {"role": "user", "content": "El plan tiene errores: " + "; ".join(errores) +
                      ". Corrige solo eso y devuelve el plan completo."}]
    raise ValueError("plan inválido tras 3 intentos: " + "; ".join(errores))


# ---------------------------------------------------------------- B6: anclaje numérico
NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def numeros_permitidos(pregunta, tabla, extra):
    vals = [float(v) for v in pd.to_numeric(tabla.stack(), errors="coerce").dropna()] if not tabla.empty else []
    texto = " ".join([pregunta, " ".join(map(str, tabla.index)), " ".join(map(str, tabla.columns)),
                      json.dumps(extra, default=str, ensure_ascii=False)])
    vals += [float(n.replace(",", ".")) for n in NUM.findall(texto)]
    return vals


def sin_anclaje(texto, permitidos):
    malos = []
    permitidos = [abs(b) for b in permitidos]
    for tok in NUM.findall(texto):
        a = abs(float(tok.replace(",", ".")))
        dec = len(tok.split(".")[1]) if "." in tok else (len(tok.split(",")[1]) if "," in tok else 0)
        if not any(abs(round(b, dec) - a) < 1e-9 or abs(b - a) < 1e-9 for b in permitidos):
            malos.append(tok)
    return malos


SISTEMA_RESPUESTA = """Redactas la respuesta final para una persona, en español, en 1 a 3 frases.
- Usa SOLO números que aparezcan en RESULTADO, copiados tal cual (punto decimal, sin separador de miles).
- No hagas cálculos nuevos ni inventes datos. Indica la unidad.
- Si RESULTADO está vacío, di claramente que no hay datos para ese periodo.
- Si hay HALLAZGOS, resúmelos sin omitir ninguno importante."""


def redactar(pregunta, tabla, meta, unidades, modelo, registro, hallazgos=None):
    if hallazgos is not None:
        # Calidad: el LLM solo ve hallazgos ya redactados por código, así no puede pegar una cifra a otra etiqueta.
        tabla = tabla.iloc[0:0]
        resultado = {"hallazgos": hallazgos, "meta": meta}
    else:
        resultado = {"tabla": json.loads(tabla.to_json(orient="index")) if not tabla.empty else {},
                     "unidades": unidades, "meta": meta}
    mensajes = [{"role": "system", "content": SISTEMA_RESPUESTA},
                {"role": "user", "content": f"PREGUNTA: {pregunta}\nRESULTADO: {json.dumps(resultado, ensure_ascii=False, default=str)}"}]
    permitidos = numeros_permitidos(pregunta, tabla, {"meta": meta, "hallazgos": hallazgos, "unidades": unidades})
    texto = llm(mensajes, modelo).strip()
    malos = sin_anclaje(texto, permitidos)
    registro["anclaje_inicial_ok"] = not malos
    if malos:
        registro["reparaciones_texto"] += 1
        mensajes += [{"role": "assistant", "content": texto},
                      {"role": "user", "content": f"Estos números no están en RESULTADO: {malos}. Reescribe usando solo números de RESULTADO."}]
        texto = llm(mensajes, modelo).strip()
        malos = sin_anclaje(texto, permitidos)
    if malos:  # respaldo determinista: nunca se entrega una cifra sin respaldo
        registro["respaldo_determinista"] = True
        texto = "No pude redactar la respuesta con cifras verificadas; este es el resultado calculado."
    registro["anclaje_final_ok"] = not sin_anclaje(texto, permitidos)
    return texto


# ---------------------------------------------------------------- redacción determinista de consultas
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
FRASE_AGG = {"promedio": "El promedio", "maximo": "El valor máximo", "minimo": "El valor mínimo",
             "suma": "El total", "conteo": "El número de lecturas", "ultimo": "El último valor"}


def fmt(v):
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{round(v, 3):g}"


def _fecha(ts, con_hora):
    base = f"{ts.day} de {MESES[ts.month - 1]} de {ts.year}"
    return f"{base} a las {ts:%H:%M}" if con_hora else base


def texto_periodo(plan):
    d, h = plan.get("desde"), plan.get("hasta")
    if not d and not h:
        return "en todo el periodo de datos"
    d = pd.Timestamp(d) if d else None
    h = pd.Timestamp(h) if h else None
    dias_completos = (d is None or d == d.normalize()) and (h is None or h == h.normalize())
    if d is not None and h is not None and dias_completos and h - d == pd.Timedelta(days=1):
        return f"el {_fecha(d, False)}"
    if dias_completos:
        ini = f"del {_fecha(d, False)}" if d is not None else "desde el inicio de los datos"
        fin = f" al {_fecha(h - pd.Timedelta(days=1), False)}" if h is not None else " hasta el final de los datos"
        return ini + fin
    ini = f"desde el {_fecha(d, True)}" if d is not None else "desde el inicio de los datos"
    return ini + (f" hasta el {_fecha(h, True)}" if h is not None else "")


def redactar_consulta(plan, tabla, meta, tj, meta_ds):
    """El código redacta: el LLM no puede contradecir ni desordenar las cifras."""
    col = plan["columnas"][0]
    t = tj[col]
    sig, uni = (t["significado"] or col).lower(), (t["unidad"] or "")
    periodo = texto_periodo(plan)
    if tabla.empty:
        texto = f"No hay datos de {sig} ({col}) {periodo}."
        if meta_ds.get("inicio"):
            texto += f" Los datos disponibles van del {meta_ds['inicio']} al {meta_ds['fin']}."
        return texto
    agg, por = plan["agregacion"], plan["por"]
    nota = " (sumando los incrementos del acumulado, no las lecturas)" if t.get("tipo") == "acumulado" else ""
    if agg == "conteo" and por == "total":
        texto = f"Hay {fmt(tabla.iloc[0, 0])} lecturas de {sig} ({col}) {periodo}."
    elif por == "total" or len(tabla) == 1:
        texto = (f"{FRASE_AGG.get(agg, 'El valor')} de {sig} ({col}) {periodo} fue {fmt(tabla.iloc[0, 0])} {uni}{nota}, "
                 f"calculado con {meta['lecturas']} lecturas.")
    else:
        s = tabla[col].dropna()
        unidad_t = "día" if por == "dia" else "hora"
        texto = (f"{FRASE_AGG.get(agg, 'El valor')} de {sig} ({col}) por {unidad_t}, {periodo}: {len(s)} filas en la tabla. "
                 f"El más alto fue {fmt(s.max())} {uni} ({s.idxmax()}) y el más bajo {fmt(s.min())} {uni} ({s.idxmin()}).")
    cad = meta_ds.get("cadencia_min")
    if cad and plan.get("desde") and plan.get("hasta") and agg != "conteo":
        esperadas = (pd.Timestamp(plan["hasta"]) - pd.Timestamp(plan["desde"])).total_seconds() / 60 / cad
        cobertura = meta["lecturas"] / len(plan["columnas"]) / esperadas
        if cobertura < 0.8:
            texto += f" Atención: en ese periodo solo hay el {cobertura:.0%} de las lecturas esperadas."
    return texto.replace("  ", " ").replace(" ,", ",")


# ---------------------------------------------------------------- flujo completo
def responder(pregunta, modelo=MODELO, confirmaciones=None, cache=None, historial=()):
    t0 = time.time()
    dfs, dics, metas = cache if cache else cargar_todo()
    conf = cargar_confirmaciones() if confirmaciones is None else confirmaciones
    tarjetas = {ds: construir(ds, dfs[ds], dics[ds], conf) for ds in dfs}
    reg = {"reparaciones_plan": 0, "reparaciones_texto": 0, "anclaje_inicial_ok": None, "anclaje_final_ok": None,
           "respaldo_determinista": False}
    salida = {"pregunta": pregunta, "modelo": modelo, "plan": None, "clase": None, "texto": "", "tabla": None, "registro": reg}
    try:
        plan = planificar(pregunta, dfs, tarjetas, metas, modelo, reg, historial)
    except Exception as e:  # noqa: BLE001
        salida.update(clase="error", texto=f"No pude interpretar la pregunta: {e}")
        return _cerrar(salida, t0)
    salida["plan"] = plan
    ds, df, tj = plan["dataset"], dfs[plan["dataset"]], tarjetas[plan["dataset"]]

    if plan["tipo"] == "preguntar":  # el propio LLM decidió que la petición es ambigua
        salida.update(clase="aclaracion", origen_aclaracion="llm", texto=plan["pregunta_al_usuario"].strip())
        return _cerrar(salida, t0)

    if plan["tipo"] in ("pronostico", "otro"):
        motivo = "pronósticos" if plan["tipo"] == "pronostico" else "este tipo de pregunta"
        salida.update(clase="fuera_de_alcance", texto=f"Todavía no puedo responder {motivo} con las herramientas disponibles; "
                                                       "no voy a estimarlo sin un modelo validado.")
        return _cerrar(salida, t0)

    if plan["tipo"] == "calidad":
        tabla, hallazgos = reporte_calidad(df, tj, metas[ds].get("cadencia_min"))
        salida.update(clase="calidad", tabla=tabla)
        salida["texto"] = redactar(pregunta, tabla, {"dataset": metas[ds]["texto"]}, {}, modelo, reg, hallazgos)
        salida["hallazgos"] = hallazgos
        return _cerrar(salida, t0)

    dudosas = [c for c in plan["columnas"] if tj[c]["estado"] != "confirmado"]
    if dudosas:  # B2: nunca se asume el significado
        salida.update(clase="aclaracion", origen_aclaracion="detector",
                      texto=pregunta_aclaracion(metas[ds]["texto"], dudosas, tj))
        return _cerrar(salida, t0)

    tabla, meta = ejecutar_consulta(plan, df, tj)
    salida.update(clase="sin_datos" if tabla.empty else "respuesta", tabla=tabla)
    salida["texto"] = redactar_consulta(plan, tabla, meta, tj, metas[ds])
    reg["anclaje_inicial_ok"] = reg["anclaje_final_ok"] = True  # cifras copiadas por código desde la tabla
    return _cerrar(salida, t0)


def _cerrar(salida, t0):
    salida["segundos"] = round(time.time() - t0, 1)
    return salida


def mostrar(s):
    print(s["texto"])
    t = s.get("tabla")
    if t is not None and not t.empty and (s["clase"] == "calidad" or len(t) > 1 or (s["plan"] or {}).get("formato") != "valor"):
        print()
        print(t.to_string())
    print(f"\n[{s['clase']} · {s['segundos']} s · plan: {json.dumps(s['plan'], ensure_ascii=False)}]")


def conversar(modelo):
    """Modo conversación: el agente recuerda sus aclaraciones y las respuestas del usuario."""
    cache, historial = cargar_todo(), []
    print(f"Kori (prototipo) con {modelo}. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("> ").strip()
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        if not pregunta:
            continue
        s = responder(pregunta, modelo, cache=cache, historial=historial)
        mostrar(s)
        print()
        historial += [("usuario", pregunta), ("agente", s["texto"])]
        historial = historial[-8:]  # contexto corto: los últimos 4 intercambios


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pregunta", nargs="?", help="sin pregunta entra en modo conversación")
    ap.add_argument("--modelo", default=MODELO)
    a = ap.parse_args()
    if a.pregunta:
        mostrar(responder(a.pregunta, a.modelo))
    else:
        conversar(a.modelo)
