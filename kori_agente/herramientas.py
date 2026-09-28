"""Herramientas deterministas: el código calcula, el LLM no."""
import pandas as pd

AGREGACIONES = {"promedio": "mean", "maximo": "max", "minimo": "min", "suma": "sum", "conteo": "count", "ultimo": "last"}


def serie_para_agregar(x, tarjeta):
    """Un acumulado con reinicios se convierte en incrementos antes de agregar."""
    x = x.dropna()
    if tarjeta.get("tipo") != "acumulado" or x.empty:
        return x
    inc = x.diff()
    reinicio = inc < 0
    inc[reinicio] = x[reinicio]
    inc.iloc[0] = 0.0
    return inc.clip(lower=0)


def ejecutar_consulta(plan, df, tarjetas):
    sub = df
    if isinstance(df.index, pd.DatetimeIndex):
        if plan.get("desde"):
            sub = sub[sub.index >= pd.Timestamp(plan["desde"])]
        if plan.get("hasta"):
            sub = sub[sub.index < pd.Timestamp(plan["hasta"])]
    cols = plan["columnas"]
    meta = {"lecturas": int(sub[cols].notna().sum().sum()) if len(sub) else 0}
    if meta["lecturas"] == 0:
        return pd.DataFrame(), meta
    series = {c: serie_para_agregar(sub[c], tarjetas[c]) for c in cols}
    agg, por = plan["agregacion"], plan["por"]
    if agg == "ninguna":
        tabla = sub[cols].dropna(how="all").head(200)
    elif por == "total" or not isinstance(df.index, pd.DatetimeIndex):
        tabla = pd.DataFrame({c: [getattr(s, AGREGACIONES[agg])()] for c, s in series.items()}, index=["total"])
    else:
        freq = "h" if por == "hora" else "D"
        tabla = pd.DataFrame({c: s.resample(freq).agg(AGREGACIONES[agg]) for c, s in series.items()})
        tabla = tabla.dropna(how="all")
        tabla.index = tabla.index.strftime("%Y-%m-%d %H:%M" if por == "hora" else "%Y-%m-%d")
    meta["desde_real"] = str(sub.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None
    meta["hasta_real"] = str(sub.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None
    return tabla.round(3), meta


def reporte_calidad(df, tarjetas, cadencia_min=None):
    """Perfil de calidad devuelto como tabla + hallazgos. La cadencia se calcula de los datos si no se da."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("el reporte de calidad necesita una columna de tiempo confirmada")
    if not cadencia_min:
        cadencia_min = df.index.to_series().diff().dt.total_seconds().median() / 60
    esperado = (df.index.max() - df.index.min()).total_seconds() / 60 / cadencia_min + 1
    filas = []
    for c in df.columns:
        x = df[c].dropna()
        gaps = x.index.to_series().diff().dt.total_seconds() / 60
        t = tarjetas[c]
        filas.append({"sensor": c, "cobertura_%": round(len(x) / esperado * 100, 1),
                      "hueco_max_h": round(gaps.max() / 60, 1), "repetidos_%": round((x.diff() == 0).mean() * 100, 1),
                      "estado_semantico": t["estado"]})
    tabla = pd.DataFrame(filas).set_index("sensor")
    ts = df.index.to_series().diff().dt.total_seconds() / 3600
    caidas = ts[ts > max(1, 6 * cadencia_min / 60)]  # caída = más de 1 h o de 6 lecturas seguidas perdidas
    hallazgos = [f"El nodo dejó de enviar datos {len(caidas)} veces por más de 1 h; la caída más larga fue de "
                 f"{caidas.max():.1f} h (terminó el {caidas.idxmax():%d/%m %H:%M})."] if len(caidas) else []
    for c, f in tabla.iterrows():
        if f["cobertura_%"] < 50:
            hallazgos.append(f"{c} ({tarjetas[c]['significado']}) solo tiene {f['cobertura_%']}% de cobertura.")
        if tarjetas[c]["estado"] == "requiere_confirmacion":
            hallazgos.append(f"{c}: {'; '.join(tarjetas[c]['motivos'])}.")
        elif f["repetidos_%"] > 60 and tarjetas[c]["tipo"] == "medicion":
            hallazgos.append(f"{c} ({tarjetas[c]['significado']}) repite el mismo valor en el {f['repetidos_%']}% "
                             f"de las lecturas (posible saturación o sensor pegado).")
    return tabla, hallazgos
