"""Expone el agente como una API compatible con OpenAI para usarlo desde LibreChat (u otra interfaz de chat).

Uso: python servidor_kori.py [--puerto 8765]
Endpoints: GET /v1/models · POST /v1/chat/completions (con o sin stream)

Aclaraciones: cuando el agente pregunta por columnas dudosas, la siguiente respuesta del usuario se interpreta
como su significado, se guarda en tarjetas_confirmadas.json y se vuelve a responder la pregunta original.
"""
import argparse
import json
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agente import OLLAMA, llm, responder
from datos import cargar_todo
from tarjetas import confirmar

CACHE = cargar_todo()
MARCA = "Columnas por confirmar"
MAX_FILAS = 60
EVALUADOS = {"qwen3:8b", "qwen3.5:9b", "qwen3:30b-a3b", "nemotron-3.5-lightning:30b"}  # modelos del benchmark


def modelos_ollama():
    with urllib.request.urlopen(OLLAMA.replace("/api/chat", "/api/tags"), timeout=10) as r:
        return sorted(m["name"] for m in json.loads(r.read())["models"] if m["name"] in EVALUADOS)


def texto_de(contenido):
    if isinstance(contenido, list):  # formato por partes: [{"type": "text", "text": ...}]
        return "".join(p.get("text", "") for p in contenido if isinstance(p, dict))
    return contenido or ""


def sin_pie(texto):
    """Quita tabla y pie que añadimos, para que el historial que ve el planificador sea solo el diálogo."""
    return texto.split("\n\n|")[0].split("\n\n---")[0].strip()


def tabla_md(df):
    df = df.reset_index() if df.index.name else df
    filas = df.head(MAX_FILAS).astype(object).where(df.head(MAX_FILAS).notna(), "")
    enc = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    cuerpo = "\n".join("| " + " | ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in f) + " |"
                       for f in filas.itertuples(index=False))
    extra = f"\n\n_Se muestran {MAX_FILAS} de {len(df)} filas._" if len(df) > MAX_FILAS else ""
    return f"{enc}\n{cuerpo}{extra}"


ESQUEMA_CONF = {"type": "object", "required": ["columnas"], "properties": {"columnas": {"type": "array", "items": {
    "type": "object", "required": ["columna", "significado", "unidad", "tipo"],
    "properties": {"columna": {"type": "string"}, "significado": {"type": "string"}, "unidad": {"type": "string"},
                   "tipo": {"type": "string", "enum": ["medicion", "acumulado", "tiempo", "calibracion", "falla"]}}}}}}


def guardar_confirmaciones(aclaracion, respuesta, modelo):
    """El LLM solo transcribe lo que dijo la persona a campos; no decide el significado."""
    m = re.search(rf"{MARCA} \((\w+)\): (.+)", aclaracion)
    dataset, columnas = m.group(1), re.findall(r"`([^`]+)`", m.group(2))
    sistema = ("Convierte la respuesta de la persona en un registro por columna. Usa SOLO lo que la persona dice; "
               "si acepta la hipótesis propuesta ('sí', 'correcto'), copia esa hipótesis. tipo: 'acumulado' si el valor "
               "se acumula y se reinicia; 'calibracion' si es un parámetro constante; 'falla' si el sensor no mide; "
               "'tiempo' si es tiempo; si no, 'medicion'. Si la persona no da unidad, usa ''.")
    crudo = llm([{"role": "system", "content": sistema},
                 {"role": "user", "content": f"Columnas: {columnas}\n\nPregunta del sistema:\n{aclaracion}\n\n"
                                             f"Respuesta de la persona:\n{respuesta}"}], modelo, ESQUEMA_CONF)
    guardadas = []
    for c in json.loads(crudo)["columnas"]:
        if c["columna"] in columnas:
            confirmar(dataset, c["columna"], c["significado"], c["unidad"], c["tipo"], "usuario de la interfaz")
            guardadas.append(f"`{c['columna']}` = {c['significado']}" + (f" ({c['unidad']})" if c["unidad"] else "") +
                             f", tipo {c['tipo']}")
    return guardadas


def contestar(mensajes, modelo):
    turnos = [(m["role"], texto_de(m.get("content"))) for m in mensajes if m.get("role") in ("user", "assistant")]
    pregunta, previos, aviso = turnos[-1][1], turnos[:-1], ""
    if len(previos) >= 2 and previos[-1][0] == "assistant" and MARCA in previos[-1][1]:
        guardadas = guardar_confirmaciones(previos[-1][1], pregunta, modelo)
        aviso = ("Guardé lo que me indicaste (no volveré a preguntarlo): " + "; ".join(guardadas) + ".\n\n") if guardadas \
            else "No pude identificar el significado de las columnas en tu respuesta.\n\n"
        pregunta, previos = previos[-2][1], previos[:-2]  # se vuelve a responder la pregunta original
    historial = [("usuario" if r == "user" else "agente", sin_pie(t)) for r, t in previos][-8:]
    s = responder(pregunta, modelo, cache=CACHE, historial=historial)

    partes = [aviso + s["texto"]]
    t, plan = s.get("tabla"), s["plan"] or {}
    if t is not None and not t.empty and (s["clase"] == "calidad" or len(t) > 1 or plan.get("formato") != "valor"):
        partes.append(tabla_md(t))
    if s["clase"] == "aclaracion" and s.get("origen_aclaracion") == "detector":
        dudosas = re.findall(r"^- `([^`]+)`", s["texto"], re.M)
        partes.append(f"---\n{MARCA} ({plan['dataset']}): " + ", ".join(f"`{c}`" for c in dudosas) +
                      "\n\n_Responde con el significado de cada una (o «sí» si la hipótesis es correcta)._")
    else:
        detalle = {k: v for k, v in plan.items() if v not in (None, "", []) and k != "pregunta_al_usuario"}
        partes.append(f"---\n<sub>Cómo lo calculé: {json.dumps(detalle, ensure_ascii=False)} · "
                      f"{s['segundos']} s · {modelo}</sub>")
    return "\n\n".join(partes)


class Manejador(BaseHTTPRequestHandler):
    def _json(self, codigo, obj):
        cuerpo = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "kori"}
                                                        for m in modelos_ollama()]})
        else:
            self._json(404, {"error": {"message": "ruta no encontrada"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": {"message": "ruta no encontrada"}})
        cuerpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        modelo = cuerpo.get("model")
        try:
            texto = contestar(cuerpo["messages"], modelo)
        except Exception as e:  # noqa: BLE001 — el error se muestra en el chat en vez de cortar la conexión
            texto = f"Ocurrió un error en el agente: {e}"
        base = {"id": f"kori-{int(time.time() * 1000)}", "created": int(time.time()), "model": modelo}
        if not cuerpo.get("stream"):
            return self._json(200, {**base, "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": texto}, "finish_reason": "stop"}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for delta, fin in (({"role": "assistant", "content": texto}, None), ({}, "stop")):
            trozo = {**base, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
            self.wfile.write(f"data: {json.dumps(trozo, ensure_ascii=False)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8765)
    a = ap.parse_args()
    print(f"Kori escuchando en http://127.0.0.1:{a.puerto}/v1")
    ThreadingHTTPServer(("127.0.0.1", a.puerto), Manejador).serve_forever()
