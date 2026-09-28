# Kori — agente analista de datos de sensores

Prototipo del agente de **Kori** (LIDI, Universidad del Azuay): responde en español preguntas sobre datos tabulares
de sensores ambientales, con texto y tabla.

- **Las herramientas calculan, el LLM planifica y pregunta.** El modelo solo llena un plan JSON con opciones cerradas;
  todas las cifras las calcula código (pandas).
- **Nunca asume el significado de una columna.** Si una columna es dudosa (`t`, `h`, un acumulado que se reinicia),
  pregunta con hipótesis y evidencia, y guarda lo confirmado.
- **Todo corre en local** con [Ollama](https://ollama.com). Los datos no salen del equipo.

> Este repositorio contiene solo código. Los datos de la estación PSHA pertenecen al LIDI y no se incluyen;
> para probar sin ellos hay un dataset sintético (`kori_agente/datos_prueba/estacion_x.csv`).

## Requisitos

- Python 3.12
- [Ollama](https://ollama.com/download) y al menos el modelo recomendado:
  ```bash
  ollama pull qwen3:8b
  ```
- Para la interfaz de chat: [Docker Desktop](https://www.docker.com/products/docker-desktop/)

## Instalación

```bash
git clone <url-del-repositorio> kori
cd kori
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Probar el agente

Modo pregunta única o conversación, en la terminal:

```bash
cd kori_agente
python agente.py "¿Cuál es el promedio de v?"      # dataset sintético: preguntará qué es `v`
python agente.py                                  # modo conversación
```

Con el dataset sintético, las columnas `t`, `h`, `v` y `p` no tienen diccionario: el agente **debe preguntar** qué
significan antes de calcular. Así se ve la regla de aclaración funcionando.

### Con los datos de la estación (solo dentro del LIDI)

Copiar la exportación de `monitoring` en `datos/Datos LIDI/` (en la raíz del repositorio):
`node_register_n_psha01.csv` y `sensors.csv`. La fuente `psha` de `kori_agente/fuentes.json` la carga
automáticamente; si los archivos no están, se omite.

Para otro dataset, basta agregar una entrada en `fuentes.json` (formato `largo` o `ancho`); no hace falta tocar código.

## Evaluación (benchmark)

```bash
cd kori_agente
python generar_gold.py psha --semilla 1                   # genera preguntas con su respuesta calculada por código
python comparar_modelos.py qwen3:8b qwen3.5:9b             # mismo benchmark con varios modelos
python tabla_modelos.py                                    # tabla comparativa de todo lo evaluado
```

Métricas: resultado correcto, texto correcto, aclaración cuando corresponde, aclaración innecesaria y segundos por
pregunta. Los resultados se guardan en `kori_agente/resultados/` (no se versionan).

## Interfaz de chat (LibreChat)

Kori se expone como una API compatible con OpenAI y se usa desde [LibreChat](https://github.com/danny-avila/LibreChat)
(MIT), sin modificar su código.

```bash
# 1) Servidor de Kori (deja la terminal abierta)
python kori_agente/servidor_kori.py            # http://127.0.0.1:8765/v1

# 2) LibreChat (en otra carpeta)
git clone --depth 1 https://github.com/danny-avila/LibreChat.git kori_librechat
cd kori_librechat
cp .env.example .env
cp ../kori/kori_interfaz/librechat.yaml ../kori/kori_interfaz/docker-compose.override.yml .
docker compose up -d
```

Abrir <http://localhost:3080>, crear una cuenta local y elegir **Kori** en el menú de modelos.
Detalles de configuración (título, claves del `.env`, modelos permitidos): [`kori_interfaz/README.md`](kori_interfaz/README.md).

## Estructura

| Archivo | Qué hace |
| --- | --- |
| `kori_agente/fuentes.json`, `datos.py` | Adaptador de cualquier fuente a formato canónico (tiempo + mediciones + diccionario) |
| `kori_agente/tarjetas.py` | Tarjetas semánticas por columna y detector de ambigüedad |
| `kori_agente/herramientas.py` | Consultas y reporte de calidad deterministas |
| `kori_agente/agente.py` | Cadena del agente: plan → validación → verificación semántica → cálculo → redacción |
| `kori_agente/generar_gold.py`, `evaluar.py` | Conjunto de referencia generado por código y métricas |
| `kori_agente/comparar_modelos.py`, `tabla_modelos.py` | Comparación de modelos |
| `kori_agente/servidor_kori.py` | API compatible con OpenAI para interfaces de chat |
| `kori_interfaz/` | Configuración de LibreChat para Kori |

---
Uso académico en el LIDI. Sin licencia de uso abierta: todos los derechos reservados.
