# Interfaz de Kori (LibreChat)

Interfaz de chat para el agente de Kori. LibreChat (licencia MIT) se usa sin modificar su código: Kori se conecta
como un endpoint compatible con OpenAI.

```
Navegador ──► LibreChat (Docker, puerto 3080) ──► servidor_kori.py (puerto 8765) ──► agente.py ──► Ollama (11434)
                                             └──► Ollama directo (para comparar respuestas sin el agente)
```

## Archivos

| Archivo | Qué hace |
| --- | --- |
| `librechat.yaml` | Endpoints `Kori` y `Ollama`, modelos permitidos, mensaje de bienvenida |
| `docker-compose.override.yml` | Monta `librechat.yaml` en el contenedor |
| `../kori_agente/servidor_kori.py` | Expone el agente como API de OpenAI y guarda las confirmaciones de columnas |

LibreChat está clonado en `%USERPROFILE%\kori_librechat` (fuera de OneDrive para no sincronizar la base de datos).
Su `.env` tiene `APP_TITLE=Kori`, `ENDPOINTS=custom`, pie propio y claves generadas al instalar.

## Arranque

1. Ollama encendido (la app de Ollama en la bandeja).
2. Servidor de Kori:
   ```powershell
   & "$env:USERPROFILE\venvs\lidi\Scripts\python.exe" "kori_agente\servidor_kori.py"
   ```
3. Docker Desktop encendido, y en `%USERPROFILE%\kori_librechat`:
   ```powershell
   docker compose up -d
   ```
4. Abrir <http://localhost:3080>, crear una cuenta local (la primera es administradora) y elegir **Kori** en el menú de modelos.

Si se cambia `librechat.yaml`, copiarlo a `%USERPROFILE%\kori_librechat` y ejecutar `docker compose restart api`.

## Aclaraciones en el chat

Cuando una columna no tiene significado confirmado, Kori pregunta y termina el mensaje con
`Columnas por confirmar (dataset): ...`. La siguiente respuesta del usuario se interpreta como el significado de esas
columnas, se guarda en `kori_agente/tarjetas_confirmadas.json` y la pregunta original se responde de nuevo.
Lo guardado no se vuelve a preguntar.
