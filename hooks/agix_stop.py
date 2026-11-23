"""Hook Stop de Claude Code: registra en la Qualia global lo que ocurrió en el turno.

Uso (lo invoca Claude Code; recibe el JSON del evento por stdin):
    python agix_stop.py

A diferencia de agix_context.py, este hook ESCRIBE: vive la experiencia con la misma función que
la herramienta agix_qualia_experience del servidor (mismo bloqueo de la carpeta de Qualia y misma
sincronización con las demás sesiones), así que no puede pisar al servidor MCP en marcha.

Para no llenar el diario de ruido solo registra turnos con consecuencias:
- el turno ejecutó herramientas que modifican algo (editar, escribir, comandos, parches), y
- han pasado al menos MIN_INTERVAL segundos desde la última experiencia registrada.
La emoción sale de un criterio simple: errores en las herramientas → miedo; si no, alegría.

Nunca bloquea ni imprime nada: cualquier error se silencia (un hook no debe romper la sesión).
Se desactiva con AGIX_STOP_HOOK=0.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MIN_INTERVAL = 60.0          # segundos entre dos experiencias registradas por el hook
THROTTLE_FILE = ".stop_hook.json"
SUMMARY_CHARS = 160
MUTATING = {"Edit", "Write", "NotebookEdit", "Bash", "PowerShell"}
MUTATING_SUFFIXES = ("apply_patch", "create_new_file", "execute_terminal_command", "edit_notebook",
                     "rename_refactoring", "reformat_file")


def _blocks(entry: dict) -> list[dict]:
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def _is_real_prompt(entry: dict) -> bool:
    """Mensaje del usuario (no un resultado de herramienta)."""
    if entry.get("type") != "user" or entry.get("isMeta"):
        return False
    blocks = _blocks(entry)
    return bool(blocks) and not any(b.get("type") == "tool_result" for b in blocks)


def _is_mutating(name: str) -> bool:
    return name in MUTATING or name.endswith(MUTATING_SUFFIXES)


def analyze_turn(transcript: Path) -> dict:
    """Resumen del último turno: herramientas que modifican, errores y último texto del asistente."""
    entries = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    start = 0
    for i, entry in enumerate(entries):
        if _is_real_prompt(entry):
            start = i + 1
    mutating = errors = 0
    last_text = ""
    for entry in entries[start:]:
        for block in _blocks(entry):
            kind = block.get("type")
            if kind == "tool_use" and _is_mutating(str(block.get("name", ""))):
                mutating += 1
            elif kind == "tool_result" and block.get("is_error"):
                errors += 1
            elif kind == "text" and entry.get("type") == "assistant" and block.get("text", "").strip():
                last_text = block["text"].strip()
    return {"mutating": mutating, "errors": errors, "text": last_text}


def build_experience(project: str, turn: dict) -> tuple[str, float, str]:
    import agix_context  # solo para reutilizar la redacción de credenciales

    summary = " ".join(turn["text"].split())[:SUMMARY_CHARS] or "sin resumen"
    event = agix_context._redact_context(f"Turno en {project}: {summary}")
    emotion = "miedo" if turn["errors"] else "alegría"
    charge = min(0.2 + 0.05 * turn["mutating"], 0.5)
    return event, round(charge, 2), emotion


def _throttled(qualia_dir: Path, now: float) -> bool:
    path = qualia_dir / THROTTLE_FILE
    try:
        last = float(json.loads(path.read_text(encoding="utf-8")).get("last", 0))
    except Exception:
        last = 0.0
    return now - last < MIN_INTERVAL


def _mark(qualia_dir: Path, now: float) -> None:
    (qualia_dir / THROTTLE_FILE).write_text(json.dumps({"last": now}), encoding="utf-8")


def main() -> int:
    if os.environ.get("AGIX_STOP_HOOK") == "0":
        return 0
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            return 0
        import memory_scope

        scope = memory_scope.resolve(payload.get("cwd") or None)
        qdir = memory_scope.qualia_dir(scope)
        transcript = payload.get("transcript_path")
        if qdir is None or not transcript or not Path(transcript).is_file():
            return 0
        turn = analyze_turn(Path(transcript))
        now = time.time()
        if not turn["mutating"] or _throttled(qdir, now):
            return 0
        event, charge, emotion = build_experience(scope.name, turn)
        # Importar el servidor es lo caro (numpy + AGIX); se hace solo si hay algo que registrar.
        with contextlib.redirect_stdout(io.StringIO()):
            import server

            server.agix_qualia_experience(event, charge, emotion)
        _mark(qdir, now)
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
