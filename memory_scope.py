"""Qué memoria de AGIX usa cada proyecto (compartido por server.py y los hooks; sin dependencias).

Orden de resolución:
1. AGIX_MCP_DATA: si está definida, esa carpeta y nada más (uso explícito y tests).
2. Un archivo `.agix.json` en la carpeta del proyecto o en cualquiera de sus ancestros:
       {"memoria": "cliente-x"}  -> memoria propia en ~/.agix-mcp/proyectos/cliente-x
       {"memoria": "global"}     -> la memoria global (~/.agix-mcp)
       {"memoria": "off"}        -> sin memoria: ni se lee ni se guarda nada
   Si el archivo está roto o el nombre no es válido, la memoria se desactiva: ante la duda,
   es mejor no guardar nada que mezclar proyectos.
3. Sin archivo: la memoria global (~/.agix-mcp), como hasta ahora.

La carpeta base (~/.agix-mcp) se puede cambiar con AGIX_MCP_HOME.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

MARKER = ".agix.json"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class Scope:
    name: str                 # "global", "off", "AGIX_MCP_DATA" o el nombre del proyecto
    data_dir: Path | None     # None si la memoria está desactivada
    source: str               # de dónde sale la decisión
    marker: Path | None = None
    problem: str | None = None

    @property
    def enabled(self) -> bool:
        return self.data_dir is not None

    def describe(self) -> str:
        if not self.enabled:
            why = f" ({self.problem})" if self.problem else ""
            return f"memoria de AGIX desactivada en este proyecto{why}"
        if self.name == "global":
            return "memoria global de AGIX"
        return f"memoria de AGIX del proyecto «{self.name}»"


def base_dir(env: dict | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("AGIX_MCP_HOME") or Path.home() / ".agix-mcp")


def find_marker(start: Path) -> Path | None:
    try:
        start = start.resolve()
    except OSError:
        return None
    for folder in (start, *start.parents):
        candidate = folder / MARKER
        if candidate.is_file():
            return candidate
    return None


def resolve(start: str | Path | None = None, env: dict | None = None) -> Scope:
    env = os.environ if env is None else env
    if env.get("AGIX_MCP_DATA"):
        return Scope("AGIX_MCP_DATA", Path(env["AGIX_MCP_DATA"]), "variable AGIX_MCP_DATA")

    base = base_dir(env)
    start = Path(start or env.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    marker = find_marker(start)
    if marker is None:
        return Scope("global", base, "sin .agix.json: memoria global")

    def off(problem: str | None) -> Scope:
        return Scope("off", None, str(marker), marker, problem)

    try:
        config = json.loads(marker.read_text(encoding="utf-8"))
        value = config["memoria"]
    except (OSError, ValueError, TypeError, KeyError):
        return off(f'{marker} no es un JSON con la clave "memoria"')
    if not isinstance(value, str):
        return off(f'"memoria" debe ser un texto en {marker}')
    value = value.strip().lower()
    if value == "off":
        return off(None)
    if value == "global":
        return Scope("global", base, str(marker), marker)
    if not NAME_RE.fullmatch(value):
        return off(f"nombre de memoria no válido en {marker}: usa minúsculas, números, - y _")
    return Scope(value, base / "proyectos" / value, str(marker), marker)
