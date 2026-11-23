"""Añade los hooks de AGIX a un settings.json de Claude Code (por defecto ~/.claude/settings.json).

    python hooks/install_hooks.py            # instala
    python hooks/install_hooks.py --dry-run  # muestra el resultado sin escribir
    python hooks/install_hooks.py --uninstall

- Solo toca la clave "hooks": permisos, plugins y el resto se conservan tal cual.
- Hace copia de seguridad (settings.json.bak-agix) antes de escribir.
- Es idempotente: si los hooks de AGIX ya están, no los duplica.
- Las rutas apuntan a este repo y a su .venv, estén donde estén.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARKERS = ("agix_context.py", "agix_stop.py")


def agix_hooks() -> dict:
    python = (REPO / ".venv" / "Scripts" / "python.exe")
    if not python.exists():  # Linux / macOS
        python = REPO / ".venv" / "bin" / "python"
    if not python.is_file():
        # In CI and editable installs there may be no repository-local venv.
        # Use the interpreter running the installer rather than a missing executable.
        python = Path(sys.executable)
    context = REPO / "hooks" / "agix_context.py"
    stop = REPO / "hooks" / "agix_stop.py"

    def entry(script: Path, mode: str = "", timeout: int = 10) -> dict:
        # Claude Code solo admite una cadena en "command" (no hay campo "args"): todo va en ella,
        # con las rutas entre comillas por si contienen espacios.
        command = f'"{python.as_posix()}" "{script.as_posix()}"' + (f" {mode}" if mode else "")
        return {"hooks": [{"type": "command", "command": command, "timeout": timeout}]}

    # Stop importa el servidor (numpy + AGIX) para registrar la experiencia: necesita más margen.
    return {"SessionStart": entry(context, "session"), "UserPromptSubmit": entry(context, "prompt"),
            "Stop": entry(stop, timeout=30)}


def _is_agix(group: dict) -> bool:
    # También reconoce el formato antiguo con "args", para sustituirlo al reinstalar.
    return any(m in " ".join([h.get("command", "")] + list(h.get("args", [])))
               for h in group.get("hooks", []) for m in MARKERS)


def merge(settings: dict, uninstall: bool = False) -> dict:
    hooks = settings.setdefault("hooks", {})
    for event, group in agix_hooks().items():
        groups = [g for g in hooks.get(event, []) if not _is_agix(g)]
        if not uninstall:
            groups.append(group)
        if groups:
            hooks[event] = groups
        else:
            hooks.pop(event, None)
    if not hooks:
        settings.pop("hooks")
    return settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=str(Path.home() / ".claude" / "settings.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()

    path = Path(args.settings)
    original = path.read_text(encoding="utf-8") if path.exists() else "{}"
    settings = json.loads(original)  # si el JSON está roto, se aborta sin tocar nada
    result = json.dumps(merge(settings, args.uninstall), ensure_ascii=False, indent=2) + "\n"
    json.loads(result)

    if args.dry_run:
        sys.stdout.reconfigure(encoding="utf-8")
        print(result)
        return 0
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak-agix"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result, encoding="utf-8")
    action = "desinstalados" if args.uninstall else "instalados"
    print(f"Hooks de AGIX {action} en {path} (copia de seguridad: {path.name}.bak-agix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
