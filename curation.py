"""Curación de la memoria de AGIX: retirada de lecciones y fusión de duplicados.

Nada se borra. Las marcas viven en curacion.json, junto a los datos, para que el formato de
AGIX (memoria.json, metacognicion.json, autonarrativa.db) no cambie:

    {"retired": {id: {"reason": str, "at": iso, "merged_into": id | null}},
     "lessons": {id: "lección consolidada o corregida"},
     "corrections": {id: [{"at", "from", "to", "reason"}]}}   # historial de cambios de lección

Identificadores: "e:<timestamp>" episodio, "d:<timestamp>" decisión, "n:<id>" autonarrativa.

- Una entrada retirada (obsoleta o fusionada) deja de mostrarse: búsquedas, lecciones de la
  revisión y hooks. Se puede restaurar.
- Una entrada fusionada en otra cuenta como repetición de esa otra (×n) y sigue contando para
  los errores repetidos; una retirada por obsoleta, no.
- La coherencia y las estadísticas por meta siguen usando todo el historial: lo que pasó, pasó.

Sin dependencias: lo usan server.py y los hooks.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

FILE = "curacion.json"
KINDS = {"e": "episodio", "d": "decisión", "n": "autonarrativa"}
_STOP = {
    "para", "pero", "como", "esto", "esta", "este", "estos", "estas", "porque", "cuando", "donde",
    "sobre", "entre", "desde", "hasta", "tambien", "todo", "todos", "cada", "solo", "with", "that",
    "this", "from", "have", "were", "will", "they", "their", "antes", "despues", "vez",
}


def empty() -> dict:
    return {"retired": {}, "lessons": {}, "corrections": {}}


def load(data_dir: Path) -> dict:
    path = Path(data_dir) / FILE
    if not path.exists():
        return empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty()
    cur = empty()
    if isinstance(data, dict):
        cur["retired"].update(data.get("retired") or {})
        cur["lessons"].update(data.get("lessons") or {})
        cur["corrections"].update(data.get("corrections") or {})
    return cur


def save(data_dir: Path, cur: dict) -> None:
    import storage  # escritura atómica con reintentos (Windows)

    storage.write_json(Path(data_dir) / FILE, cur)


# --------------------------------------------------------------------------- identificadores


def episode_id(timestamp) -> str:
    return f"e:{timestamp}"


def decision_id(timestamp) -> str:
    return f"d:{timestamp}"


def narrative_id(row_id) -> str:
    return f"n:{row_id}"


def kind(entry_id: str) -> str:
    return entry_id.split(":", 1)[0] if ":" in entry_id else ""


# --------------------------------------------------------------------------- consultas


def is_retired(cur: dict, entry_id: str) -> bool:
    return entry_id in cur["retired"]


def is_obsolete(cur: dict, entry_id: str) -> bool:
    """Retirada por obsoleta o errónea (no por duplicada)."""
    info = cur["retired"].get(entry_id)
    return info is not None and not info.get("merged_into")


def repeats(cur: dict, entry_id: str) -> int:
    """Cuenta fusiones transitivas sin recursión infinita ante ciclos o datos corruptos."""
    count = 0
    seen: set[str] = set()
    pending = [entry_id]
    retired = cur.get("retired", {})
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        count += 1
        pending.extend(
            item for item, info in retired.items()
            if isinstance(info, dict) and info.get("merged_into") == current
            and item not in seen
        )
    return count


def lesson(cur: dict, entry_id: str, original):
    return cur["lessons"].get(entry_id, original)


# --------------------------------------------------------------------------- cambios


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def retire(cur: dict, entry_id: str, reason: str, merged_into: str | None = None) -> None:
    cur["retired"][entry_id] = {"reason": reason, "at": _now(), "merged_into": merged_into}


def restore(cur: dict, entry_id: str) -> bool:
    return cur["retired"].pop(entry_id, None) is not None


def set_lesson(cur: dict, entry_id: str, text: str | None, reason: str | None = None) -> None:
    """Sustituye el texto de la lección (la original se conserva en los datos de AGIX).
    Con text=None vuelve a la original. Cada cambio queda anotado en "corrections"."""
    previous = cur["lessons"].get(entry_id)
    if text:
        cur["lessons"][entry_id] = text
    else:
        cur["lessons"].pop(entry_id, None)
    cur["corrections"].setdefault(entry_id, []).append(
        {"at": _now(), "from": previous, "to": text, "reason": reason})


# --------------------------------------------------------------------------- duplicados


def tokens(text: str) -> set[str]:
    nfkd = unicodedata.normalize("NFKD", (text or "").lower())
    plain = "".join(c for c in nfkd if not unicodedata.combining(c))
    return {t for t in re.findall(r"\w+", plain) if len(t) >= 4 and t not in _STOP}


def similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def duplicate_groups(items: list[dict], threshold: float) -> list[list[dict]]:
    """items: [{"id", "text", "success", "timestamp"}] de un mismo tipo y activos.

    Agrupa (por componentes conexas) los pares con similitud ≥ threshold y el mismo resultado:
    un fallo y un acierto con palabras parecidas cuentan cosas distintas y no se agrupan.
    """
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    toks = [tokens(it["text"]) for it in items]
    best: dict[int, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if items[i].get("success") != items[j].get("success") or not toks[i] or not toks[j]:
                continue
            sim = len(toks[i] & toks[j]) / len(toks[i] | toks[j])
            if sim >= threshold:
                parent[find(i)] = find(j)
                best[i] = max(best.get(i, 0.0), sim)
                best[j] = max(best.get(j, 0.0), sim)
    groups: dict[int, list[dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append({**items[i], "similarity": round(best.get(i, 0.0), 3)})
    return [sorted(g, key=lambda it: str(it.get("timestamp") or "")) for g in groups.values() if len(g) > 1]
