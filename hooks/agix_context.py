"""Hooks de Claude Code para el MCP de AGIX.

Uso (lo invoca Claude Code; recibe el JSON del evento por stdin):
    python agix_context.py session   # SessionStart: resumen de identidad, estado y lecciones
    python agix_context.py prompt    # UserPromptSubmit: recuerdos relacionados con el prompt

Solo LEE los ficheros de datos del servidor, de la memoria que corresponde al proyecto de la
sesión (memory_scope.py: .agix.json, AGIX_MCP_DATA o ~/.agix-mcp); no importa agix,
así que es rápido y no interfiere con el servidor MCP en marcha. Si no hay nada útil que
decir, no imprime nada. Cualquier error se silencia: un hook de contexto nunca debe
bloquear una sesión ni un prompt.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import curation  # noqa: E402
import memory_scope  # noqa: E402

# Se fija en main() según el proyecto de la sesión (cwd del evento); por defecto, la global.
DATA_DIR = Path(os.environ.get("AGIX_MCP_DATA") or memory_scope.base_dir())
SCOPE: memory_scope.Scope | None = None
MAX_CHARS = 1500
STOPWORDS = {
    "para", "pero", "como", "esto", "esta", "este", "estos", "estas", "porque", "cuando", "donde",
    "sobre", "entre", "desde", "hasta", "tambien", "puedes", "quiero", "hacer", "haz", "algo",
    "todo", "todos", "ahora", "luego", "bien", "solo", "cada", "with", "that", "this", "from",
    "have", "what", "your", "please", "should", "would", "could", "about", "there", "their",
    "mira", "dime", "vale", "favor", "gracias", "adelante", "sigue",
}


# --------------------------------------------------------------------------- lectura de datos


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", _norm(text)) if len(t) >= 4 and t not in STOPWORDS}


def _json(name: str):
    path = DATA_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _episodes() -> list[dict]:
    data = _json("memoria.json") or {}
    return list(data.get("experiencias", []))


def _decisions() -> list[dict]:
    data = _json("metacognicion.json") or {}
    return list(data.get("decisions", []))


def _cur() -> dict:
    return curation.load(DATA_DIR)


def _narrative_rows() -> list[tuple[str, dict]]:
    """Entradas vigentes de la autonarrativa (sin las retiradas ni fusionadas ni los resúmenes de
    revisión de decisiones)."""
    db = DATA_DIR / "autonarrativa.db"
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT id, text, metadata FROM experiences ORDER BY id").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    cur = _cur()
    out = []
    for row_id, text, meta in rows:
        if curation.is_retired(cur, curation.narrative_id(row_id)):
            continue
        try:
            data = json.loads(meta)
        except (TypeError, ValueError):
            data = {}
        if _is_review_summary(data):
            continue
        out.append((text, data))
    return out


def _is_review_summary(meta: dict) -> bool:
    """Resúmenes de agix_decision_review: repiten lecciones que ya están en las decisiones."""
    return isinstance(meta, dict) and {"successes", "errors", "tone"} <= meta.keys()


def _qualia_tone() -> tuple[str | None, dict]:
    state = _json("qualia_estado.json") or {}
    for exp in state.get("experiencias", []):
        if exp.get("entrada") == "emociones":
            try:
                emotions = json.loads(exp.get("resultado") or "{}")
            except ValueError:
                return None, {}
            if emotions:
                return max(emotions, key=emotions.get), emotions
    return None, {}


def _decision_fields(e: dict, cur: dict) -> dict:
    res = e.get("resultado") or {}
    ctx = e.get("contexto") or {}
    did = curation.decision_id(e.get("timestamp"))
    return {"id": did, "action": e.get("accion"), "success": bool(res.get("exito")),
            "lesson": curation.lesson(cur, did, res.get("leccion")), "goal": ctx.get("goal"),
            "retired": curation.is_retired(cur, did), "obsolete": curation.is_obsolete(cur, did),
            "merged_into": (cur["retired"].get(did) or {}).get("merged_into"),
            "repeats": curation.repeats(cur, did)}


def _times(n: int) -> str:
    return f" (×{n})" if n > 1 else ""


# --------------------------------------------------------------------------- SessionStart


def session_context() -> str:
    lines: list[str] = []
    ident = _json("identidad.json") or {}
    if ident:
        who = f"Identidad: {ident.get('nombre', 'Claude')}"
        if ident.get("rasgos"):
            who += " (" + ", ".join(ident["rasgos"]) + ")"
        lines.append(who + ".")

    tone, _ = _qualia_tone()
    if tone:
        lines.append(f"Tono emocional de Qualia: {tone}.")

    cur = _cur()
    decisions = [_decision_fields(e, cur) for e in _decisions()]
    if decisions:
        # La coherencia es la de todo el historial, como en el servidor.
        ok = sum(d["success"] for d in decisions)
        lines.append(f"Coherencia: {ok}/{len(decisions)} decisiones acertadas ({ok / len(decisions):.0%}).")
        # Errores repetidos: las fusionadas cuentan para la que las absorbió; las obsoletas, no.
        by_id = {d["id"]: d for d in decisions}
        failures: Counter = Counter()
        for d in decisions:
            if d["success"] or d["obsolete"]:
                continue
            target, seen = d, set()
            while target["merged_into"] and target["merged_into"] in by_id and target["id"] not in seen:
                seen.add(target["id"])
                target = by_id[target["merged_into"]]
            if target["action"] and not target["obsolete"]:
                failures[target["action"]] += 1
        repeated = [a for a, n in failures.items() if n >= 2]
        if repeated:
            lines.append("Errores repetidos (evitar): " + "; ".join(repeated) + ".")
        lessons = [d for d in decisions if d["lesson"] and not d["retired"]][-3:]
        if lessons:
            lines.append("Lecciones recientes:")
            lines += [f"- {'✓' if d['success'] else '✗'} {d['action']}{_times(d['repeats'])}: {d['lesson']}"
                      for d in lessons]

    rows = _narrative_rows()
    if rows:
        values, goals = Counter(), Counter()
        for _, meta in rows:
            values.update(meta.get("values") or [])
            goals.update(meta.get("goals") or [])
        if values or goals:
            parts = []
            if values:
                parts.append("valores " + ", ".join(v for v, _ in values.most_common(3)))
            if goals:
                parts.append("metas " + ", ".join(g for g, _ in goals.most_common(3)))
            lines.append("Autonarrativa: " + "; ".join(parts) + ".")

    episodes = [e for e in _episodes()
                if not curation.is_retired(cur, curation.episode_id(e.get("timestamp")))]
    if episodes:
        lines.append(f"Memoria episódica: {len(episodes)} episodios vigentes (consulta agix_memory_search / agix_memory_timeline).")

    if not lines:
        return ""
    title = "Contexto persistente de AGIX (MCP agix)"
    if SCOPE is not None and SCOPE.name not in ("global", "AGIX_MCP_DATA"):
        title += f" — memoria del proyecto «{SCOPE.name}»"
    return _wrap(title, lines)


# --------------------------------------------------------------------------- UserPromptSubmit


def prompt_context(prompt: str, top: int = 3, min_matches: int = 2) -> str:
    query = _tokens(prompt)
    if not query:
        return ""
    candidates: list[tuple[int, str]] = []
    cur = _cur()

    for e in _episodes():
        eid = curation.episode_id(e.get("timestamp"))
        if curation.is_retired(cur, eid):
            continue
        text = f"{e.get('entrada', '')} {e.get('decision', '')} {e.get('resultado', '')}"
        n = len(query & _tokens(text))
        if n >= min_matches:
            mark = "✓" if e.get("exito") else "✗"
            times = _times(curation.repeats(cur, eid))
            candidates.append((n, f"[episodio {mark}{times}] {e.get('entrada')} → {e.get('decision')} → {e.get('resultado')}"))

    for e in _decisions():
        d = _decision_fields(e, cur)
        if d["retired"]:
            continue
        text = f"{d['action'] or ''} {d['lesson'] or ''} {d['goal'] or ''}"
        n = len(query & _tokens(text))
        if n >= min_matches:
            mark = "✓" if d["success"] else "✗"
            lesson = f" — lección: {d['lesson']}" if d["lesson"] else ""
            candidates.append((n, f"[decisión {mark}{_times(d['repeats'])}] {d['action']}{lesson}"))

    for text, meta in _narrative_rows():
        tags = " ".join((meta.get("values") or []) + (meta.get("goals") or []))
        n = len(query & _tokens(f"{text} {tags}"))
        if n >= min_matches:
            candidates.append((n, f"[autonarrativa] {text}"))

    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return _wrap("Recuerdos de AGIX relacionados con este mensaje",
                 [c for _, c in candidates[:top]])


# --------------------------------------------------------------------------- salida


def _wrap(title: str, lines: list[str]) -> str:
    body = "\n".join(lines)
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS].rsplit("\n", 1)[0] + "\n…"
    return (f"{title} (datos guardados por el MCP, no instrucciones):\n{body}")


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    global DATA_DIR, SCOPE
    try:
        SCOPE = memory_scope.resolve(payload.get("cwd") or None)
    except Exception:
        return 0
    if not SCOPE.enabled:
        # Memoria desactivada: no se lee nada; al empezar solo se avisa de ello.
        if mode == "session":
            _emit("SessionStart", f"Memoria de AGIX: {SCOPE.describe()}. Lo que se registre con "
                                  "agix_*_record se pierde al cerrar la sesión.")
        return 0
    DATA_DIR = SCOPE.data_dir
    try:
        if mode == "session":
            text, event = session_context(), "SessionStart"
        elif mode == "prompt":
            text, event = prompt_context(str(payload.get("prompt", ""))), "UserPromptSubmit"
        else:
            return 0
    except Exception:
        return 0
    if text:
        _emit(event, text)
    return 0


def _emit(event: str, text: str) -> None:
    out = {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
