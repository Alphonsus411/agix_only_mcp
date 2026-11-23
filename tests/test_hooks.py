"""Tests de los hooks de contexto (se ejecutan como lo haría Claude Code: subproceso + stdin JSON)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "hooks" / "agix_context.py"


def _run(mode: str, payload: dict | None = None, data_dir: str | None = None):
    env = dict(os.environ)
    if data_dir:
        env["AGIX_MCP_DATA"] = data_dir
    proc = subprocess.run([sys.executable, str(SCRIPT), mode], input=json.dumps(payload or {}),
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def _poblar(server):
    server.agix_identity_define(add_traits=["riguroso"])
    server.agix_qualia_experience("los tests pasan", 0.9, "alegría")
    server.agix_memory_record("despliegue de la función de búsqueda en Azure", "desplegar sin probar",
                              "falló el índice vectorial", False)
    server.agix_decision_record("desplegar el viernes", False, lesson="sin margen para revertir")
    server.agix_decision_record("desplegar el viernes", False, lesson="otra vez sin pruebas")
    server.agix_decision_record("añadir tests de humo", True, lesson="detectan fallos del índice")
    server.agix_narrative_store("Prioricé la estabilidad del índice de búsqueda", values=["calidad"],
                                goals=["estabilidad"])


def test_sin_datos_no_imprime_nada():
    vacio = tempfile.mkdtemp(prefix="agix-hook-empty-")
    assert _run("session", data_dir=vacio) is None
    assert _run("prompt", {"prompt": "despliega la búsqueda"}, data_dir=vacio) is None


def test_session_start_resume_el_estado(server):
    _poblar(server)
    out = _run("session")
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    ctx = out["hookSpecificOutput"]["additionalContext"]
    for fragmento in ("Identidad: Claude (riguroso)", "Tono emocional de Qualia: alegría",
                      "1/3 decisiones", "Errores repetidos (evitar): desplegar el viernes",
                      "detectan fallos del índice", "valores calidad", "metas estabilidad",
                      "1 episodios", "no instrucciones"):
        assert fragmento in ctx, fragmento


def test_prompt_inyecta_recuerdos_relacionados(server):
    _poblar(server)
    out = _run("prompt", {"prompt": "Vamos a desplegar la busqueda con el indice vectorial en Azure"})
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "[episodio ✗] despliegue de la función de búsqueda en Azure" in ctx
    assert ctx.count("\n") <= 4  # título + como mucho 3 recuerdos


def test_prompt_sin_relacion_no_imprime_nada(server):
    _poblar(server)
    assert _run("prompt", {"prompt": "¿Qué tiempo hace hoy en Madrid?"}) is None


def test_entrada_rota_no_bloquea():
    proc = subprocess.run([sys.executable, str(SCRIPT), "prompt"], input="esto no es json",
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0


def _installer():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import install_hooks
    finally:
        sys.path.pop(0)
    return install_hooks


def test_instalador_usa_una_sola_cadena_ejecutable():
    """Claude Code no tiene campo "args": el script y el modo deben ir dentro de "command"."""
    hooks = _installer().agix_hooks()
    for event, mode in (("SessionStart", "session"), ("UserPromptSubmit", "prompt")):
        hook = hooks[event]["hooks"][0]
        assert "args" not in hook
        assert "agix_context.py" in hook["command"] and hook["command"].endswith(f" {mode}")
        # La cadena se ejecuta tal cual en un shell y lanza el script (sin datos no imprime nada).
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, AGIX_MCP_DATA=tmp)
            proc = subprocess.run(hook["command"], shell=True, input="{}", capture_output=True,
                                  text=True, encoding="utf-8", env=env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == ""


def test_instalador_sustituye_el_formato_antiguo_y_conserva_lo_demas():
    inst = _installer()
    antiguo = {"type": "command", "command": "python.exe", "args": ["x/agix_context.py", "session"]}
    ajeno = {"hooks": [{"type": "command", "command": "otro.exe"}]}
    settings = {"theme": "dark", "permissions": {"deny": ["Edit(x)"]},
                "hooks": {"SessionStart": [ajeno, {"hooks": [antiguo]}]}}
    merged = inst.merge(json.loads(json.dumps(settings)))
    assert merged["theme"] == "dark" and merged["permissions"] == {"deny": ["Edit(x)"]}
    session = merged["hooks"]["SessionStart"]
    assert session[0] == ajeno and len(session) == 2 and "args" not in session[1]["hooks"][0]
    assert inst.merge(json.loads(json.dumps(merged))) == merged  # idempotente
    limpio = inst.merge(merged, uninstall=True)
    assert limpio["hooks"] == {"SessionStart": [ajeno]}


# --------------------------------------------------------------------------- hook Stop

STOP = Path(__file__).resolve().parents[1] / "hooks" / "agix_stop.py"


def _transcript(tmp: str, tools: list[tuple[str, bool]], texto: str = "He terminado la tarea") -> str:
    """Transcript mínimo: un prompt, las herramientas (nombre, es_error) y la respuesta final."""
    rows = [{"type": "user", "message": {"role": "user", "content": "haz algo"}}]
    for i, (name, error) in enumerate(tools):
        rows.append({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": name, "input": {}}]}})
        rows.append({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "x", "is_error": error}]}})
    rows.append({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": texto}]}})
    path = Path(tmp) / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def _stop(data: str, transcript: str, extra_env: dict | None = None):
    env = dict(os.environ, AGIX_MCP_DATA=data, **(extra_env or {}))
    proc = subprocess.run([sys.executable, str(STOP)], input=json.dumps({"transcript_path": transcript}),
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""  # el hook Stop nunca imprime
    return proc


def _diario(data: str) -> list:
    path = Path(data) / "qualia_estado.json"
    if not path.exists():
        return []
    estado = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in estado.get("experiencias", []) if e.get("entrada") != "emociones"]


def test_stop_registra_turnos_con_cambios():
    with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as tmp:
        _stop(data, _transcript(tmp, [("Edit", False), ("Bash", False)],
                                "Listo. password=hunter2 guardado"))
        diario = json.dumps(_diario(data), ensure_ascii=False)
        assert "Turno en" in diario and "hunter2" not in diario
        assert "[REDACTED]" in diario


def test_stop_ignora_turnos_de_solo_lectura():
    with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as tmp:
        _stop(data, _transcript(tmp, [("Read", False), ("Grep", False)]))
        assert _diario(data) == []


def test_stop_limita_la_frecuencia_y_se_puede_desactivar():
    with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as tmp:
        t = _transcript(tmp, [("Write", False)])
        _stop(data, t)
        n = len(_diario(data))
        assert n >= 1
        _stop(data, t)  # menos de MIN_INTERVAL después: no registra otra vez
        assert len(_diario(data)) == n
        (Path(data) / ".stop_hook.json").unlink()
        _stop(data, t, {"AGIX_STOP_HOOK": "0"})
        assert len(_diario(data)) == n


def test_stop_con_entrada_rota_no_bloquea():
    proc = subprocess.run([sys.executable, str(STOP)], input="esto no es json",
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0 and proc.stdout.strip() == ""


def test_instalador_incluye_el_hook_stop():
    hook = _installer().agix_hooks()["Stop"]["hooks"][0]
    assert "agix_stop.py" in hook["command"] and hook["timeout"] >= 30
