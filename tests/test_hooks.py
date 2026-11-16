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
