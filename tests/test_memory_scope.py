"""Memoria separada por proyecto (memory_scope.py, server.py y el hook)."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import memory_scope  # noqa: E402

HOOK = ROOT / "hooks" / "agix_context.py"


def _project(tmp_path, memoria=None, raw=None):
    project = tmp_path / "proyecto"
    (project / "src" / "sub").mkdir(parents=True)
    if raw is not None:
        (project / ".agix.json").write_text(raw, encoding="utf-8")
    elif memoria is not None:
        (project / ".agix.json").write_text(json.dumps({"memoria": memoria}), encoding="utf-8")
    return project


def _env(tmp_path, **extra):
    return {"AGIX_MCP_HOME": str(tmp_path / "home"), **extra}


# --------------------------------------------------------------------------- resolución


def test_sin_marcador_usa_la_global(tmp_path):
    scope = memory_scope.resolve(_project(tmp_path), _env(tmp_path))
    assert scope.name == "global" and scope.data_dir == tmp_path / "home"


def test_modo_estricto_sin_marcador_desactiva_memoria(tmp_path):
    env = _env(tmp_path, AGIX_MCP_REQUIRE_SCOPE="1")
    scope = memory_scope.resolve(_project(tmp_path), env)
    assert not scope.enabled
    assert scope.name == "off"
    assert scope.source == "AGIX_MCP_REQUIRE_SCOPE"


def test_modo_estricto_respeta_proyecto_explicito(tmp_path):
    env = _env(tmp_path, AGIX_MCP_REQUIRE_SCOPE="true")
    scope = memory_scope.resolve(_project(tmp_path, "cliente-x"), env)
    assert scope.enabled
    assert scope.name == "cliente-x"


def test_modo_estricto_respeta_off_y_global_explicitos(tmp_path):
    env = _env(tmp_path, AGIX_MCP_REQUIRE_SCOPE="yes")
    assert not memory_scope.resolve(_project(tmp_path, "off"), env).enabled
    assert memory_scope.resolve(_project(tmp_path, "global"), env).name == "global"


def test_marcador_con_nombre_desde_una_subcarpeta(tmp_path):
    project = _project(tmp_path, "cliente-x")
    scope = memory_scope.resolve(project / "src" / "sub", _env(tmp_path))
    assert scope.name == "cliente-x"
    assert scope.data_dir == tmp_path / "home" / "proyectos" / "cliente-x"
    assert scope.marker == project / ".agix.json"


@pytest.mark.parametrize("memoria, name", [("global", "global"), ("OFF", "off"), (" off ", "off")])
def test_valores_especiales(tmp_path, memoria, name):
    scope = memory_scope.resolve(_project(tmp_path, memoria), _env(tmp_path))
    assert scope.name == name
    assert scope.enabled == (name == "global") and scope.problem is None


@pytest.mark.parametrize("raw", [
    "esto no es json", "[]", '{"otra": 1}', '{"memoria": 3}',
    '{"memoria": "../../escapar"}', '{"memoria": "con espacios"}', '{"memoria": ""}',
])
def test_marcador_roto_desactiva_en_vez_de_mezclar(tmp_path, raw):
    scope = memory_scope.resolve(_project(tmp_path, raw=raw), _env(tmp_path))
    assert not scope.enabled and scope.problem


def test_agix_mcp_data_manda_sobre_el_marcador(tmp_path):
    env = _env(tmp_path, AGIX_MCP_DATA=str(tmp_path / "fija"))
    scope = memory_scope.resolve(_project(tmp_path, "cliente-x"), env)
    assert scope.data_dir == tmp_path / "fija"


# --------------------------------------------------------------------------- hook


def _hook(mode, cwd, env, prompt=""):
    full = {k: v for k, v in os.environ.items() if k not in ("AGIX_MCP_DATA", "CLAUDE_PROJECT_DIR")}
    full.update(env)
    proc = subprocess.run([sys.executable, str(HOOK), mode],
                          input=json.dumps({"cwd": str(cwd), "prompt": prompt}),
                          capture_output=True, text=True, encoding="utf-8", env=full, timeout=30)
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]


def _decisions(folder: Path, action: str):
    folder.mkdir(parents=True, exist_ok=True)
    entry = {"accion": action, "resultado": {"exito": False, "leccion": "lección de " + action},
             "contexto": {"goal": "despliegue"}}
    (folder / "metacognicion.json").write_text(json.dumps({"decisions": [entry]}), encoding="utf-8")


def test_hook_lee_solo_la_memoria_del_proyecto(tmp_path):
    env = _env(tmp_path)
    _decisions(tmp_path / "home", "secreto global desplegar")
    _decisions(tmp_path / "home" / "proyectos" / "cliente-x", "desplegar cliente equis")
    project = _project(tmp_path, "cliente-x")

    text = _hook("session", project / "src", env)
    assert "cliente-x" in text and "cliente equis" in text and "secreto global" not in text

    text = _hook("prompt", project, env, "vamos a desplegar lo del cliente equis")
    assert "cliente equis" in text and "secreto global" not in text

    other = tmp_path / "otro"
    other.mkdir()
    assert "secreto global" in _hook("session", other, env)


def test_hook_con_memoria_desactivada_no_lee_nada(tmp_path):
    env = _env(tmp_path)
    _decisions(tmp_path / "home", "secreto global desplegar")
    project = _project(tmp_path, "off")
    text = _hook("session", project, env)
    assert "desactivada" in text and "secreto" not in text
    assert _hook("prompt", project, env, "secreto global desplegar") is None


# --------------------------------------------------------------------------- servidor


@pytest.fixture()
def server_in(monkeypatch, tmp_path):
    cwd = os.getcwd()

    def load(project: Path):
        monkeypatch.delenv("AGIX_MCP_DATA", raising=False)
        monkeypatch.setenv("AGIX_MCP_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))
        monkeypatch.syspath_prepend(str(ROOT))
        sys.modules.pop("server", None)
        return importlib.import_module("server")

    yield load
    os.chdir(cwd)
    sys.modules.pop("server", None)


def test_servidor_guarda_en_la_memoria_del_proyecto(server_in, tmp_path):
    srv = server_in(_project(tmp_path, "cliente-x"))
    scope = srv.agix_memory_scope()
    assert scope["memory"] == "cliente-x" and scope["persistent"]
    srv.agix_decision_record("desplegar", True, lesson="bien")
    assert (tmp_path / "home" / "proyectos" / "cliente-x" / "metacognicion.json").exists()
    assert not (tmp_path / "home" / "metacognicion.json").exists()
    assert "cliente-x" in srv.INSTRUCTIONS


def test_servidor_con_memoria_desactivada_no_toca_ninguna_memoria(server_in, tmp_path):
    srv = server_in(_project(tmp_path, "off"))
    scope = srv.agix_memory_scope()
    assert scope["memory"] == "off" and not scope["persistent"] and scope["data_dir"] is None
    srv.agix_decision_record("desplegar", True, lesson="bien")
    assert not (tmp_path / "home").exists()
    assert "se pierde" in srv.INSTRUCTIONS
