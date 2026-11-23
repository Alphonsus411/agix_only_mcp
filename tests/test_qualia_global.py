"""Qualia global: un solo espíritu para todo el PC, compartido por los proyectos (memory_scope, server, hook)."""
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
import storage  # noqa: E402

HOOK = ROOT / "hooks" / "agix_context.py"


def _project(tmp_path, name):
    project = tmp_path / name
    (project / ".git").mkdir(parents=True, exist_ok=True)
    return project


def _env(tmp_path, **extra):
    return {"AGIX_MCP_HOME": str(tmp_path / "home"), **extra}


# --------------------------------------------------------------------------- ruta


def test_qualia_dir_es_global_para_cualquier_proyecto(tmp_path):
    env = _env(tmp_path)
    a = memory_scope.resolve(_project(tmp_path, "a"), env)
    b = memory_scope.resolve(_project(tmp_path, "b"), env)
    assert a.data_dir != b.data_dir
    assert memory_scope.qualia_dir(a, env) == memory_scope.qualia_dir(b, env) == tmp_path / "home" / "qualia"


def test_qualia_dir_con_marcador_global_o_de_proyecto(tmp_path):
    project = _project(tmp_path, "a")
    (project / ".agix.json").write_text(json.dumps({"memoria": "cliente-x"}), encoding="utf-8")
    scope = memory_scope.resolve(project, _env(tmp_path))
    assert memory_scope.qualia_dir(scope, _env(tmp_path)) == tmp_path / "home" / "qualia"


def test_qualia_dir_agix_mcp_data_la_deja_junto_a_los_datos(tmp_path):
    env = _env(tmp_path, AGIX_MCP_DATA=str(tmp_path / "fija"))
    scope = memory_scope.resolve(_project(tmp_path, "a"), env)
    assert memory_scope.qualia_dir(scope, env) == tmp_path / "fija"


def test_qualia_dir_sin_memoria_es_none(tmp_path):
    project = _project(tmp_path, "a")
    (project / ".agix.json").write_text(json.dumps({"memoria": "off"}), encoding="utf-8")
    scope = memory_scope.resolve(project, _env(tmp_path))
    assert memory_scope.qualia_dir(scope, _env(tmp_path)) is None


# --------------------------------------------------------------------------- bloqueo por carpeta


def test_bloqueo_anidado_de_dos_carpetas_bloquea_las_dos(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import storage\n"
        "try:\n"
        "    with storage.lock(Path(sys.argv[2]), timeout=0.3): print('libre')\n"
        "except storage.StorageBusy: print('ocupada')\n"
    )
    with storage.lock(a):
        with storage.lock(b):  # antes se daba por tomado sin bloquear b
            out = subprocess.run([sys.executable, "-c", code, str(ROOT), str(b)],
                                 capture_output=True, text=True, timeout=30)
    assert out.stdout.strip() == "ocupada", out.stderr


def test_bloqueo_reentrante_en_la_misma_carpeta(tmp_path):
    with storage.lock(tmp_path):
        with storage.lock(tmp_path):
            pass


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


def test_dos_proyectos_comparten_la_qualia_pero_no_la_memoria(server_in, tmp_path):
    srv_a = server_in(_project(tmp_path, "a"))
    srv_a.agix_qualia_experience("terminé un cambio en A", 0.9, "alegria")
    srv_a.agix_decision_record("accion de A", True, lesson="solo A")
    name_a = srv_a.SCOPE.name

    srv_b = server_in(_project(tmp_path, "b"))
    state = srv_b.agix_qualia_state()
    assert state["emotions"].get("alegria", 0) > 0.3  # B ve lo que vivió A
    assert srv_b.agix_memory_scope()["qualia"]["shared"] is True

    home = tmp_path / "home"
    assert (home / "qualia" / "qualia_estado.json").exists()
    for project_dir in (home / "proyectos").iterdir():
        assert not (project_dir / "qualia_estado.json").exists()
        assert not (project_dir / "qualia_modelos.json").exists()
    # la memoria sigue aislada
    assert srv_b.SCOPE.name != name_a
    assert not (home / "proyectos" / srv_b.SCOPE.name / "metacognicion.json").exists()


def test_una_sesion_ve_lo_que_otra_hace_despues_de_arrancar(server_in, tmp_path):
    srv_a = server_in(_project(tmp_path, "a"))
    srv_a.agix_qualia_state()  # A arranca con el estado vacío
    first = srv_a.agix_qualia_state()["emotions"].get("miedo", 0)

    srv_b = server_in(_project(tmp_path, "b"))
    srv_b.agix_qualia_experience("algo asusta en B", 0.8, "miedo")

    # A es el módulo ya cargado antes de que B escribiera: debe recargar la Qualia global.
    # (En la realidad cada servidor es un proceso; aquí se devuelve el cwd al de A, que B cambió.)
    os.chdir(srv_a.DATA_DIR)
    assert srv_a.agix_qualia_state()["emotions"].get("miedo", 0) > first


def test_primera_vez_parte_de_la_qualia_de_la_raiz_global(server_in, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    legacy = server_in(_project(tmp_path, "semilla"))  # crea un estado cualquiera
    legacy.agix_qualia_experience("vivido antes", 0.9, "curiosidad")
    saved = (home / "qualia" / "qualia_estado.json").read_text(encoding="utf-8")
    # se mueve a la raíz, como estaba antes de la Qualia global, y se borra la carpeta nueva
    for name in memory_scope.QUALIA_FILES:
        src = home / "qualia" / name
        if src.exists():
            src.replace(home / name)
    for p in (home / "qualia").iterdir():
        if p.is_file():
            p.unlink()
    (home / "qualia").rmdir()

    srv = server_in(_project(tmp_path, "nuevo"))
    assert (home / "qualia" / "qualia_estado.json").read_text(encoding="utf-8") == saved
    assert (home / "qualia_estado.json").exists()  # el original no se toca
    assert srv.agix_qualia_state()["emotions"].get("curiosidad", 0) > 0.3


def test_reset_de_qualia_afecta_a_todos_los_proyectos(server_in, tmp_path):
    srv_a = server_in(_project(tmp_path, "a"))
    srv_a.agix_qualia_experience("algo", 0.9, "alegria")
    srv_a.agix_qualia_configure(reset=True)
    srv_b = server_in(_project(tmp_path, "b"))
    assert srv_b.agix_qualia_state()["emotions"].get("alegria", 0) == 0


def test_con_memoria_desactivada_no_toca_la_qualia_global(server_in, tmp_path):
    project = _project(tmp_path, "a")
    (project / ".agix.json").write_text(json.dumps({"memoria": "off"}), encoding="utf-8")
    srv = server_in(project)
    srv.agix_qualia_experience("efímero", 0.9, "alegria")
    assert srv.agix_memory_scope()["qualia"] == {"shared": False, "dir": None}
    assert not (tmp_path / "home").exists()


# --------------------------------------------------------------------------- hook


def test_hook_lee_el_tono_de_la_qualia_global(server_in, tmp_path):
    project = _project(tmp_path, "a")
    srv = server_in(project)
    srv.agix_decision_record("accion de A", True, lesson="algo")
    srv.agix_qualia_experience("gran noticia", 0.95, "alegria")

    env = {k: v for k, v in os.environ.items() if k not in ("AGIX_MCP_DATA", "CLAUDE_PROJECT_DIR")}
    env["AGIX_MCP_HOME"] = str(tmp_path / "home")
    other = _project(tmp_path, "b")  # otro proyecto, con su propia memoria
    proc = subprocess.run([sys.executable, str(HOOK), "session"], input=json.dumps({"cwd": str(other)}),
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)
    assert proc.returncode == 0, proc.stderr
    # B no tiene memoria propia todavía, pero el tono de Qualia sí viene de la global
    text = proc.stdout and json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
    assert text and "alegria" in text.lower()
