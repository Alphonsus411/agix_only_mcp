"""Varias sesiones a la vez, archivos corruptos y copias de seguridad (storage.py + server.py)."""
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import storage  # noqa: E402


def _python(code: str, data_dir: Path, **popen):
    env = dict(os.environ, AGIX_MCP_DATA=str(data_dir), PYTHONPATH=str(ROOT))
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)], env=env, cwd=str(ROOT),
                            stderr=subprocess.DEVNULL, text=True, **popen)


def _decisions(data_dir: Path) -> list[str]:
    data = json.loads((data_dir / "metacognicion.json").read_text(encoding="utf-8"))
    return [d["accion"] for d in data["decisions"]]


# --------------------------------------------------------------------------- concurrencia


def test_dos_sesiones_no_se_pisan(tmp_path):
    """Antes: A cargaba la memoria, B registraba, A registraba y borraba lo de B."""
    a = _python("""
        import sys, server
        print("listo", flush=True)
        sys.stdin.readline()
        server.agix_decision_record("decision A", True)
        print(server.agix_memory_search("episodio de B")["results"][0]["decision"], flush=True)
    """, tmp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert a.stdout.readline().strip() == "listo"  # A ya tiene la memoria cargada
    b = _python("""
        import server
        server.agix_decision_record("decision B", True)
        server.agix_memory_record("episodio de B", "decision de B", "bien", True)
    """, tmp_path)
    assert b.wait(timeout=120) == 0
    out, _ = a.communicate("seguir\n", timeout=120)
    assert a.returncode == 0
    assert sorted(_decisions(tmp_path)) == ["decision A", "decision B"]
    assert out.strip() == "decision de B"  # A ve lo que B registró sin reiniciarse


def test_registros_simultaneos_no_se_pierden_ni_comparten_id(tmp_path):
    code = """
        import sys, server
        tag = sys.argv[-1] if len(sys.argv) > 1 else "{tag}"
        for i in range(15):
            server.agix_decision_record(f"{tag}-{{i}}", True)
            server.agix_memory_record(f"{tag}-{{i}}", "d", "o", True)
    """
    procs = [_python(code.format(tag=t), tmp_path) for t in ("uno", "dos", "tres")]
    assert all(p.wait(timeout=240) == 0 for p in procs)
    decisions = json.loads((tmp_path / "metacognicion.json").read_text(encoding="utf-8"))["decisions"]
    episodes = json.loads((tmp_path / "memoria.json").read_text(encoding="utf-8"))["experiencias"]
    assert len(decisions) == 45 and len(episodes) == 45
    assert len({d["timestamp"] for d in decisions}) == 45
    assert len({e["timestamp"] for e in episodes}) == 45


def test_el_bloqueo_es_reentrante_y_excluye_a_otros_procesos(tmp_path):
    with storage.lock(tmp_path):
        with storage.lock(tmp_path):  # reentrante: no se bloquea a sí mismo
            pass
        other = _python(f"""
            import storage
            from pathlib import Path
            try:
                with storage.lock(Path(r"{tmp_path}"), timeout=0.5):
                    print("entró")
            except storage.StorageBusy:
                print("ocupado")
        """, tmp_path, stdout=subprocess.PIPE)
        out, _ = other.communicate(timeout=60)
    assert out.strip() == "ocupado"


# --------------------------------------------------------------------------- corrupción


def test_archivo_corrupto_se_aparta_y_no_se_sobrescribe(server, restart, tmp_path):
    data = Path(server.DATA_DIR)
    server.agix_decision_record("antes", True)
    storage.daily_backup(data)  # la copia del día, hecha cuando los datos estaban bien
    (data / "metacognicion.json").write_text('{"decisions": [{"accion": "ant', encoding="utf-8")
    (data / "memoria.json").write_text("no es json", encoding="utf-8")

    srv = restart(server)
    scope = srv.agix_memory_scope()
    assert len(scope["storage_problems"]) == 2
    quarantined = sorted(p.name.split(".corrupto-")[0] for p in data.glob("*.corrupto-*"))
    assert quarantined == ["memoria.json", "metacognicion.json"]
    srv.agix_decision_record("despues", True)
    assert _decisions(data) == ["despues"]
    # El contenido corrupto sigue intacto en su copia apartada, y la copia diaria tiene "antes".
    corrupt = next(data.glob("metacognicion.json.corrupto-*")).read_text(encoding="utf-8")
    assert corrupt == '{"decisions": [{"accion": "ant'
    backup = next((data / "backups").iterdir()) / "metacognicion.json"
    assert "antes" in backup.read_text(encoding="utf-8")


def test_la_copia_no_incluye_json_corruptos(tmp_path):
    (tmp_path / "bueno.json").write_text("{}", encoding="utf-8")
    (tmp_path / "roto.json").write_text("{", encoding="utf-8")
    today = storage.daily_backup(tmp_path)
    assert sorted(p.name for p in today.iterdir()) == ["bueno.json"]


def test_escritura_atomica_no_deja_temporales(tmp_path):
    storage.write_json(tmp_path / "x.json", {"a": 1})
    storage.write_json(tmp_path / "x.json", {"a": 2})
    assert json.loads((tmp_path / "x.json").read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


# --------------------------------------------------------------------------- copias


def test_copia_diaria_una_vez_al_dia_y_rotacion(tmp_path):
    (tmp_path / "memoria.json").write_text('{"experiencias": []}', encoding="utf-8")
    import sqlite3

    conn = sqlite3.connect(tmp_path / "autonarrativa.db")
    conn.execute("create table t (x)")
    conn.execute("insert into t values (1)")
    conn.commit()
    root = tmp_path / "backups"
    for day in range(1, 10):
        (root / f"2026-01-0{day}").mkdir(parents=True)
    today = storage.daily_backup(tmp_path, keep=7)
    conn.close()
    assert today is not None and (today / "memoria.json").exists()
    assert sqlite3.connect(today / "autonarrativa.db").execute("select x from t").fetchone() == (1,)
    assert storage.daily_backup(tmp_path, keep=7) is None  # ya hay copia de hoy
    days = sorted(d.name for d in root.iterdir())
    assert len(days) == 7 and days[-1] == today.name and "2026-01-01" not in days


def test_el_servidor_hace_la_copia_al_arrancar(server, restart):
    server.agix_decision_record("algo", True)
    srv = restart(server)
    assert srv.agix_memory_scope()["backups"] == [time.strftime("%Y-%m-%d", time.gmtime())]
