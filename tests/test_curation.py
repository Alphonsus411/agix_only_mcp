"""Retirada de lecciones y fusión de duplicados (curation.py, server.py y el hook)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "hooks" / "agix_context.py"


def _hook(mode, data_dir, prompt=""):
    env = dict(os.environ, AGIX_MCP_DATA=str(data_dir))
    proc = subprocess.run([sys.executable, str(HOOK), mode], input=json.dumps({"prompt": prompt}),
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"] if proc.stdout.strip() else ""


def _ids(server, action):
    return [r["id"] for r in server.agix_metacognition_state(recent=100)["recent"] if r["action"] == action]


# --------------------------------------------------------------------------- retirada


def test_retirar_una_leccion_la_oculta_sin_borrarla(server):
    server.agix_decision_record("corregir plasticidad en el venv", True,
                                lesson="la plasticidad de qualia anula las emociones")
    server.agix_decision_record("usar git commit -F", True, lesson="evita el problema de comillas")
    did = _ids(server, "corregir plasticidad en el venv")[0]

    out = server.agix_memory_retire([did], reason="corregido en 4fab2eb")
    assert out["changed"] == [did]

    review = server.agix_decision_review(store_in_narrative=False)
    assert [r["action"] for r in review["recorded_lessons"]] == ["usar git commit -F"]
    assert "plasticidad" not in json.dumps(review["lessons"], ensure_ascii=False)
    # Sigue en el historial y en la coherencia.
    state = server.agix_metacognition_state(recent=10)
    assert state["total_decisions"] == 2
    assert [r for r in state["recent"] if r["id"] == did][0]["retired"]["reason"] == "corregido en 4fab2eb"

    text = _hook("session", server.DATA_DIR)
    assert "plasticidad" not in text and "git commit -F" in text and "2/2" in text

    server.agix_memory_retire([did], restore=True)
    review = server.agix_decision_review(store_in_narrative=False)
    assert len(review["recorded_lessons"]) == 2


def test_retirar_exige_motivo_e_ids_conocidos(server):
    server.agix_decision_record("algo", True, lesson="x")
    did = _ids(server, "algo")[0]
    with pytest.raises(ValueError, match="motivo"):
        server.agix_memory_retire([did])
    with pytest.raises(ValueError, match="desconocidos"):
        server.agix_memory_retire(["d:no-existe"], reason="x")


def test_episodios_retirados_no_salen_en_busquedas_ni_en_el_hook(server):
    server.agix_memory_record("despliegue del indice vectorial", "desplegar sin probar",
                              "fallo el indice vectorial", False)
    eid = server.agix_memory_recent()["recent"][0]["id"]
    server.agix_memory_retire([eid], reason="el indice ya no existe")
    assert server.agix_memory_search("indice vectorial")["results"] == []
    assert server.agix_memory_recent()["recent"] == []
    assert server.agix_memory_timeline("2000-01-01")["count"] == 0
    assert server.agix_memory_timeline("2000-01-01", include_retired=True)["count"] == 1
    assert "vectorial" not in _hook("prompt", server.DATA_DIR, "el indice vectorial del despliegue")


def test_retirar_entrada_de_la_autonarrativa(server):
    nid = server.agix_narrative_store("Aprendí que la plasticidad anula emociones", values=["rigor"])["id"]
    server.agix_narrative_store("Prioricé la estabilidad del despliegue", values=["calidad"])
    server.agix_memory_retire([f"n:{nid}"], reason="ya no es cierto")
    texts = [r["text"] for r in server.agix_narrative_query("plasticidad emociones", top=5)["results"]]
    assert all("plasticidad" not in t for t in texts)
    assert "plasticidad" not in _hook("prompt", server.DATA_DIR, "la plasticidad anula emociones")


# --------------------------------------------------------------------------- duplicados


def test_detecta_duplicados_solo_con_el_mismo_resultado(server):
    server.agix_decision_record("git commit -m con comillas en PowerShell", False,
                                lesson="PowerShell parte los argumentos con comillas; usar git commit -F")
    server.agix_decision_record("Usar git commit -F con el mensaje en un archivo", True,
                                lesson="evita el problema de comillas de PowerShell")
    server.agix_decision_record("Usar git commit -F para el commit del instalador", True,
                                lesson="evita el problema de comillas de PowerShell otra vez")
    server.agix_decision_record("añadir tests de humo", True, lesson="detectan fallos del indice")
    groups = server.agix_memory_duplicates(threshold=0.5, kinds=["d"])["groups"]
    assert len(groups) == 1
    actions = {e["text"].split(":")[0] for e in groups[0]["entries"]}
    assert actions == {"Usar git commit -F con el mensaje en un archivo",
                       "Usar git commit -F para el commit del instalador"}
    assert all(e["success"] for e in groups[0]["entries"])
    assert groups[0]["suggested_keep"] == groups[0]["entries"][-1]["id"]


def test_fusionar_cuenta_repeticiones_y_consolida_la_leccion(server):
    for i in range(3):
        server.agix_decision_record(f"Usar git commit -F caso {i}", True,
                                    lesson="evita el problema de comillas de PowerShell")
    ids = [r["id"] for r in server.agix_metacognition_state()["recent"]]
    out = server.agix_memory_merge(ids[-1], ids[:-1], lesson="En PowerShell 5.1, git commit -F siempre")
    assert out["repeats"] == 3

    review = server.agix_decision_review(store_in_narrative=False)
    assert len(review["recorded_lessons"]) == 1
    assert review["recorded_lessons"][0]["lesson"] == "En PowerShell 5.1, git commit -F siempre"
    assert review["recorded_lessons"][0]["repeats"] == 3
    assert server.agix_metacognition_state()["total_decisions"] == 3  # nada se borra

    text = _hook("session", server.DATA_DIR)
    assert "(×3)" in text and "siempre" in text
    assert server.agix_memory_duplicates(kinds=["d"])["groups"] == []

    server.agix_memory_retire([ids[0]], restore=True)  # deshacer una fusión
    assert server.agix_decision_review(store_in_narrative=False)["recorded_lessons"][-1]["repeats"] == 2


def test_fusionados_siguen_contando_como_error_repetido(server):
    server.agix_decision_record("desplegar el viernes sin margen", False, lesson="sin margen para revertir")
    server.agix_decision_record("desplegar el viernes por la tarde", False, lesson="sin margen para revertir")
    ids = [r["id"] for r in server.agix_metacognition_state()["recent"]]
    server.agix_memory_merge(ids[1], [ids[0]])
    review = server.agix_decision_review(store_in_narrative=False)
    assert review["repeated_errors"] == ["desplegar el viernes por la tarde"]
    assert "Errores repetidos" in _hook("session", server.DATA_DIR)
    # Retirada por obsoleta: deja de contar.
    server.agix_memory_retire([ids[1]], reason="ya no se despliega los viernes")
    assert server.agix_decision_review(store_in_narrative=False)["repeated_errors"] == []
    assert "Errores repetidos" not in _hook("session", server.DATA_DIR)


def test_fusion_valida_tipos_y_estado(server):
    server.agix_decision_record("a", True, lesson="x")
    server.agix_decision_record("b", True, lesson="x")
    server.agix_memory_record("s", "d", "o", True)
    server.agix_memory_record("s2", "d2", "o2", True)
    d1, d2 = [r["id"] for r in server.agix_metacognition_state()["recent"]]
    e1, e2 = [r["id"] for r in server.agix_memory_recent()["recent"]]
    with pytest.raises(ValueError, match="mismo tipo"):
        server.agix_memory_merge(d1, [e1])
    with pytest.raises(ValueError, match="distinto"):
        server.agix_memory_merge(d1, [d1])
    with pytest.raises(ValueError, match="lección consolidable"):
        server.agix_memory_merge(e2, [e1], lesson="x")
    server.agix_memory_merge(d2, [d1])
    with pytest.raises(ValueError, match="ya retiradas"):
        server.agix_memory_merge(d1, [d2])


def test_la_curacion_sobrevive_a_un_reinicio(server, restart):
    server.agix_decision_record("obsoleta", False, lesson="ya no vale")
    did = _ids(server, "obsoleta")[0]
    server.agix_memory_retire([did], reason="motivo")
    srv = restart(server)
    assert srv.agix_decision_review(store_in_narrative=False)["recorded_lessons"] == []
    assert (Path(srv.DATA_DIR) / "curacion.json").exists()
