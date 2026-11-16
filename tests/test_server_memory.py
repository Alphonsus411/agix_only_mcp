"""Tests de memoria episódica (línea temporal, narrativa) y autonarrativa del servidor MCP."""
import threading
from datetime import datetime, timedelta

import pytest

HOY = datetime.utcnow().date().isoformat()  # los timestamps de AGIX están en UTC
MANANA = (datetime.utcnow().date() + timedelta(days=1)).isoformat()


# --------------------------------------------------------------- memoria episódica


def test_timeline_devuelve_episodios_del_dia(server):
    server.agix_memory_record("bug en la búsqueda", "revisar filtros", "corregido", True)
    server.agix_memory_record("deploy en viernes", "desplegar sin tests", "caída", False)
    r = server.agix_memory_timeline(HOY, HOY)
    assert r["count"] == 2
    assert [e["entrada"] for e in r["episodes"]] == ["bug en la búsqueda", "deploy en viernes"]
    assert r["success_rate"] == 0.5
    assert server.agix_memory_timeline(MANANA)["count"] == 0
    assert server.agix_memory_timeline(HOY, only_success=False)["count"] == 1


def test_timeline_trunca_a_los_mas_recientes(server):
    for i in range(5):
        server.agix_memory_record(f"situación {i}", "d", f"r{i}", True)
    r = server.agix_memory_timeline(HOY, n=2)
    assert r["count"] == 5 and r["truncated"]
    assert [e["entrada"] for e in r["episodes"]] == ["situación 3", "situación 4"]


def test_narrativa_se_reconstruye_tras_reinicio(server, restart):
    for i in range(6):
        server.agix_memory_record(f"episodio {i}", "d", f"resultado {i}", True)
    antes = server.agix_memory_narrative()
    assert "episodio 0" in antes["narrative"] and "episodio 5" in antes["narrative"]
    s2 = restart(server)
    despues = s2.agix_memory_narrative()
    assert despues["narrative"] == antes["narrative"]
    assert despues["episodes_total"] == 6


def test_busqueda_ignora_tildes(server):
    server.agix_memory_record("fallo en la búsqueda híbrida", "ajustar pesos", "mejoró", True)
    r = server.agix_memory_search("busqueda hibrida")
    assert r["results"] and r["results"][0]["entrada"] == "fallo en la búsqueda híbrida"


# --------------------------------------------------------------- autonarrativa


def _historia(server):
    server.agix_narrative_store("Arreglé el bug de la búsqueda con Carlos",
                                values=["calidad"], goals=["estabilidad"], affects=["alivio"],
                                metadata={"proyecto": "neinver"})
    server.agix_narrative_store("Un despliegue un viernes tumbó producción",
                                values=["prudencia"], goals=["estabilidad"], affects=["frustración"],
                                metadata={"proyecto": "neinver"})
    server.agix_narrative_store("Integré AGIX como servidor MCP",
                                values=["curiosidad"], goals=["aprendizaje"], affects=["entusiasmo"],
                                metadata={"proyecto": "agix"})


def test_modelo_de_si_mismo_sobrevive_al_reinicio(server, restart):
    _historia(server)
    antes = server.agix_narrative_self()
    assert antes["goals"][0] == {"name": "estabilidad", "count": 2}
    s2 = restart(server)
    despues = s2.agix_narrative_self()
    assert despues["goals"] == antes["goals"]
    assert despues["values"] == antes["values"] and despues["experiences"] == 3


def test_consulta_ordena_por_relevancia_y_filtra(server):
    _historia(server)
    r = server.agix_narrative_query("busqueda bug", top=2)
    assert r["results"][0]["text"].startswith("Arreglé el bug")
    assert len(r["results"]) == 2
    r = server.agix_narrative_query("estabilidad", filters={"proyecto": "agix"})
    assert [x["text"] for x in r["results"]] == ["Integré AGIX como servidor MCP"]


def test_evaluar_accion_no_cambia_nada(server, restart):
    _historia(server)
    antes = server.agix_narrative_self()
    for _ in range(3):
        r = server.agix_narrative_evaluate_action("push forzado sin revisar")
    assert r["committed"] is False and r["matched"] == []
    despues = server.agix_narrative_self()
    assert (despues["planning_score"], despues["motivation_score"]) == (
        antes["planning_score"], antes["motivation_score"])
    s2 = restart(server)
    assert s2.agix_narrative_self()["planning_score"] == antes["planning_score"]


def test_compatibilidad_ignora_palabras_vacias(server):
    _historia(server)
    r = server.agix_narrative_evaluate_action(
        "llevar la corrección al repo con tests para ganar estabilidad y calidad")
    # contenido: llevar, correccion, repo, tests, ganar, estabilidad, calidad -> 2/7
    assert set(r["matched"]) == {"estabilidad", "calidad"}
    assert r["compatibility"] == pytest.approx(2 / 7, abs=1e-3)
    assert "llevar" in r["unmatched"] and "la" not in r["unmatched"]


def test_evaluar_accion_ignora_tildes_y_persiste_puntuaciones_con_commit(server, restart):
    _historia(server)
    r = server.agix_narrative_evaluate_action("priorizar estabilidad y calidad aunque cause frustracion",
                                              commit=True)
    assert set(r["matched"]) == {"estabilidad", "calidad", "frustración"}
    assert r["compatibility"] > 0 and r["committed"]
    assert r["planning_score"] > 0.5 and r["motivation_score"] > 0.5
    planning, motivation = r["planning_score"], r["motivation_score"]
    s2 = restart(server)
    yo = s2.agix_narrative_self()
    assert yo["planning_score"] == pytest.approx(planning)
    assert yo["motivation_score"] == pytest.approx(motivation)


def test_autonarrativa_funciona_desde_otro_hilo(server):
    errores = []

    def trabajo():
        try:
            server.agix_narrative_store("desde un hilo de trabajo", goals=["robustez"])
            server.agix_narrative_query("hilo")
        except Exception as exc:  # pragma: no cover - el test falla abajo
            errores.append(exc)

    t = threading.Thread(target=trabajo)
    t.start()
    t.join()
    assert not errores, errores
