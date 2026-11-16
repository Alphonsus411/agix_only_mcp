"""Tests de la metacognición persistente del servidor MCP."""
import pytest


def _decisiones(server):
    server.agix_decision_record("desplegar el viernes", False, lesson="sin tests de integración",
                                goal="estabilidad")
    server.agix_decision_record("añadir tests de humo", True, lesson="detectaron el fallo antes",
                                goal="estabilidad")
    server.agix_decision_record("desplegar el viernes", False, lesson="otra vez sin margen para revertir",
                                goal="estabilidad")
    server.agix_decision_record("refactorizar el indexador", True, goal="calidad")


def test_registro_y_coherencia(server):
    r = server.agix_decision_record("probar antes de subir", True, goal="calidad")
    assert r["coherence"] == 1.0 and r["total_decisions"] == 1
    assert r["recorded"]["resultado"]["exito"] is True and r["recorded"]["timestamp"]
    _decisiones(server)
    estado = server.agix_metacognition_state()
    assert estado["total_decisions"] == 5
    assert estado["coherence"] == pytest.approx(3 / 5)
    assert estado["by_goal"]["estabilidad"] == {"successes": 1, "errors": 2, "success_rate": 0.333}


def test_decisiones_persisten_tras_reinicio(server, restart):
    _decisiones(server)
    server.agix_decision_review(store_in_narrative=False)
    antes = server.agix_metacognition_state()
    s2 = restart(server)
    despues = s2.agix_metacognition_state()
    assert despues["total_decisions"] == 4
    assert despues["coherence"] == antes["coherence"]
    assert despues["recent"] == antes["recent"]
    assert despues["last_review"]["summary"] == antes["last_review"]["summary"]


def test_revision_con_lecciones_reales_y_errores_repetidos(server):
    _decisiones(server)
    r = server.agix_decision_review()
    assert r["successes"] == 2 and r["errors"] == 2
    lecciones = {x["lesson"] for x in r["recorded_lessons"]}
    assert "sin tests de integración" in lecciones and "detectaron el fallo antes" in lecciones
    assert r["repeated_errors"] == ["desplegar el viernes"]
    assert r["by_action"]["desplegar el viernes"]["success_rate"] == 0.0
    assert r["by_goal"]["calidad"]["success_rate"] == 1.0


def test_revision_tono_proporcional_aunque_el_primer_registro_sea_un_error(server):
    # Caso real: 1 error primero y 4 aciertos daba "frustracion"
    server.agix_decision_record("editar settings.json", False,
                                lesson="la regla de deny lo bloqueó, y era lo correcto")
    for a in ("instalador", "corregir plasticidad", "histéresis", "revisar Qualia"):
        server.agix_decision_record(a, True)
    r = server.agix_decision_review()
    assert (r["successes"], r["errors"]) == (4, 1)
    assert r["emotional_tone"] == "satisfaccion"
    assert r["lessons"] == ["'editar settings.json': la regla de deny lo bloqueó, y era lo correcto"]


def test_revision_tono_frustracion_y_ambivalencia(server):
    server.agix_decision_record("a", True)
    server.agix_decision_record("b", False)
    assert server.agix_decision_review(store_in_narrative=False)["emotional_tone"] == "ambivalencia"
    server.agix_decision_record("c", False)
    r = server.agix_decision_review(store_in_narrative=False)
    assert r["emotional_tone"] == "frustracion"
    assert "Mejorar acción 'c'" in r["lessons"]  # sin lección registrada: mensaje genérico


def test_revision_se_guarda_en_la_autonarrativa(server):
    _decisiones(server)
    antes = server.agix_narrative_self()["experiences"]
    server.agix_decision_review()
    assert server.agix_narrative_self()["experiences"] == antes + 1
    r = server.agix_narrative_query("revision decisiones aciertos errores", top=1)
    assert r["results"][0]["text"].startswith("Revisión de 4 decisiones")


def test_revision_sin_guardar_no_toca_la_autonarrativa(server):
    _decisiones(server)
    server.agix_decision_review(store_in_narrative=False)
    assert server.agix_narrative_self()["experiences"] == 0


def test_meta_evaluate_comparte_historial(server, restart):
    server.agix_decision_record("algo que falló", False)
    r = server.agix_meta_evaluate({"metas": [], "historial": []}, "algo que salió bien",
                                  {"exito": True, "impacto": {"pro_vida": 0.8, "no_dano": 0.9, "respeto": 0.8}})
    assert r["coherence"] == 0.5
    s2 = restart(server)
    assert s2.agix_metacognition_state()["total_decisions"] == 2


def test_auto_organizacion_con_modulos(server):
    server.agix_decision_record("integrar memoria", True, modules=["memoria", "qualia"],
                                dependencies={"qualia": ["memoria"]})
    so = server.agix_metacognition_state()["self_organization"]
    assert so["active_modules"] == ["memoria", "qualia"]
    assert so["module_complexity"] > 0 and so["interdependency"] == 0.5
    assert so["score"] > 0


def test_conflicto_gana_la_voz_respaldada_por_la_memoria(server):
    server.agix_memory_record("migración a pgvector", "migrar por lotes", "sin pérdida de datos", True)
    r = server.agix_resolve_internal_conflict([
        {"text": "reescribir todo de cero", "weight": 0.6},
        {"text": "migrar por lotes como la otra vez", "weight": 0.55, "memory": "migración a pgvector"},
    ])
    assert r["winner"] == "migrar por lotes como la otra vez"
    assert r["memory_backed"] == ["migrar por lotes como la otra vez"]
    assert r["resolved"] and r["iterations"] > 1


def test_conflicto_sin_respaldo_gana_el_mayor_peso(server):
    r = server.agix_resolve_internal_conflict([
        {"text": "A", "weight": 0.9}, {"text": "B", "weight": 0.2}])
    assert r["winner"] == "A" and r["iterations"] == 1 and r["resolved"]


def test_conflicto_requiere_dos_hipotesis(server):
    with pytest.raises(ValueError):
        server.agix_resolve_internal_conflict([{"text": "sola", "weight": 1}])
