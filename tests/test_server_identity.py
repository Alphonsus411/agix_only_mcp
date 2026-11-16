"""Tests de la identidad persistente y el autorretrato unificado del servidor MCP."""


def test_identidad_por_defecto(server):
    r = server.agix_identity_define()
    assert r["name"] == "Claude" and r["origin"] == "AGIX MCP"
    assert r["narrative"] == "Soy Claude, nací en AGIX MCP."


def test_definir_y_persistir_identidad(server, restart):
    server.agix_identity_define(name="Luma", origin="el laboratorio de Adolfo",
                                attributes={"rol": "asistente", "estilo": "riguroso"},
                                add_traits=["curioso", "prudente"])
    server.agix_identity_define(remove_attributes=["estilo"], remove_traits=["prudente"])
    server.agix_identity_event("primer servidor MCP desplegado")
    s2 = restart(server)
    r = s2.agix_identity_define()
    assert r["name"] == "Luma" and r["origin"] == "el laboratorio de Adolfo"
    assert r["attributes"] == {"rol": "asistente"}
    assert r["traits"] == ["curioso"]
    assert "rol: asistente" in r["narrative"]
    p = s2.agix_identity_portrait()
    assert p["identity"]["recent_history"] == ["primer servidor MCP desplegado"]
    assert p["self_query"] == "¿Estoy cumpliendo mi propósito como Luma?"


def test_evento_puede_ir_a_la_autonarrativa(server):
    server.agix_identity_event("decidí priorizar la estabilidad", to_narrative=True,
                               values=["calidad"], goals=["estabilidad"])
    server.agix_identity_event("evento solo de identidad")
    yo = server.agix_narrative_self()
    assert yo["experiences"] == 1
    assert yo["goals"][0]["name"] == "estabilidad"


def test_autorretrato_integra_todos_los_modulos(server):
    server.agix_identity_define(add_traits=["metódico"])
    server.agix_narrative_store("cuidé la calidad del código", values=["calidad"], goals=["estabilidad"])
    server.agix_qualia_experience("los tests pasan", 0.9, "alegría")
    for ok in (True, True, True, False):
        server.agix_decision_record("añadir tests", ok, goal="calidad")
    server.agix_decision_record("desplegar el viernes", False, goal="velocidad")
    server.agix_identity_event("integré la metacognición")

    p = server.agix_identity_portrait()
    assert p["identity"]["traits"] == ["metódico"]
    assert p["affect"]["tone"] == "alegría"
    assert p["narrative_self"]["values"][0]["name"] == "calidad"
    assert p["metacognition"]["strong_goals"] == ["calidad"]
    assert p["metacognition"]["weak_goals"] == ["velocidad"]
    assert p["metacognition"]["coherence"] == 0.6
    s = p["synthesis"]
    for fragmento in ("Soy Claude", "metódico", "calidad", "alegría", "60%", "velocidad",
                      "integré la metacognición"):
        assert fragmento in s, fragmento


def test_autorretrato_sin_datos_no_falla(server):
    p = server.agix_identity_portrait()
    assert p["metacognition"]["coherence"] is None
    assert p["synthesis"].startswith("Soy Claude")
