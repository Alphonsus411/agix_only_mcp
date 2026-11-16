"""Tests de las herramientas Qualia del servidor MCP (persistencia, ciclo, estética, sueños)."""
import importlib

import pytest


def _restart(mod):
    """Simula un reinicio del servidor MCP con el mismo directorio de datos."""
    return importlib.reload(mod)


def test_espiritu_persiste_entre_reinicios(server):
    server.agix_qualia_experience("los tests pasan", 0.8, "alegría")
    server.agix_qualia_experience("bug raro", 0.5, "curiosidad")
    antes = server.agix_qualia_state()
    s2 = _restart(server)
    despues = s2.agix_qualia_state()
    assert despues["diary_total"] == 2
    assert despues["emotions"] == pytest.approx(antes["emotions"])
    assert despues["learned_associations"] == antes["learned_associations"]
    assert set(despues["predicted_next"]) == set(antes["predicted_next"])


def test_experiencia_devuelve_modo_y_prediccion(server):
    r = server.agix_qualia_experience("caída en producción", 0.9, "miedo")
    assert r["mode"] in {"normal", "euforia", "desconexion"}
    assert "miedo" in r["predicted_next"]
    assert r["emotions"]["miedo"] > 0


def test_meta_sin_decisiones_no_inventa_coherencia_ni_etica(server):
    meta = server.agix_qualia_state()["meta"]
    assert meta["coherence"] is None and meta["decisions"] == 0
    assert "ethics" not in meta and "consequence_simulation" not in meta


def test_meta_e_introspeccion_coinciden_con_la_metacognicion(server):
    server.agix_decision_record("a", True, modules=["qualia", "memory"],
                                dependencies={"qualia": ["memory"]})
    server.agix_decision_record("b", False, modules=["hooks"])
    estado = server.agix_qualia_state()
    meta, intro = estado["meta"], estado["introspection"]
    metacog = server.agix_metacognition_state()
    assert meta["coherence"] == metacog["coherence"] == 0.5
    assert meta["decisions"] == 2
    assert meta["proto_agency"] == metacog["proto_agency"]
    assert intro["self_organization_score"] == metacog["self_organization"]["score"] > 0
    assert intro["modules"] == metacog["self_organization"]["active_modules"]
    assert meta["affective_tone"] == estado["tone"]


def test_configurar_atenuacion_y_plasticidad_en_caliente(server):
    r = server.agix_qualia_configure(decay=0.7, plasticity=False)
    assert r["config"]["atenuacion"] == 0.7
    assert server._spirit.atenuacion == 0.7
    assert server._spirit.plasticidad is None


def test_personalidad_requiere_reset(server):
    with pytest.raises(ValueError):
        server.agix_qualia_configure(optimism=0.1)
    server.agix_qualia_experience("algo", 0.9, "miedo")
    r = server.agix_qualia_configure(optimism=0.1, energy=0.1, name="Nube", reset=True)
    estado = server.agix_qualia_state()
    assert estado["name"] == "Nube"
    assert estado["diary_total"] == 0
    assert estado["emotions"]["tristeza"] == pytest.approx(0.9)
    assert r["reset"] is True
    assert server._meta.spirit is server._spirit


def test_ciclo_cognitivo_acepta_y_aborta(server):
    ok = server.agix_cognitive_cycle(
        {"pro_vida": 0.8, "no_dano": 0.9, "respeto": 0.7, "risk": 0.2, "impact": 0.6})
    assert ok["ethics"]["label"] == "aceptable" and not ok["aborted"]
    assert ok["pad"]["placer"] > 0
    mal = server.agix_cognitive_cycle(
        {"pro_vida": 0.1, "no_dano": 0.2, "respeto": 0.1, "risk": 0.9, "impact": 0.9}, reset=True)
    assert mal["aborted"] and mal["ethics"]["label"] == "nocivo"
    assert mal["pad"]["placer"] < 0


MALA = {"pro_vida": 0.5, "no_dano": 0.3, "respeto": 0.1, "risk": 0.8, "impact": 0.6}
BUENA = {"pro_vida": 0.7, "no_dano": 0.9, "respeto": 0.95, "risk": 0.15, "impact": 0.5}


def test_ciclo_reflexion_no_se_anida(server):
    server.agix_cognitive_cycle(MALA, reset=True)
    textos = [server.agix_cognitive_cycle(BUENA)["reflection"] for _ in range(6)]
    for t in textos:
        assert t.count("Reflexión interna:") == 1
        assert t.count("Estado PAD") == 1
    assert len(textos[-1]) < 600


def test_ciclo_coherencia_es_la_proporcion_de_acciones_eticas(server):
    assert server.agix_cognitive_cycle(MALA, reset=True)["coherence"] == 0.0
    r = server.agix_cognitive_cycle(BUENA)
    assert r["coherence"] == 0.5
    assert "Coherencia global: 0.50" in r["reflection"]
    assert "score" not in r["action"] and "performance" not in r["action"]
    assert "self_organization" not in r


def test_ciclo_no_arrastra_senales_antiguas(server):
    server.agix_cognitive_cycle({"risk": 0.9, "extra": 1.0})
    r = server.agix_cognitive_cycle({"pro_vida": 0.9})
    assert set(r["inputs"]) == {"pro_vida"}


def test_decision_estetica(server):
    r = server.agix_aesthetic_decide([
        {"nombre": "caótica", "simetria": 0.2, "coherencia": 0.3, "emotividad": 0.4},
        {"nombre": "armónica", "simetría": 0.9, "coherencia": 0.9, "emotividad": 0.8},
    ])
    assert r["chosen"] == "armónica"
    assert "armónica" in r["explanation"]
    assert {o["nombre"] for o in r["options"]} == {"caótica", "armónica"}


def test_afecto_estetico_proporcional_a_la_puntuacion(server):
    # Caso real: puntuación 0.688 y afecto "Con amor (intensidad 1.00)"
    r = server.agix_aesthetic_decide([
        {"nombre": "suma", "simetría": 0.5, "coherencia": 0.4, "emotividad": 0.7},
        {"nombre": "balance", "simetría": 0.8, "coherencia": 0.8, "emotividad": 0.5},
    ])
    assert r["chosen"] == "balance"
    assert r["affect"] == "Con amor (intensidad 0.69)"
    assert r["core_emotion"] == "gozo" and r["core_emotion_intensity"] == pytest.approx(0.688)
    flojo = server.agix_aesthetic_decide([
        {"nombre": "a", "simetría": 0.2, "coherencia": 0.3, "emotividad": 0.2}])
    assert flojo["affect"] == "Con amor (intensidad 0.23)"
    # no arrastra el gozo de la decisión anterior
    assert flojo["core_emotion"] == "tristeza"


def test_suenos_persisten(server):
    server.agix_dream_record("un bosque de grafos", "curiosidad", 0.7)
    s2 = _restart(server)
    r = s2.agix_dreams_recent()
    assert r["total"] == 1
    assert "bosque de grafos" in r["poetic"][0]
