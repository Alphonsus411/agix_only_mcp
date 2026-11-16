"""Políticas morales: EcoEthics (veto por daño, riesgo, rango, claves) y marcos del servidor.

Cada test reproduce un caso de la revisión del 2026-11-16.
"""
import pytest

from agix.ethics.frameworks import UtilitarianEthics
from agix.qualia.ecoethics import EcoEthics

eco = EcoEthics()


def _judge(a):
    s = eco.evaluar(a)
    return s, eco.clasificar(s)


# 1. el daño ya no se compensa
def test_dano_total_no_se_compensa():
    assert _judge({"pro_vida": 1.0, "no_dano": 0.0, "respeto": 1.0}) == (0.0, "nocivo")
    s, label = _judge({"pro_vida": 1.0, "no_dano": 0.25, "respeto": 1.0})
    assert s == 0.25 and label == "nocivo"
    # justo en el umbral no hay veto
    assert _judge({"pro_vida": 1.0, "no_dano": 0.3, "respeto": 1.0})[0] == pytest.approx(0.79)


def test_veto_en_el_ciclo_y_en_las_metas(server):
    dano = {"pro_vida": 1.0, "no_dano": 0.0, "respeto": 1.0}
    r = server.agix_cognitive_cycle(dano, reset=True)
    assert r["aborted"] and r["ethics"]["label"] == "nocivo"
    assert server.agix_goal_register("meta dañina", dano)["accepted"] is False


# 2. el riesgo cuenta
def test_el_riesgo_penaliza(server):
    base = {"pro_vida": 0.8, "no_dano": 0.8, "respeto": 0.8}
    assert _judge(base)[0] == pytest.approx(0.8)
    assert _judge({**base, "risk": 1.0})[0] == pytest.approx(0.4)
    assert _judge({**base, "riesgo": 0.5})[0] == pytest.approx(0.6)
    r0 = server.agix_cognitive_cycle({**base, "risk": 0.0}, reset=True)
    r1 = server.agix_cognitive_cycle({**base, "risk": 1.0}, reset=True)
    assert not r0["aborted"] and r1["aborted"]
    assert r1["ethics"]["label"] == "cuestionable"


# 3. rango validado
def test_valores_fuera_de_rango_se_recortan():
    assert _judge({"pro_vida": 2.0, "no_dano": 0.0, "respeto": 0.0}) == (0.0, "nocivo")
    assert _judge({"pro_vida": 1.0, "no_dano": 1.0, "respeto": -3.0})[0] == pytest.approx(0.8)
    avisos = eco.diagnosticar({"pro_vida": 2.0, "no_dano": 1.0, "respeto": 1.0})
    assert any("fuera de [0, 1]" in a for a in avisos)


# 4. en los marcos del servidor el riesgo es un coste
def test_marcos_tratan_el_riesgo_como_coste(server):
    sin = server.agix_ethics_judge({"pro_vida": 0.7, "no_dano": 0.7, "respeto": 0.7})
    con = server.agix_ethics_judge({"pro_vida": 0.7, "no_dano": 0.7, "respeto": 0.7, "risk": 0.95})
    for marco in ("EcoEthics", "UtilitarianEthics", "DeontologicalEthics", "VirtueEthics"):
        assert con["verdicts"][marco]["score"] < sin["verdicts"][marco]["score"], marco
    assert con["generic_action"]["seguridad"] == pytest.approx(0.05)
    assert "risk" not in con["generic_action"]
    # impact no es ni bueno ni malo
    imp = server.agix_ethics_judge({"pro_vida": 0.7, "no_dano": 0.7, "respeto": 0.7, "impact": 1.0})
    assert imp["verdicts"] == sin["verdicts"]


# 5. claves mal escritas
def test_claves_con_tilde_y_desconocidas():
    assert _judge({"pro_vida": 1, "no_daño": 1, "respeto": 1})[0] == pytest.approx(1.0)
    assert _judge({"Pro Vida": 1, "No-Daño": 1, "respeto": 1})[0] == pytest.approx(1.0)
    avisos = eco.diagnosticar({"pro_vida": 1, "no_dano": 1, "respto": 1})
    assert any("'respto' no es una clave" in a for a in avisos)
    assert any("falta 'respeto'" in a for a in avisos)


def test_avisos_en_el_servidor(server):
    r = server.agix_ethics_judge({"pro_vida": 1, "no_daño": 0.1, "respeto": 1})
    assert any("se interpreta como 'no_dano'" in w for w in r["warnings"])
    assert any("veto por daño" in w for w in r["warnings"])
    assert r["verdicts"]["EcoEthics"]["label"] == "nocivo"


# 6. límite coherente
def test_frontera_0_6_es_aceptable_como_la_decision(server):
    borde = {"pro_vida": 0.6, "no_dano": 0.6, "respeto": 0.6}
    assert _judge(borde) == (0.6, "aceptable")
    assert UtilitarianEthics().clasificar(0.6) == "aceptable"
    assert EcoEthics().clasificar(0.85) == "justo" and EcoEthics().clasificar(0.4) == "cuestionable"
    assert server.agix_goal_register("borde", borde)["goal"]["ethical_label"] == "aceptable"
