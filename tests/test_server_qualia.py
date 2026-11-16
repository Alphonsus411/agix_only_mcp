"""Tests de las herramientas Qualia del servidor MCP (persistencia, ciclo, estética, sueños)."""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def server(monkeypatch):
    """Importa server.py con un directorio de datos temporal y propio."""
    data = tempfile.mkdtemp(prefix="agixmcp-test-")
    monkeypatch.setenv("AGIX_MCP_DATA", data)
    monkeypatch.syspath_prepend(str(ROOT))
    cwd = os.getcwd()
    sys.modules.pop("server", None)
    mod = importlib.import_module("server")
    yield mod
    os.chdir(cwd)
    sys.modules.pop("server", None)


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


def test_suenos_persisten(server):
    server.agix_dream_record("un bosque de grafos", "curiosidad", 0.7)
    s2 = _restart(server)
    r = s2.agix_dreams_recent()
    assert r["total"] == 1
    assert "bosque de grafos" in r["poetic"][0]
