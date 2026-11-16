"""Tests del criterio experimental de modos por balance afectivo (con histéresis)."""
import pytest

HOY = [("eje de AGIX", 0.9, "curiosidad"), ("bug plasticidad", 0.6, "miedo"),
       ("59 tests", 0.85, "alegría"), ("regla de seguridad", 0.4, "sorpresa")]
MALA_RACHA = [("caída prod", 0.9, "miedo"), ("cliente enfadado", 0.8, "tristeza"),
              ("otro fallo", 0.9, "frustración"), ("pista", 0.5, "curiosidad"),
              ("arreglo", 0.8, "alegría"), ("gracias", 0.9, "alegría"), ("deploy ok", 0.9, "alegría")]
BUENA_RACHA = [("a", 0.9, "alegría"), ("b", 0.8, "curiosidad"), ("c", 0.9, "alegría"),
               ("d", 0.5, "sorpresa"), ("e", 0.3, "miedo")]
ABBR = {"normal": "N", "euforia": "E", "desconexion": "D"}


def _modos(server, eventos, criterio="balance"):
    server.agix_qualia_configure(modes=criterio)
    return "".join(ABBR[server.agix_qualia_experience(*ev)["mode"]] for ev in eventos)


def test_balance_es_el_criterio_por_defecto(server):
    assert type(server._spirit).__name__ == "BalancedQualiaSpirit"
    assert server.agix_qualia_state()["config"]["modos"] == "balance"


def test_dia_mixto_no_cae_en_desconexion(server):
    modos = _modos(server, HOY)
    assert "D" not in modos
    assert modos.endswith("N")


def test_criterio_suma_reproduce_la_oscilacion_original(server):
    assert _modos(server, HOY, "suma") == "EDDE"


def test_mala_racha_entra_en_desconexion_y_se_recupera(server):
    modos = _modos(server, MALA_RACHA)
    assert modos == "NNDDDDN"


def test_histeresis_un_evento_positivo_no_saca_de_la_desconexion(server):
    modos = _modos(server, MALA_RACHA)
    primera_d = modos.index("D")
    # tras la primera curiosidad y la primera alegría sigue desconectado
    assert modos[primera_d:primera_d + 4] == "DDDD"


def test_buena_racha_entra_en_euforia_y_sale_con_el_miedo(server):
    assert _modos(server, BUENA_RACHA) == "NEEEN"


def test_modo_y_balance_sobreviven_al_reinicio(server, restart):
    _modos(server, MALA_RACHA[:4])
    assert server.agix_qualia_state()["mode"] == "desconexion"
    s2 = restart(server)
    estado = s2.agix_qualia_state()
    assert estado["mode"] == "desconexion"
    # la histéresis sigue activa: una alegría moderada no basta para salir
    assert s2.agix_qualia_experience("buena noticia", 0.6, "alegría")["mode"] == "desconexion"


def test_cambiar_criterio_en_caliente_conserva_el_estado(server):
    server.agix_qualia_experience("algo", 0.7, "alegría")
    antes = server.agix_qualia_state()["emotions"]
    server.agix_qualia_configure(modes="suma")
    assert type(server._spirit).__name__ == "QualiaSpirit"
    assert server.agix_qualia_state()["emotions"] == antes


def test_criterio_invalido(server):
    with pytest.raises(ValueError):
        server.agix_qualia_configure(modes="otro")


def test_balance_y_activacion(server):
    srv = server
    assert srv.affect_balance({}) == (0.0, 0.0)
    assert srv.affect_balance({"alegría": 0.8, "tristeza": 0.2})[0] == pytest.approx(0.6)
    assert srv.affect_balance({"frustracion": 0.5})[0] == -1.0  # sin tilde también cuenta
    assert srv.affect_balance({"sorpresa": 0.6}) == (0.0, 0.6)  # sin valencia: solo activación
