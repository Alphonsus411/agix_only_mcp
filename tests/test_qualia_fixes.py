"""Tests de las correcciones locales a agix 1.11.0 (Qualia y SyntheticLoop)."""
import contextlib
import io

import numpy as np
import pytest

import agix.memory  # noqa: F401  (evita la importación circular de reasoning)
from agix.emotion.emotion_simulator import PADState
from agix.orchestrator.synthetic_loop import SyntheticLoop
from agix.perception.sensor_map import Sensor, SensorMap
from agix.qualia.emotional_rnn import EmotionalRNN
from agix.qualia.neuro_plastic import EmotionalPlasticity
from agix.qualia.spirit import QualiaSpirit


# --------------------------------------------------------------- plasticidad


def test_plasticidad_sin_aprendizaje_es_identidad():
    p = EmotionalPlasticity()
    emociones = {"alegría": 0.7, "miedo": 0.2}
    p._ensure_model(emociones)
    assert p.adjust(emociones) == pytest.approx(emociones)


def test_plasticidad_no_anula_ni_dispara_las_emociones():
    p = EmotionalPlasticity(learning_rate=0.05)
    emociones = {"alegría": 0.8, "curiosidad": 0.6}
    for _ in range(200):
        p.update(emociones)
        emociones = p.adjust(emociones)
    assert all(0.0 < v <= 1.0 for v in emociones.values())
    assert sum(emociones.values()) > 0.5


def test_plasticidad_acepta_emociones_nuevas_y_conserva_lo_aprendido():
    p = EmotionalPlasticity(learning_rate=0.1)
    p.update({"alegría": 0.9})
    peso_aprendido = p.model.weights[0, 0]
    p.update({"alegría": 0.1, "miedo": 0.8})
    assert p.emotion_order == ["alegría", "miedo"]
    assert p.model.weights.shape == (2, 2)
    assert p.model.weights[0, 0] >= peso_aprendido
    assert p.adjust({"alegría": 0.0, "miedo": 0.8})["miedo"] > 0.0


def test_adjust_conserva_emociones_desconocidas():
    p = EmotionalPlasticity()
    p.update({"alegría": 0.5})
    assert p.adjust({"sorpresa": 0.4})["sorpresa"] == 0.4


# --------------------------------------------------------------- espíritu


def test_espiritu_con_plasticidad_no_queda_desconectado():
    # Con el código original, la primera experiencia ya anulaba todas las emociones
    sp = QualiaSpirit("Luma", plasticidad=True)
    sp.experimentar("se encuentra la causa", 0.6, "curiosidad")
    sp.experimentar("arreglo desplegado", 0.8, "alegría")
    assert not sp.modo_desconexion
    assert sp.estado_emocional.tono_general() == "alegría"


def test_espiritu_con_plasticidad_no_pierde_emociones_nuevas():
    sp = QualiaSpirit("Luma", plasticidad=True)
    base = QualiaSpirit("Base")
    eventos = [("tests fallan", 0.7, "tristeza"), ("caída en producción", 0.9, "miedo"),
               ("arreglo desplegado", 0.9, "alegría")]
    for ev in eventos:
        sp.experimentar(*ev)
        base.experimentar(*ev)
    emociones = sp.estado_emocional.emociones
    assert set(emociones) == set(base.estado_emocional.emociones)
    assert emociones["miedo"] > 0.0, "una emoción nueva no debe perderse"
    # La plasticidad modula, pero no colapsa: misma magnitud que sin plasticidad
    assert sum(emociones.values()) == pytest.approx(
        sum(base.estado_emocional.emociones.values()), rel=0.35)


def test_prediccion_incluye_emociones_nuevas():
    sp = QualiaSpirit("Kai")
    sp.experimentar("a", 0.5, "alegría")
    sp.experimentar("b", 0.6, "alegría")
    sp.experimentar("c", 0.8, "miedo")
    pred = sp.predecir_futuro()
    assert set(pred) == {"alegría", "miedo"}
    assert all(len(v) == 2 for v in sp._historial_vectores)


# --------------------------------------------------------------- atenuación

MALA_RACHA = [("tests fallan", 0.7, "tristeza"), ("caída en producción", 0.9, "miedo")]


def _eventos_hasta_reconectar(sp, maximo=20):
    for ev in MALA_RACHA:
        sp.experimentar(*ev)
    assert sp.modo_desconexion
    for n in range(1, maximo + 1):
        sp.experimentar(f"buena noticia {n}", 0.8, "alegría")
        if not sp.modo_desconexion:
            return n
    return None


@pytest.mark.parametrize("plasticidad", [False, True])
def test_atenuacion_permite_salir_de_la_desconexion(plasticidad):
    sp = QualiaSpirit("Luma", plasticidad=plasticidad)
    n = _eventos_hasta_reconectar(sp)
    assert n is not None and n <= 10
    assert sp.estado_emocional.tono_general() == "alegría"


def test_sin_atenuacion_se_conserva_el_comportamiento_original():
    sp = QualiaSpirit("Luma", atenuacion=1.0)
    assert _eventos_hasta_reconectar(sp) is None


def test_mayor_atenuacion_recupera_antes():
    lenta = _eventos_hasta_reconectar(QualiaSpirit("A", atenuacion=0.95))
    rapida = _eventos_hasta_reconectar(QualiaSpirit("B", atenuacion=0.7))
    assert rapida < lenta


@pytest.mark.parametrize("valor", [0.0, -0.1, 1.5])
def test_atenuacion_invalida(valor):
    with pytest.raises(ValueError):
        QualiaSpirit("X", atenuacion=valor)


def test_atenuacion_no_afecta_a_una_experiencia_aislada():
    sp = QualiaSpirit("X")
    sp.experimentar("hola", 0.6, "curiosidad")
    assert sp.estado_emocional.emociones["curiosidad"] == pytest.approx(0.6)


# --------------------------------------------------------------- RNN


@pytest.mark.parametrize("tipo", ["rnn", "lstm"])
def test_rnn_crece_conservando_pesos(tipo):
    rnn = EmotionalRNN(2, rnn_type=tipo)
    seq = np.array([[0.2, 0.4], [0.3, 0.5]])
    rnn.update(seq[:-1], seq[1:])
    why_antes = rnn.Why.copy()
    rnn.expand_inputs(1)
    assert rnn.input_size == rnn.output_size == 3
    np.testing.assert_array_equal(rnn.Why[:2], why_antes)
    out = rnn.forward(np.array([[0.2, 0.4, 0.9]]))
    assert out.shape == (1, 3)


def test_rnn_entrada_nueva_no_altera_salida_previa():
    rnn = EmotionalRNN(2)
    x = np.array([[0.2, 0.4]])
    rnn.reset_state()
    antes = rnn.forward(x)[-1]
    rnn.expand_inputs(1)
    rnn.reset_state()
    despues = rnn.forward(np.array([[0.2, 0.4, 0.0]]))[-1]
    np.testing.assert_allclose(despues[:2], antes)


# --------------------------------------------------------------- SyntheticLoop


def _loop(valores):
    sm = SensorMap()
    for k, v in valores.items():
        sm.registrar_sensor(Sensor(k, "interno"))
        sm.enviar_entrada(k, v)
    return SyntheticLoop(sm)


def _cycle(loop):
    with contextlib.redirect_stdout(io.StringIO()):
        return loop.cycle()


ETICA = {"pro_vida": 0.8, "no_dano": 0.9, "respeto": 0.7, "risk": 0.2, "impact": 0.6}
DANINA = {"pro_vida": 0.1, "no_dano": 0.2, "respeto": 0.1, "risk": 0.9, "impact": 0.9}


def test_loop_emocion_reacciona_a_las_entradas():
    pad_etica = _cycle(_loop(ETICA))["emocion"]
    pad_danina = _cycle(_loop(DANINA))["emocion"]
    assert pad_etica.placer > 0 > pad_danina.placer
    assert pad_danina.activacion > pad_etica.activacion


def test_loop_no_cuenta_las_entradas_dos_veces():
    pad = _cycle(_loop(ETICA))["emocion"]
    pesos = SyntheticLoop.DEFAULT_PAD_WEIGHTS
    esperado = sum(pesos[k].placer * v for k, v in ETICA.items())
    assert pad.placer == pytest.approx(esperado)


def test_loop_no_acumula_moduladores():
    loop = _loop(ETICA)
    for _ in range(3):
        _cycle(loop)
    assert len(loop.emotion._moduladores) == 1


def test_loop_sigue_abortando_acciones_daninas():
    out = _cycle(_loop(DANINA))
    assert out["etica"]["label"] == "nocivo"
    assert out["accion"].get("aborted") is True


def test_loop_acepta_simulador_propio():
    from agix.emotion.emotion_simulator import EmotionSimulator

    sim = EmotionSimulator(pesos={"risk": PADState(-1.0, 0.0, 0.0)})
    sm = SensorMap()
    sm.registrar_sensor(Sensor("risk", "interno"))
    sm.enviar_entrada("risk", 0.5)
    loop = SyntheticLoop(sm, emotion_simulator=sim)
    assert _cycle(loop)["emocion"].placer == pytest.approx(-0.5)
