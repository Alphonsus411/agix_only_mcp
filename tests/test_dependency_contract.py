"""Contrato explícito entre el servidor MCP y la versión instalada de AGIX.

Estos tests son intencionadamente estrictos: no ocultan incompatibilidades con
adaptadores silenciosos ni alteran la instalación de AGIX.
"""
import inspect
from importlib.metadata import version

import pytest

from agix.qualia.spirit import QualiaSpirit
from agix.qualia.emotional_rnn import EmotionalRNN
from agix.orchestrator.synthetic_loop import SyntheticLoop


def _requires_parameter(callable_obj, parameter):
    signature = inspect.signature(callable_obj)
    assert parameter in signature.parameters, (
        f"AGIX {version('agix')}: {callable_obj.__qualname__} no acepta "
        f"{parameter!r}; MCP y motor AGIX tienen contratos incompatibles. "
        "Revisar el parche patches/qualia-fixes.patch en el repositorio "
        "fuente de AGIX antes de desplegar."
    )


def test_qualia_spirit_acepta_atenuacion():
    _requires_parameter(QualiaSpirit.__init__, "atenuacion")


def test_emotional_rnn_expande_entradas():
    assert callable(getattr(EmotionalRNN, "expand_inputs", None)), (
        f"AGIX {version('agix')}: EmotionalRNN carece de expand_inputs; "
        "la expansión de emociones no está soportada por esta dependencia."
    )


@pytest.mark.parametrize("parameter", ["inercia_pad", "emotion_simulator"])
def test_synthetic_loop_acepta_parametros_requeridos(parameter):
    _requires_parameter(SyntheticLoop.__init__, parameter)
