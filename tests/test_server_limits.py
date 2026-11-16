"""Límites de entrada: ninguna llamada puede dejar el servidor colgado."""
import time

import pytest


@pytest.mark.parametrize("expression, match", [
    ("9**9**9 == 0", "exponente"),
    ("(2**100) > 0", "exponente"),
    ("x**(y+1) == 1", "exponente"),
    ("x**2 + y**2 + x**3 + y**3 == 0", "potencias"),
    (" and ".join(f"v{i} == 1" for i in range(9)), "variables"),
    ("x == 1 and " * 60 + "x == 1", "larga"),
])
def test_verify_rechaza_lo_que_colgaba(server, expression, match):
    t = time.time()
    with pytest.raises(ValueError, match=match):
        server.agix_verify_expression(expression)
    assert time.time() - t < 1


@pytest.mark.parametrize("expression, expected", [
    ("x + y == 2 and x != y", "unsat"),
    ("x**2 == 1 and y**-1 == -1", "sat"),
    ("((9**16)**16)**16 > 0 and a == 1", "sat"),
    (" and ".join(f"v{i} >= -1" for i in range(8)), "sat"),
    ("esto no es python ==", "unknown"),
])
def test_verify_sigue_funcionando_dentro_de_los_limites(server, expression, expected):
    assert server.agix_verify_expression(expression)["result"] == expected


def test_verify_peor_caso_permitido_es_rapido(server):
    names = "abcdefgh"
    expr = "((9**16)**16)**16 > 0 and " + " and ".join(f"{n}*{n}+{n} >= -5" for n in names)
    t = time.time()
    assert server.agix_verify_expression(expr)["result"] == "sat"
    assert time.time() - t < 10


@pytest.mark.parametrize("kwargs, match", [
    ({"episodes": 100000, "max_steps": 1000}, "episodes"),
    ({"episodes": 10000, "max_steps": 10}, "episodes"),
    ({"episodes": 1000, "max_steps": 1000}, "episodes × max_steps"),
    ({"max_steps": 0}, "max_steps"),
    ({"observations": 1000}, "observations"),
    ({"actions": -1}, "actions"),
])
def test_simulate_rechaza_lo_que_colgaba(server, kwargs, match):
    with pytest.raises(ValueError, match=match):
        server.agix_simulate_agent(**kwargs)


def test_simulate_peor_caso_permitido_es_rapido(server):
    t = time.time()
    server.agix_simulate_agent(observations=100, actions=20, episodes=1000, max_steps=100, seed=1)
    server.agix_simulate_agent(observations=100, actions=20, episodes=10, max_steps=10000, seed=1)
    assert time.time() - t < 20
