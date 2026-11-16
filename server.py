"""Servidor MCP que expone AGIX como herramientas de apoyo al razonamiento.

Transporte: stdio. Nada debe escribirse en stdout salvo el protocolo MCP, por eso
el logging va a stderr y cada llamada a AGIX se envuelve en redirect_stdout.

Estado persistente en AGIX_MCP_DATA (por defecto ~/.agix-mcp):
  memoria.json   experiencias registradas (GestorDeMemoria)
  ontologia.json conceptos, atributos y relaciones (Ontology)
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import warnings
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path
from typing import Any

# --- Logging y entorno antes de importar agix (agix configura logging al importarse)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
log = logging.getLogger("agix-mcp")

DATA_DIR = Path(os.environ.get("AGIX_MCP_DATA", Path.home() / ".agix-mcp")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
# GestorDeMemoria solo acepta rutas dentro del cwd.
os.chdir(DATA_DIR)
MEMORY_FILE = "memoria.json"
ONTOLOGY_FILE = DATA_DIR / "ontologia.json"

with contextlib.redirect_stdout(sys.stderr):
    import agix.memory  # noqa: F401  (evita la importación circular de reasoning)
    import numpy as np
    from agix.memory import GestorDeMemoria
    from agix.memory.symbolic import Ontology
    from agix.reasoning.logic_core import Fact
    from agix.evaluation.formal.verifier import verify as agix_verify
    from agix.evaluation.formal.fagi_index import compute_fagi_index
    from agix.evaluation.harmony import HarmonyEvaluator
    from agix.ethics import AlignmentInterface
    from agix.ethics.frameworks import DeontologicalEthics, UtilitarianEthics, VirtueEthics
    from agix.qualia.ecoethics import EcoEthics
    from agix.metacognition.intention_evaluator import IntentionEvaluator
    from agix.metacognition.dynamic_evaluator import DynamicMetaEvaluator
    from agix.goals.goal_engine import GoalEngine
    from agix.emotion.emotion_simulator import EmotionSimulator, PADState
    from agix.qualia.qualia_core import EmotionalState
    from agix.qualia.spirit import QualiaSpirit

from mcp.server.mcpserver import MCPServer

INSTRUCTIONS = """\
Herramientas de AGIX (framework de Adolfo González) para apoyar el razonamiento.
- agix_logic_infer: encadenamiento hacia delante con variables; úsalo para comprobar
  deducciones (transitividad, reglas de negocio, dependencias) en vez de hacerlas de cabeza.
- agix_verify_expression: satisfacibilidad de expresiones (dominio pequeño, por defecto -1..1).
- agix_fuzzy_infer: decisiones con lógica difusa (riesgo, prioridad, confianza).
- agix_nash_equilibria: equilibrios de Nash de juegos bimatriciales.
- agix_memory_*: memoria persistente de experiencias entre sesiones (decisión, resultado, lección).
- agix_ontology_*: grafo persistente de conceptos, atributos y relaciones.
- agix_ethics_judge / agix_intention_evaluate / agix_meta_evaluate / agix_goal_register:
  evaluación ética e intencional; requieren que traduzcas el plan a puntuaciones 0..1.
- agix_emotion_update / agix_qualia_*: modelo afectivo PAD y espíritu Qualia (experimental).
- agix_fagi_index, agix_harmony, agix_simulate_agent: métricas y simulación de agentes.
Las puntuaciones éticas/afectivas son heurísticas: tómalas como una segunda opinión, no como verdad.
"""

server = MCPServer(name="agix", version="1.0.0", instructions=INSTRUCTIONS)

# --------------------------------------------------------------------------- utilidades


def _plain(obj: Any) -> Any:
    """Convierte numpy, tuplas, dataclasses y similares a tipos JSON."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        return round(obj, 4)
    if hasattr(obj, "__dataclass_fields__"):
        return _plain({k: getattr(obj, k) for k in obj.__dataclass_fields__})
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(sys.stderr):
        yield


# --------------------------------------------------------------------------- estado

_memory = GestorDeMemoria()
if Path(MEMORY_FILE).exists():
    try:
        with _quiet():
            _memory.cargar(MEMORY_FILE)
    except Exception as exc:  # memoria corrupta: no bloquear el servidor
        log.warning("No se pudo cargar la memoria: %s", exc)

_ontology = Ontology()


def _load_ontology() -> None:
    if not ONTOLOGY_FILE.exists():
        return
    data = json.loads(ONTOLOGY_FILE.read_text(encoding="utf-8"))
    for name, c in data.items():
        concept = _ontology.conceptualize(name)
        concept.attributes.update(c.get("attributes", {}))
        for rel, targets in c.get("relations", {}).items():
            concept.relations.setdefault(rel, [])
            for t in targets:
                if t not in concept.relations[rel]:
                    concept.relations[rel].append(t)


def _save_ontology() -> None:
    data = {
        n: {"attributes": c.attributes, "relations": c.relations}
        for n, c in _ontology.concepts.items()
    }
    ONTOLOGY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


try:
    _load_ontology()
except Exception as exc:
    log.warning("No se pudo cargar la ontología: %s", exc)

_spirit = QualiaSpirit(nombre="Claude")
_meta = DynamicMetaEvaluator(spirit=_spirit)
_goals = GoalEngine()

# --------------------------------------------------------------------------- razonamiento lógico

_ATOM = re.compile(r"^\s*(?P<neg>no\s+|not\s+|!)?(?P<pred>[\wáéíóúñü]+)\s*\((?P<args>[^()]*)\)\s*$", re.I)


def _parse_atom(text: str) -> tuple[bool, str, tuple[str, ...]]:
    m = _ATOM.match(text)
    if not m:
        raise ValueError(f"Átomo mal formado: {text!r}. Formato: predicado(a, b)")
    args = tuple(a.strip() for a in m.group("args").split(",") if a.strip())
    return bool(m.group("neg")), m.group("pred"), args


def _is_var(term: str) -> bool:
    return term[:1].isupper() or term.startswith("?") or term == "_"


def _split_top(text: str, sep: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _parse_rule(text: str):
    if "->" not in text:
        raise ValueError(f"Regla sin '->': {text!r}. Formato: p(X,Y), q(Y,Z) -> r(X,Z)")
    body, head = text.split("->", 1)
    conds = [_parse_atom(a) for a in _split_top(body, ",")]
    heads = [_parse_atom(a) for a in _split_top(head, ",")]
    if any(neg for neg, _, _ in heads):
        raise ValueError("La conclusión de una regla no puede estar negada")
    if not any(not neg for neg, _, _ in conds):
        raise ValueError("Cada regla necesita al menos una condición positiva")
    return conds, heads


def _unify(pattern: tuple[str, ...], fact: tuple[str, ...], binding: dict) -> dict | None:
    if len(pattern) != len(fact):
        return None
    b = dict(binding)
    for p, v in zip(pattern, fact):
        if p == "_":
            continue
        if _is_var(p):
            if p in b and b[p] != v:
                return None
            b[p] = v
        elif p != v:
            return None
    return b


def _ground(args: tuple[str, ...], b: dict) -> tuple[str, ...]:
    return tuple(b.get(a, a) for a in args)


def _matches(conds, idx, facts: set, binding: dict):
    if idx == len(conds):
        yield binding
        return
    neg, pred, args = conds[idx]
    if neg:  # negación por fallo: se evalúa con las variables ya ligadas
        ground = _ground(args, binding)
        if any(_is_var(a) and a != "_" for a in ground):
            raise ValueError("Las variables de una condición negada deben aparecer antes en positivo ('_' sí vale)")
        if not any(fp == pred and _unify(ground, fa, {}) is not None for fp, fa in facts):
            yield from _matches(conds, idx + 1, facts, binding)
        return
    for fpred, fargs in list(facts):
        if fpred != pred:
            continue
        b = _unify(args, fargs, binding)
        if b is not None:
            yield from _matches(conds, idx + 1, facts, b)


def _fact_str(pred: str, args: tuple[str, ...]) -> str:
    return str(Fact(pred, list(args)))


@server.tool()
def agix_logic_infer(
    facts: list[str],
    rules: list[str] | None = None,
    transitive: list[str] | None = None,
    symmetric: list[str] | None = None,
    query: str | None = None,
    max_iterations: int = 50,
) -> dict:
    """Encadenamiento hacia delante hasta punto fijo, con variables y explicación.

    facts: hechos base, p. ej. ["padre(ana,juan)", "padre(juan,maria)"].
    rules: reglas "cond1, cond2 -> conclusion"; las variables empiezan por mayúscula o '?',
        '_' es comodín y "no p(X)" es negación por fallo. Ej.: "padre(X,Y), padre(Y,Z) -> abuelo(X,Z)".
    transitive / symmetric: nombres de predicados binarios con esa propiedad (atajo de reglas).
    query: patrón opcional para filtrar el resultado, p. ej. "abuelo(ana, Q)".
    Devuelve los hechos derivados con la regla y la sustitución que los produjo.
    """
    base = set()
    for f in facts:
        neg, pred, args = _parse_atom(f)
        if neg:
            raise ValueError(f"Los hechos no pueden estar negados: {f!r}")
        base.add((pred, args))

    rule_texts = list(rules or [])
    for p in transitive or []:
        rule_texts.append(f"{p}(X,Y), {p}(Y,Z) -> {p}(X,Z)")
    for p in symmetric or []:
        rule_texts.append(f"{p}(X,Y) -> {p}(Y,X)")
    parsed = [(t, *_parse_rule(t)) for t in rule_texts]

    known = set(base)
    derived: list[dict] = []
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        new = []
        for text, conds, heads in parsed:
            for b in _matches(conds, 0, known, {}):
                for _, hpred, hargs in heads:
                    g = _ground(hargs, b)
                    if any(_is_var(a) for a in g):
                        raise ValueError(f"Variable sin ligar en la conclusión de: {text}")
                    if (hpred, g) not in known and all(x["key"] != (hpred, g) for x in new):
                        new.append({"key": (hpred, g), "rule": text, "binding": b, "iteration": iterations})
        if not new:
            break
        for n in new:
            known.add(n["key"])
        derived.extend(new)
    else:
        log.warning("Límite de iteraciones alcanzado")

    out = [
        {"fact": _fact_str(*d["key"]), "rule": d["rule"], "binding": d["binding"], "iteration": d["iteration"]}
        for d in derived
    ]
    result: dict[str, Any] = {
        "derived": out,
        "derived_count": len(out),
        "total_facts": len(known),
        "iterations": iterations,
        "fixed_point": iterations < max_iterations or not derived,
    }
    if query:
        qneg, qpred, qargs = _parse_atom(query)
        answers = []
        for pred, args in sorted(known):
            if pred == qpred:
                b = _unify(qargs, args, {})
                if b is not None:
                    answers.append({"fact": _fact_str(pred, args), "binding": b})
        result["query"] = query
        result["answers"] = answers if not qneg else []
        if qneg:
            result["holds"] = not answers
    return result


@server.tool()
def agix_verify_expression(expression: str) -> dict:
    """Comprueba si una expresión aritmética/booleana en Python es satisfacible.

    Usa el verificador formal de AGIX: prueba todas las asignaciones de variables en {-1, 0, 1}.
    Devuelve 'sat', 'unsat' o 'unknown'. 'unsat' solo es fiable dentro de ese dominio.
    Ej.: "x + y == 2 and x != y".
    """
    with _quiet():
        res = agix_verify(expression)
    return {"expression": expression, "result": res, "domain": [-1, 0, 1],
            "note": "unsat significa sin solución en {-1,0,1}, no en general"}


@server.tool()
def agix_fuzzy_infer(inputs: dict[str, dict], output: dict, rules: list[dict]) -> dict:
    """Inferencia difusa de Mamdani (scikit-fuzzy) con términos bajo/medio/alto.

    inputs: {"riesgo": {"value": 7, "range": [0, 10]}, "beneficio": {"value": 3, "range": [0, 10]}}
    output: {"name": "prioridad", "range": [0, 10]}
    rules: [{"if": {"riesgo": "alto", "beneficio": "bajo"}, "then": "bajo", "op": "and"}, ...]
       op puede ser "and" (por defecto) u "or". Términos válidos: bajo, medio, alto.
    Devuelve el valor desfuzzificado y su término dominante.
    """
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    terms = ["bajo", "medio", "alto"]

    def universe(rng):
        lo, hi = float(rng[0]), float(rng[1])
        return np.linspace(lo, hi, 201)

    def add_terms(var, lo, hi):
        mid = (lo + hi) / 2
        var["bajo"] = fuzz.trimf(var.universe, [lo, lo, mid])
        var["medio"] = fuzz.trimf(var.universe, [lo, mid, hi])
        var["alto"] = fuzz.trimf(var.universe, [mid, hi, hi])

    ants = {}
    for name, spec in inputs.items():
        rng = spec.get("range", [0, 10])
        a = ctrl.Antecedent(universe(rng), name)
        add_terms(a, float(rng[0]), float(rng[1]))
        ants[name] = a
    orng = output.get("range", [0, 10])
    cons = ctrl.Consequent(universe(orng), output["name"])
    add_terms(cons, float(orng[0]), float(orng[1]))

    frules = []
    for r in rules:
        clauses = []
        for var, term in r["if"].items():
            if var not in ants or term not in terms:
                raise ValueError(f"Cláusula inválida {var}={term}; términos: {terms}")
            clauses.append(ants[var][term])
        expr = clauses[0]
        for c in clauses[1:]:
            expr = (expr | c) if r.get("op", "and") == "or" else (expr & c)
        if r["then"] not in terms:
            raise ValueError(f"Término de salida inválido: {r['then']}")
        frules.append(ctrl.Rule(expr, cons[r["then"]]))

    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(frules))
    for name, spec in inputs.items():
        sim.input[name] = float(spec["value"])
    with _quiet():
        sim.compute()
    value = float(sim.output[output["name"]])
    memberships = {t: float(fuzz.interp_membership(cons.universe, cons[t].mf, value)) for t in terms}
    return {"output": output["name"], "value": round(value, 4),
            "term": max(memberships, key=memberships.get), "memberships": _plain(memberships)}


@server.tool()
def agix_nash_equilibria(row_payoffs: list[list[float]], col_payoffs: list[list[float]] | None = None) -> dict:
    """Equilibrios de Nash (enumeración de soportes, nashpy) de un juego bimatricial.

    row_payoffs[i][j]: pago del jugador fila con la estrategia i frente a j.
    col_payoffs: pagos del jugador columna; si se omite, juego de suma cero (-row_payoffs).
    """
    import nashpy

    A = np.array(row_payoffs, dtype=float)
    B = -A if col_payoffs is None else np.array(col_payoffs, dtype=float)
    game = nashpy.Game(A, B)
    eqs = []
    with _quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sr, sc in game.support_enumeration():
            eqs.append({"row_strategy": _plain(sr), "col_strategy": _plain(sc),
                        "payoffs": _plain(game[sr, sc])})
    return {"equilibria": eqs, "count": len(eqs), "zero_sum": col_payoffs is None}


# --------------------------------------------------------------------------- memoria


def _exp_dict(e) -> dict:
    return _plain(e)


@server.tool()
def agix_memory_record(
    situation: str,
    decision: str,
    outcome: str,
    success: bool,
    ethical_judgment: str | None = None,
    emotion: dict[str, float] | None = None,
) -> dict:
    """Registra una experiencia en la memoria persistente de AGIX (memoria.json).

    situation: contexto o problema; decision: qué se hizo; outcome: qué pasó / lección.
    emotion: opcional, p. ej. {"placer": 0.2, "activacion": 0.5, "dominancia": 0.1}.
    """
    with _quiet():
        exp = _memory.registrar(situation, decision, outcome, success,
                                juicio_etico=ethical_judgment, emocion=emotion)
        _memory.guardar(MEMORY_FILE)
    return {"stored": _exp_dict(exp), "total": len(_memory.experiencias)}


def _score(query: str, text: str) -> float:
    q = set(re.findall(r"\w+", query.lower()))
    t = set(re.findall(r"\w+", (text or "").lower()))
    overlap = len(q & t) / len(q) if q else 0.0
    return 0.7 * overlap + 0.3 * SequenceMatcher(None, query.lower(), (text or "").lower()).ratio()


@server.tool()
def agix_memory_search(query: str, n: int = 5, only_success: bool | None = None) -> dict:
    """Busca experiencias pasadas por palabras clave y similitud en todos los campos."""
    scored = []
    for e in _memory.experiencias:
        if only_success is not None and e.exito != only_success:
            continue
        blob = " ".join(str(x) for x in (e.entrada, e.decision, e.resultado, e.juicio_etico or ""))
        s = _score(query, blob)
        if s > 0.15:
            scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {"query": query, "results": [{"score": round(s, 3), **_exp_dict(e)} for s, e in scored[:n]],
            "total_memories": len(_memory.experiencias)}


@server.tool()
def agix_memory_recent(n: int = 10) -> dict:
    """Devuelve las n experiencias más recientes y un resumen de aciertos/errores."""
    exps = _memory.experiencias
    ok = sum(1 for e in exps if e.exito)
    return {"recent": [_exp_dict(e) for e in exps[-n:]], "total": len(exps),
            "success_rate": round(ok / len(exps), 3) if exps else None}


# --------------------------------------------------------------------------- ontología


@server.tool()
def agix_ontology_add(
    concept: str,
    attributes: dict[str, str] | None = None,
    relations: dict[str, list[str]] | None = None,
) -> dict:
    """Añade o amplía un concepto en la ontología persistente.

    relations: {"es_un": ["animal"], "parte_de": ["sistema"]}; crea los conceptos destino si faltan.
    """
    c = _ontology.conceptualize(concept)
    for k, v in (attributes or {}).items():
        c.add_attribute(k, str(v))
    for rel, targets in (relations or {}).items():
        for t in targets:
            _ontology.conceptualize(t)
            if t not in c.relations.get(rel, []):
                _ontology.relate_concepts(concept, t, rel)
    _save_ontology()
    return {"concept": concept, "attributes": c.attributes, "relations": c.relations,
            "total_concepts": len(_ontology.concepts)}


@server.tool()
def agix_ontology_query(
    concept: str | None = None,
    attribute: str | None = None,
    value: str | None = None,
    relation: str | None = None,
    depth: int = 3,
) -> dict:
    """Consulta la ontología.

    - concept: devuelve sus atributos, relaciones salientes y entrantes; con relation,
      recorre esa relación de forma transitiva hasta 'depth' (p. ej. cadena es_un).
    - attribute + value: conceptos con ese atributo.
    - sin argumentos: lista de conceptos.
    """
    if attribute is not None:
        hits = _ontology.find_by_attribute(attribute, value) if value is not None else [
            c for c in _ontology.concepts.values() if attribute in c.attributes]
        return {"matches": [{"concept": c.name, "attributes": c.attributes} for c in hits]}
    if concept is None:
        return {"concepts": sorted(_ontology.concepts), "total": len(_ontology.concepts)}
    c = _ontology.get_concept(concept)
    if c is None:
        return {"error": f"Concepto desconocido: {concept}", "concepts": sorted(_ontology.concepts)}
    incoming = {}
    for other in _ontology.concepts.values():
        for rel, targets in other.relations.items():
            if concept in targets:
                incoming.setdefault(rel, []).append(other.name)
    out: dict[str, Any] = {"concept": concept, "attributes": c.attributes,
                           "relations": c.relations, "incoming": incoming}
    if relation:
        chain, frontier, seen = [], [concept], {concept}
        for level in range(1, depth + 1):
            nxt = []
            for name in frontier:
                node = _ontology.get_concept(name)
                for t in (node.relations.get(relation, []) if node else []):
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
                        chain.append({"concept": t, "level": level})
            frontier = nxt
        out["closure"] = {"relation": relation, "reachable": chain}
        inherited = {}
        for item in chain:  # herencia de atributos por la cadena
            node = _ontology.get_concept(item["concept"])
            for k, v in (node.attributes if node else {}).items():
                inherited.setdefault(k, v)
        out["inherited_attributes"] = {k: v for k, v in inherited.items() if k not in c.attributes}
    return out


# --------------------------------------------------------------------------- ética e intención

_FRAMEWORKS = {
    "eco": lambda cfg: EcoEthics(),
    "utilitarian": lambda cfg: UtilitarianEthics(cfg.get("weights")),
    "deontological": lambda cfg: DeontologicalEthics(cfg.get("rules")),
    "virtue": lambda cfg: VirtueEthics(cfg.get("virtues")),
}


@server.tool()
def agix_ethics_judge(action: dict[str, float], frameworks: list[dict] | None = None) -> dict:
    """Evalúa una acción con uno o varios marcos éticos de AGIX.

    action: puntuaciones 0..1, p. ej. {"pro_vida": 0.8, "no_dano": 0.9, "respeto": 0.7}
        (EcoEthics usa esas tres claves; otros marcos aceptan claves libres).
    frameworks: lista de {"type": "eco|utilitarian|deontological|virtue", ...config}
        utilitarian: {"weights": {clave: peso}}; deontological: {"rules": {clave: minimo}};
        virtue: {"virtues": [claves]}. Por defecto: los cuatro con su configuración básica.
    Etiquetas: justo > 0.85, aceptable > 0.6, cuestionable > 0.4, nocivo.
    """
    specs = frameworks or [{"type": "eco"}, {"type": "utilitarian"},
                           {"type": "deontological", "rules": {k: 0.6 for k in action}},
                           {"type": "virtue"}]
    objs = []
    for s in specs:
        t = s.get("type")
        if t not in _FRAMEWORKS:
            raise ValueError(f"Marco desconocido {t}; opciones: {list(_FRAMEWORKS)}")
        objs.append(_FRAMEWORKS[t](s))
    with _quiet():
        res = AlignmentInterface(objs).judge(action)
    if isinstance(res, tuple):  # con un solo marco AGIX devuelve (score, label)
        res = {objs[0].__class__.__name__: res}
    verdicts = {k: {"score": round(float(v[0]), 4), "label": v[1]} for k, v in res.items()}
    mean = sum(v["score"] for v in verdicts.values()) / len(verdicts)
    return {"verdicts": verdicts, "mean_score": round(mean, 4),
            "worst": min(verdicts.items(), key=lambda kv: kv[1]["score"])[0]}


@server.tool()
def agix_intention_evaluate(
    consequences: dict[str, float] | None = None,
    goals: list[dict] | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Evaluador de intenciones de AGIX (metacognición).

    consequences: {"beneficio": 0..1, "riesgo": 0..1, "cooperacion": 0..1} -> prob. de éxito y decisión.
    goals: [{"nombre": str, "prioridad": 0..10, "incompatibles": [nombres]}] -> nivel de conflicto 0..1.
    history: [{"etica": 0..1, "motivacion": 0..1}] -> veredicto sobre la conducta.
    """
    ev = IntentionEvaluator()
    out: dict[str, Any] = {}
    with _quiet():
        if consequences is not None:
            out["consequences"] = ev.simular_consecuencias(consequences)
        if goals is not None:
            out["goal_conflict"] = ev.evaluar_conflictos(goals)
        if history is not None:
            out["conduct"] = ev.juzgar_conducta(history)
    return _plain(out)


@server.tool()
def agix_meta_evaluate(context: dict, action: str, result: dict) -> dict:
    """Metaevaluación lógica, afectiva y ética (DynamicMetaEvaluator) con historial de sesión.

    context: {"metas": [{"nombre","prioridad","incompatibles"}], "historial": [{"etica","motivacion"}]}
    result: {"exito": bool, "impacto": {"pro_vida","no_dano","respeto" | "beneficio","riesgo","cooperacion"}}
    La coherencia acumula los éxitos de todas las acciones evaluadas en la sesión.
    """
    with _quiet():
        report = _meta.meta_evaluate(context, action, result)
        message = "No sé qué pensar de esta acción" if report["coherence"] is None else (
            "No me siento cómodo con esta acción." if report["coherence"] < 0.4 or report["ethics"] < 0.5
            else "Acción aceptada sin conflictos.")
    return _plain({**report, "message": message})


@server.tool()
def agix_goal_register(description: str, impact: dict[str, float], priority: int = 5) -> dict:
    """Registra una meta en el GoalEngine; se rechaza si su puntuación ética (EcoEthics) < 0.6.

    impact: {"pro_vida", "no_dano", "respeto"} en 0..1. priority: 1 (máxima) .. 10.
    """
    try:
        with _quiet():
            g = _goals.agregar_meta(description, prioridad=priority, impacto=impact)
        accepted, goal, reason = True, _plain(getattr(g, "__dict__", g)), None
    except ValueError as exc:
        accepted, goal, reason = False, None, str(exc)
    with _quiet():
        summary = _goals.resumen_metas()
    return {"accepted": accepted, "goal": goal, "reason": reason, "goals": _plain(summary)}


# --------------------------------------------------------------------------- emoción y qualia


@server.tool()
def agix_emotion_update(
    events: dict[str, float],
    weights: dict[str, list[float]],
    feelings: dict[str, float] | None = None,
) -> dict:
    """Calcula un estado emocional PAD (placer, activación, dominancia) con EmotionSimulator.

    events: intensidad de cada estímulo, p. ej. {"amenaza": 1, "logro": 0.5}.
    weights: efecto PAD de cada estímulo, p. ej. {"amenaza": [-0.4, 0.6, -0.3], "logro": [0.5, 0.3, 0.4]}.
    feelings: emociones previas con intensidad, p. ej. {"miedo": 0.7}.
    """
    pesos = {k: PADState(*map(float, v)) for k, v in weights.items()}
    sim = EmotionSimulator(pesos=pesos)
    qualia = EmotionalState()
    with _quiet():
        for k, v in (feelings or {}).items():
            qualia.sentir(k, float(v))
        sim.actualizar(events, {}, qualia)
        st = sim.estado()
    return {"pad": {"placer": round(st.placer, 4), "activacion": round(st.activacion, 4),
                    "dominancia": round(st.dominancia, 4)},
            "tone": qualia.tono_general() if feelings else None}


@server.tool()
def agix_qualia_experience(event: str, charge: float, emotion: str) -> dict:
    """Hace que el espíritu Qualia de la sesión viva una experiencia y reflexione.

    charge: intensidad 0..1; emotion: tipo (alegría, miedo, curiosidad, tristeza...).
    """
    with _quiet():
        _spirit.experimentar(event, float(charge), emotion)
        reflection = _spirit.reflexionar()
    return {"reflection": reflection, "emotions": _plain(_spirit.estado_emocional.emociones),
            "tone": _spirit.estado_emocional.tono_general()}


@server.tool()
def agix_qualia_state() -> dict:
    """Estado actual del espíritu Qualia: emociones, tono, diario y metaevaluación."""
    with _quiet():
        diario = _spirit.diario()
        meta = _spirit.metaevaluar()
    return _plain({"emotions": _spirit.estado_emocional.emociones,
                   "tone": _spirit.estado_emocional.tono_general(),
                   "diary": diario, "meta": meta})


# --------------------------------------------------------------------------- métricas y simulación


@server.tool()
def agix_fagi_index(
    task_scores: list[float],
    adaptability_scores: list[float],
    transfer_scores: list[float],
    weights: list[float] | None = None,
) -> dict:
    """Índice de generalidad fAGI de AGIX a partir de puntuaciones 0..1."""
    with _quiet():
        v = compute_fagi_index(task_scores, adaptability_scores, transfer_scores, weights)
    return {"fagi_index": round(float(v), 4)}


@server.tool()
def agix_harmony(goals_vs_actions: float, emotions_vs_actions: float, identity_vs_actions: float) -> dict:
    """Armonía interna (0..1) entre metas, emociones, identidad y acciones, con diagnóstico."""
    data = {"metas_vs_acciones": goals_vs_actions, "emociones_vs_acciones": emotions_vs_actions,
            "identidad_vs_acciones": identity_vs_actions}
    h = HarmonyEvaluator()
    with _quiet():
        score = h.evaluar(data)
        diag = h.diagnostico(data)
    return _plain({"score": score, "diagnosis": diag})


@server.tool()
def agix_simulate_agent(observations: int = 10, actions: int = 4, episodes: int = 3,
                        max_steps: int = 5, seed: int | None = None) -> dict:
    """Simula un GeneticAgent de AGIX en ToyEnvironment con AutoAgent y devuelve sus métricas."""
    import random

    from agix.agents.genetic import GeneticAgent
    from agix.autoagent import AutoAgent
    from agix.environments.toy_env import ToyEnvironment
    from agix.evaluation import evaluar

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    with _quiet():
        agent = AutoAgent(GeneticAgent(actions), ToyEnvironment(observations), evaluator=evaluar)
        metrics = agent.run(episodes=episodes, max_steps=max_steps)
    return _plain({"metrics": metrics, "episodes": episodes, "max_steps": max_steps})


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
