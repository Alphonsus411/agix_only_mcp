"""Servidor MCP que expone AGIX como herramientas de apoyo al razonamiento.

Transporte: stdio. Nada debe escribirse en stdout salvo el protocolo MCP, por eso
el logging va a stderr y cada llamada a AGIX se envuelve en redirect_stdout.

Estado persistente en la carpeta que elige memory_scope.py (por defecto ~/.agix-mcp; por
proyecto con un .agix.json; AGIX_MCP_DATA la fija a mano):
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

# Memoria por proyecto (memory_scope.py): se decide con la carpeta desde la que Claude Code
# lanza el servidor, antes del chdir. Con la memoria desactivada (.agix.json {"memoria": "off"})
# se usa una carpeta temporal que se borra al salir: nada se lee ni se conserva.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import curation  # noqa: E402
import memory_scope  # noqa: E402
import storage  # noqa: E402

START_DIR = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
SCOPE = memory_scope.resolve(START_DIR)
if SCOPE.enabled:
    DATA_DIR = SCOPE.data_dir.resolve()
else:
    import tempfile

    _TMP = tempfile.TemporaryDirectory(prefix="agix-off-", ignore_cleanup_errors=True)
    DATA_DIR = Path(_TMP.name).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
# Varias sesiones comparten la carpeta (storage.py): copia diaria antes de tocar nada.
if SCOPE.enabled:
    with storage.lock(DATA_DIR):
        storage.daily_backup(DATA_DIR)
# Retiradas y fusiones de lecciones (curation.py): marcas aparte, AGIX no se entera.
storage.read_json(DATA_DIR / curation.FILE)  # si está corrupto, se aparta (no se sobrescribe)
_cur = curation.load(DATA_DIR)


def _save_cur() -> None:
    curation.save(DATA_DIR, _cur)
    _seen[DATA_DIR / curation.FILE] = storage.mtime(DATA_DIR / curation.FILE)
# GestorDeMemoria solo acepta rutas dentro del cwd.
os.chdir(DATA_DIR)
MEMORY_FILE = "memoria.json"
ONTOLOGY_FILE = DATA_DIR / "ontologia.json"

with contextlib.redirect_stdout(sys.stderr):
    import agix.memory  # noqa: F401  (evita la importación circular de reasoning)
    import numpy as np
    from agix.memory import EpisodicMemory, GestorDeMemoria  # noqa: F401
    from agix.autonarrative import AutonarrativeCore, Experience
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
    from agix.metacognition.manager import MetacognitionManager
    from agix.metacognition.decision_reviewer import DecisionReviewer
    from agix.goals.goal_engine import GoalEngine
    from agix.emotion.emotion_simulator import EmotionSimulator, PADState
    from agix.qualia.qualia_core import EmotionalState
    from agix.qualia.spirit import QualiaSpirit
    from agix.qualia.neuro_plastic import EmotionalPlasticity
    from agix.qualia.emotional_rnn import EmotionalRNN
    from agix.qualia.qualia_spirit_core import QualiaSpiritCore
    from agix.qualia.aesthetic_engine import AestheticEngine
    from agix.qualia.dream_bank import DreamBank
    from agix.identity.personality import PersonalityProfile
    from agix.identity.identity_core import SyntheticIdentity
    from agix.learning.plasticity import HebbianPlasticity
    from agix.perception.sensor_map import Sensor, SensorMap
    with warnings.catch_warnings():  # el paquete orchestrator importa el hub y avisa del token
        warnings.simplefilter("ignore")
        from agix.orchestrator.synthetic_loop import SyntheticLoop

from mcp.server.mcpserver import MCPServer

INSTRUCTIONS = """\
Herramientas de AGIX (framework de Adolfo González) para apoyar el razonamiento.
- agix_logic_infer: encadenamiento hacia delante con variables; úsalo para comprobar
  deducciones (transitividad, reglas de negocio, dependencias) en vez de hacerlas de cabeza.
- agix_verify_expression: satisfacibilidad de expresiones (dominio pequeño, por defecto -1..1).
- agix_fuzzy_infer: decisiones con lógica difusa (riesgo, prioridad, confianza).
- agix_nash_equilibria: equilibrios de Nash de juegos bimatriciales.
- agix_memory_*: memoria persistente de experiencias entre sesiones (decisión, resultado, lección).
- agix_memory_timeline / agix_memory_narrative: línea temporal y resumen narrativo de los episodios.
- agix_narrative_*: autonarrativa persistente (experiencias con valores, metas y afectos), modelo de
  sí mismo y evaluación de si una acción encaja con esa identidad.
- agix_decision_record / agix_decision_review / agix_metacognition_state: metacognición persistente;
  registra decisiones con su resultado y lección, revisa aciertos/errores y detecta errores repetidos.
- agix_resolve_internal_conflict: delibera entre hipótesis según su peso y el resultado y el tono de
  los episodios que las respaldan (un recuerdo de fracaso resta).
- agix_identity_define / agix_identity_event: identidad sintética persistente (nombre, origen,
  atributos, rasgos, hitos). agix_identity_portrait: autorretrato que une identidad, Qualia,
  autonarrativa y metacognición.
- agix_ontology_*: grafo persistente de conceptos, atributos y relaciones.
- agix_ethics_judge / agix_intention_evaluate / agix_meta_evaluate / agix_goal_register:
  evaluación ética e intencional; requieren que traduzcas el plan a puntuaciones 0..1. Incluye el
  riesgo ("risk") cuando lo haya: penaliza; y un no_dano < 0.3 veta la acción aunque el resto sea alto.
- agix_emotion_update: cálculo puntual de un estado PAD.
- agix_qualia_experience / _state / _configure: espíritu Qualia persistente entre sesiones, con
  personalidad, atenuación, plasticidad hebbiana, RNN que predice el siguiente estado y modos
  (euforia/desconexión).
- agix_cognitive_cycle: ciclo completo percepción → emoción → ética → control → reflexión;
  aborta acciones con ética < 0.6 (con veto por daño y penalización por riesgo).
- agix_aesthetic_decide: elegir entre opciones por armonía; agix_dream_record / agix_dreams_recent:
  banco de sueños persistente.
- agix_fagi_index, agix_harmony, agix_simulate_agent: métricas y simulación de agentes.
Las puntuaciones éticas/afectivas son heurísticas: tómalas como una segunda opinión, no como verdad.
- agix_memory_scope: qué memoria usa este proyecto (global, propia del proyecto o desactivada).
- agix_memory_retire / agix_memory_duplicates / agix_memory_merge / agix_lesson_correct: curación
  sin borrar nada; retira lo obsoleto (con motivo), fusiona duplicados (revisa cada grupo) y
  corrige el texto de una lección errónea o no verificada.
"""
INSTRUCTIONS += f"Memoria activa en esta sesión: {SCOPE.describe()}."
if not SCOPE.enabled:
    INSTRUCTIONS += (" Lo que registres se pierde al cerrar la sesión: no uses agix_*_record"
                     " salvo que el usuario lo pida.")

server = MCPServer(name="agix", version="1.0.0", instructions=INSTRUCTIONS)

# Cada llamada a una herramienta se ejecuta con el bloqueo de la carpeta de datos y tras
# recargar lo que otras sesiones hayan escrito (_sync): así ninguna sesión pisa los registros
# de otra. Las herramientas tardan como mucho un par de segundos (ver límites de entrada).
import functools as _functools  # noqa: E402

_register_tool = server.tool


def _tool(*args, **kwargs):
    register = _register_tool(*args, **kwargs)

    def decorate(fn):
        @_functools.wraps(fn)
        def call(*a, **kw):
            with storage.lock(DATA_DIR):
                _sync()
                return fn(*a, **kw)

        register(call)
        return call

    return decorate


server.tool = _tool

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
    if hasattr(obj, "isoformat"):  # datetime / date
        return obj.isoformat()
    return str(obj)


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(sys.stderr):
        yield


# --------------------------------------------------------------------------- estado

# Marcas de tiempo: todas en UTC con la zona explícita ("...+00:00"). AGIX mezclaba UTC sin zona
# (episodios, decisiones) con hora local sin zona (sueños, metas): el mismo minuto real salía
# como 08:05 en la memoria y 09:06 en los sueños.
from datetime import datetime as _datetime, timezone as _timezone  # noqa: E402


_last_stamp: _datetime | None = None


def _see_stamp(value) -> None:
    """Las marcas nuevas serán posteriores a esta (también a las escritas por otras sesiones)."""
    global _last_stamp
    try:
        dt = _to_utc(value)
    except (TypeError, ValueError):
        return
    if _last_stamp is None or dt > _last_stamp:
        _last_stamp = dt


def _utc_now_iso() -> str:
    """Hora UTC estrictamente creciente: el reloj de Windows avanza a saltos y dos registros
    seguidos (de esta sesión o de otra) podían compartir marca de tiempo, que es también su id
    (curation.py)."""
    global _last_stamp
    from datetime import timedelta

    now = _datetime.now(_timezone.utc)
    if _last_stamp is not None and now <= _last_stamp:
        now = _last_stamp + timedelta(microseconds=1)
    _last_stamp = now
    return now.isoformat()


def _to_utc(value, naive_is_local: bool = False) -> _datetime:
    """datetime o ISO → datetime en UTC. Sin zona: UTC (lo que AGIX guardaba en memoria y
    decisiones) o, con naive_is_local, la hora local de esta máquina (sueños antiguos)."""
    dt = value if isinstance(value, _datetime) else _datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.astimezone(_timezone.utc) if naive_is_local else dt.replace(tzinfo=_timezone.utc)
    return dt.astimezone(_timezone.utc)


def _to_utc_iso(value, naive_is_local: bool = False):
    try:
        return _to_utc(value, naive_is_local).isoformat()
    except (TypeError, ValueError):
        return value  # ilegible: se deja como estaba


# Memoria episódica: registro completo (memoria.json, formato de GestorDeMemoria) + árbol
# narrativo resumido. Se lee y escribe aquí (no con cargar/guardar de AGIX) para que la escritura
# sea atómica y un archivo corrupto se aparte en vez de sobrescribirse con una memoria vacía.
from dataclasses import asdict as _asdict  # noqa: E402

from agix.memory.experiential import Experiencia as _Experiencia  # noqa: E402

MEMORY_PATH = DATA_DIR / MEMORY_FILE


def _episode_fragment(e) -> str:
    return f"{e.timestamp}: {e.entrada} -> {e.resultado}"


def _load_memory() -> None:
    global _memory
    mem = EpisodicMemory()
    data = storage.read_json(MEMORY_PATH, {}) or {}
    for d in data.get("experiencias", []) if isinstance(data, dict) else []:
        try:
            e = _Experiencia(**d)
        except TypeError as exc:
            log.warning("Episodio ilegible, se omite: %s", exc)
            continue
        if e.timestamp:  # episodios antiguos: UTC sin zona → UTC explícito
            e.timestamp = _to_utc_iso(e.timestamp)
            _see_stamp(e.timestamp)
        mem.experiencias.append(e)
    mem.dialogos = list(data.get("dialogos", [])) if isinstance(data, dict) else []
    # El árbol narrativo no se persiste en AGIX: se reconstruye reinsertando los episodios.
    for e in mem.experiencias:
        mem.insert(_episode_fragment(e))
    _memory = mem


def _save_memory() -> None:
    data = {"experiencias": [{**_asdict(e), "emocion": e.emocion if not hasattr(e.emocion, "__dataclass_fields__")
                              else _asdict(e.emocion)} for e in _memory.experiencias],
            "dialogos": _memory.dialogos}
    storage.write_json(MEMORY_PATH, data)
    _seen[MEMORY_PATH] = storage.mtime(MEMORY_PATH)


_seen: dict[Path, int | None] = {}  # mtime de cada archivo la última vez que esta sesión lo leyó
_load_memory()
_seen[MEMORY_PATH] = storage.mtime(MEMORY_PATH)

# Autonarrativa: experiencias en SQLite + SelfModel (valores, metas, afectos) derivado de ellas.
NARRATIVE_DB = DATA_DIR / "autonarrativa.db"
NARRATIVE_SCORES = DATA_DIR / "autonarrativa_scores.json"
_narrative = AutonarrativeCore(str(NARRATIVE_DB))
# AGIX abre la conexión en el hilo de importación; las herramientas pueden ejecutarse en otros hilos.
import sqlite3 as _sqlite3  # noqa: E402
import threading as _threading  # noqa: E402

_narrative.conn.close()
_narrative.conn = _sqlite3.connect(str(NARRATIVE_DB), check_same_thread=False)
_narrative_lock = _threading.Lock()


def _replay_narrative_self_model() -> None:
    """AGIX no persiste el SelfModel: se reconstruye a partir de la base de datos."""
    for _text, meta in _narrative.conn.execute("SELECT text, metadata FROM experiences ORDER BY id"):
        _narrative.self_model.update_self_model(Experience(_text, json.loads(meta)))
    scores = storage.read_json(NARRATIVE_SCORES)
    if isinstance(scores, dict):
        _narrative.self_model.planning_score = scores.get("planning_score", 0.5)
        _narrative.self_model.motivation_score = scores.get("motivation_score", 0.5)


def _save_narrative_scores() -> None:
    sm = _narrative.self_model
    storage.write_json(NARRATIVE_SCORES, {"planning_score": sm.planning_score,
                                          "motivation_score": sm.motivation_score}, indent=None)


try:
    _replay_narrative_self_model()
except Exception as exc:
    log.warning("No se pudo reconstruir el modelo autonarrativo: %s", exc)

_ontology = Ontology()


def _load_ontology() -> None:
    global _ontology
    _ontology = Ontology()
    data = storage.read_json(ONTOLOGY_FILE, {})
    if not isinstance(data, dict):
        return
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
    storage.write_json(ONTOLOGY_FILE, data)
    _seen[ONTOLOGY_FILE] = storage.mtime(ONTOLOGY_FILE)


try:
    _load_ontology()
except Exception as exc:
    log.warning("No se pudo cargar la ontología: %s", exc)
_seen[ONTOLOGY_FILE] = storage.mtime(ONTOLOGY_FILE)

# --- Espíritu Qualia persistente --------------------------------------------
QUALIA_CONFIG = DATA_DIR / "qualia_config.json"
QUALIA_STATE = "qualia_estado.json"  # relativo al cwd (DATA_DIR): EpisodicMemory exige rutas dentro del cwd
QUALIA_MODELS = DATA_DIR / "qualia_modelos.json"
DREAMS_FILE = "suenos.json"

DEFAULT_QUALIA_CONFIG = {
    "nombre": "Claude",
    "optimismo": 0.6,
    "introversion": 0.4,
    "energia": 0.6,
    "atenuacion": 0.9,
    "plasticidad": True,
    "modos": "balance",  # "balance" (experimental) o "suma" (criterio original de AGIX)
}

# --- Modos por balance afectivo (experimental) -------------------------------
# El criterio original de AGIX compara SUMAS absolutas (total > 1.5 → euforia; tristeza+miedo
# > 0.8 → desconexión): con varias emociones activas la euforia salta casi siempre, sin mirar
# su signo, y el espíritu oscila entre extremos. Aquí se usan:
#   balance  = (positivas − negativas) / (positivas + negativas)   ∈ [-1, 1]
#   activación = intensidad media de las emociones activas          ∈ [0, 1]
# con histéresis (umbral de entrada más exigente que el de salida) para no saltar de modo
# en cada experiencia. Las emociones sin valencia conocida (sorpresa...) solo cuentan en la
# activación.
POSITIVE_EMOTIONS = {"alegria", "curiosidad", "gozo", "serenidad", "satisfaccion", "entusiasmo",
                     "alivio", "amor", "gratitud", "orgullo", "calma", "confianza", "esperanza"}
NEGATIVE_EMOTIONS = {"tristeza", "miedo", "enojo", "ira", "frustracion", "culpa", "ansiedad",
                     "remordimiento", "verguenza", "asco", "desesperanza"}
MODE_THRESHOLDS = {
    "euforia_entrada": (0.5, 0.5),   # balance > 0.5 y activación > 0.5
    "euforia_salida": (0.3, 0.4),    # sale si balance < 0.3 o activación < 0.4
    "desconexion_entrada": -0.4,     # balance < -0.4 (o activación < 0.1: apatía)
    "desconexion_salida": -0.2,      # sale si balance > -0.2 y activación >= 0.15
}


def _strip_accents(text: str) -> str:
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def affect_balance(emotions: dict[str, float]) -> tuple[float, float]:
    """(balance, activación) de un conjunto de emociones."""
    active = {k: v for k, v in emotions.items() if v > 0.05}
    if not active:
        return 0.0, 0.0
    pos = sum(v for k, v in active.items() if _strip_accents(k) in POSITIVE_EMOTIONS)
    neg = sum(v for k, v in active.items() if _strip_accents(k) in NEGATIVE_EMOTIONS)
    balance = (pos - neg) / (pos + neg) if pos + neg > 0 else 0.0
    return balance, sum(active.values()) / len(active)


class BalancedQualiaSpirit(QualiaSpirit):
    """QualiaSpirit cuyos modos dependen del balance afectivo, con histéresis."""

    def experimentar(self, evento: str, carga: float, tipo_emocion="sorpresa"):
        was_euphoric, was_disconnected = self.modo_euforia, self.modo_desconexion
        super().experimentar(evento, carga, tipo_emocion)  # aplica también el criterio original
        balance, arousal = affect_balance(self.estado_emocional.emociones)
        t = MODE_THRESHOLDS
        if was_disconnected:
            disconnected = not (balance > t["desconexion_salida"] and arousal >= 0.15)
        else:
            disconnected = balance < t["desconexion_entrada"] or arousal < 0.1
        if disconnected:
            euphoric = False
        elif was_euphoric:
            euphoric = balance >= t["euforia_salida"][0] and arousal >= t["euforia_salida"][1]
        else:
            euphoric = balance > t["euforia_entrada"][0] and arousal > t["euforia_entrada"][1]
        self.modo_desconexion, self.modo_euforia = disconnected, euphoric
        self.ultimo_balance = (balance, arousal)

_RNN_PARAMS = ("Wxh", "Whh", "bh", "Wf", "Wi", "Wc", "Wo", "bf", "bi", "bc", "bo", "Why", "by")


def _load_qualia_config() -> dict:
    cfg = dict(DEFAULT_QUALIA_CONFIG)
    data = storage.read_json(QUALIA_CONFIG)
    if isinstance(data, dict):
        cfg.update(data)
    return cfg


def _new_spirit(cfg: dict) -> QualiaSpirit:
    personality = PersonalityProfile(cfg["optimismo"], cfg["introversion"], cfg["energia"])
    cls = BalancedQualiaSpirit if cfg.get("modos", "balance") == "balance" else QualiaSpirit
    return cls(nombre=cfg["nombre"], plasticidad=bool(cfg["plasticidad"]),
               personality=personality, atenuacion=float(cfg["atenuacion"]))


def _save_models(sp: QualiaSpirit) -> None:
    data: dict[str, Any] = {"modos": {"euforia": sp.modo_euforia, "desconexion": sp.modo_desconexion}}
    if sp.plasticidad is not None and sp.plasticidad.model is not None:
        data["plasticidad"] = {"order": sp.plasticidad.emotion_order,
                               "weights": sp.plasticidad.model.weights.tolist()}
    if sp.predictor is not None:
        p = sp.predictor
        data["rnn"] = {
            "order": sp.emotion_order, "type": p.rnn_type, "hidden": p.hidden_size,
            "params": {n: getattr(p, n).tolist() for n in _RNN_PARAMS if hasattr(p, n)},
            "history": [v.tolist() for v in sp._historial_vectores],
        }
    storage.write_json(QUALIA_MODELS, data, indent=None)


def _load_models(sp: QualiaSpirit) -> None:
    data = storage.read_json(QUALIA_MODELS)
    if not isinstance(data, dict):
        return
    modes = data.get("modos") or {}
    sp.modo_euforia = bool(modes.get("euforia", False))
    sp.modo_desconexion = bool(modes.get("desconexion", False))
    if "plasticidad" in data and sp.plasticidad is not None:
        pl = data["plasticidad"]
        n = len(pl["order"])
        sp.plasticidad.emotion_order = list(pl["order"])
        sp.plasticidad.model = HebbianPlasticity(n, n, learning_rate=sp.plasticidad.learning_rate)
        sp.plasticidad.model.weights = np.array(pl["weights"], dtype=float)
    if "rnn" in data:
        r = data["rnn"]
        rnn = EmotionalRNN(len(r["order"]), hidden_size=r["hidden"], rnn_type=r["type"])
        for name, value in r["params"].items():
            setattr(rnn, name, np.array(value, dtype=float))
        sp.predictor = rnn
        sp.emotion_order = list(r["order"])
        sp._historial_vectores = [np.array(v, dtype=float) for v in r["history"]]


def _save_spirit() -> None:
    try:
        # AGIX escribe directamente sobre el archivo: se le hace escribir un temporal y se
        # sustituye de golpe, para que un corte no deje el estado a medias.
        tmp = f"{QUALIA_STATE}.{os.getpid()}.tmp"
        with _quiet():
            _spirit.guardar_estado(tmp)
        storage.write_text(DATA_DIR / QUALIA_STATE, (DATA_DIR / tmp).read_text(encoding="utf-8"))
        (DATA_DIR / tmp).unlink(missing_ok=True)
        _save_models(_spirit)
    except Exception as exc:  # la persistencia no debe romper la herramienta
        log.warning("No se pudo guardar el estado de Qualia: %s", exc)


def _load_spirit() -> QualiaSpirit:
    cfg = _load_qualia_config()
    sp = _new_spirit(cfg)
    if Path(QUALIA_STATE).exists() and storage.read_json(DATA_DIR / QUALIA_STATE) is not None:
        try:
            with _quiet():
                sp.cargar_estado(QUALIA_STATE)
            _load_models(sp)
        except Exception as exc:
            log.warning("No se pudo restaurar el estado de Qualia: %s", exc)
    return sp


_spirit = _load_spirit()
_meta = DynamicMetaEvaluator(spirit=_spirit)
_goals = GoalEngine()
_dreams = DreamBank()
DREAMS_PATH = DATA_DIR / DREAMS_FILE


def _save_dreams() -> None:
    storage.write_json(DREAMS_PATH, _dreams.suenos)
    _seen[DREAMS_PATH] = storage.mtime(DREAMS_PATH)


def _load_dreams() -> None:
    data = storage.read_json(DREAMS_PATH, [])
    _dreams.suenos = list(data) if isinstance(data, list) else []
    # DreamBank guardaba la hora local sin zona: se convierte a UTC explícito y se persiste
    migrated = False
    for s in _dreams.suenos:
        new = _to_utc_iso(s.get("timestamp"), naive_is_local=True)
        if new != s.get("timestamp"):
            s["timestamp"], migrated = new, True
    if migrated:
        _save_dreams()


try:
    with storage.lock(DATA_DIR):
        _load_dreams()
except Exception as exc:
    log.warning("No se pudo cargar el banco de sueños: %s", exc)
_seen[DREAMS_PATH] = storage.mtime(DREAMS_PATH)
_loop: SyntheticLoop | None = None

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
def agix_memory_scope() -> dict:
    """Qué memoria de AGIX usa esta sesión y por qué, sus copias de seguridad diarias (backups/)
    y los problemas de almacenamiento detectados (p. ej. un archivo corrupto apartado).

    Cada proyecto puede tener memoria propia con un archivo .agix.json en su carpeta (o en una
    superior): {"memoria": "nombre"} usa ~/.agix-mcp/proyectos/nombre, {"memoria": "global"} la
    global y {"memoria": "off"} ninguna. Sin archivo, memoria propia de la raíz del proyecto
    (carpeta con .git) en ~/.agix-mcp/proyectos/. AGIX_MCP_DATA manda sobre todo.
    """
    backups = DATA_DIR / storage.BACKUP_DIR
    return {
        "memory": SCOPE.name,
        "description": SCOPE.describe(),
        "persistent": SCOPE.enabled,
        "data_dir": str(DATA_DIR) if SCOPE.enabled else None,
        "decided_by": SCOPE.source,
        "problem": SCOPE.problem,
        "project_dir": START_DIR,
        "backups": sorted(d.name for d in backups.iterdir() if d.is_dir()) if backups.exists() else [],
        "storage_problems": list(storage.problems),
    }


# Límites de agix_verify_expression: el verificador de AGIX prueba 3^n asignaciones y calcula
# potencias sin tope, así que una expresión larga, con muchas variables o con una torre de
# potencias (9**9**9) dejaba el servidor colgado.
VERIFY_MAX_LENGTH = 500
VERIFY_MAX_VARIABLES = 8  # 3^8 = 6561 asignaciones
VERIFY_MAX_POWERS = 3
VERIFY_MAX_EXPONENT = 16
# Mismo criterio que el verificador de AGIX para contar variables.
_VERIFY_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _check_verify_limits(expression: str) -> None:
    import ast

    if len(expression) > VERIFY_MAX_LENGTH:
        raise ValueError(f"Expresión demasiado larga ({len(expression)} caracteres; "
                         f"máximo {VERIFY_MAX_LENGTH})")
    names = {v for v in _VERIFY_NAME.findall(expression)
             if not re.fullmatch(r"True|False|and|or|not", v)}
    if len(names) > VERIFY_MAX_VARIABLES:
        raise ValueError(f"Demasiadas variables ({len(names)}; máximo {VERIFY_MAX_VARIABLES}): "
                         f"se probarían 3^{len(names)} asignaciones")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return  # el verificador de AGIX devuelve 'unknown'

    def small_exponent(node: ast.AST) -> bool:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            node = node.operand
        if isinstance(node, ast.Name):  # una variable vale -1, 0 o 1
            return True
        return (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                and abs(node.value) <= VERIFY_MAX_EXPONENT)

    powers = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow)]
    if len(powers) > VERIFY_MAX_POWERS:
        raise ValueError(f"Demasiadas potencias ({len(powers)}; máximo {VERIFY_MAX_POWERS})")
    if not all(small_exponent(p.right) for p in powers):
        raise ValueError("El exponente de una potencia debe ser una variable o un número de valor "
                         f"absoluto ≤ {VERIFY_MAX_EXPONENT} (no otra potencia ni una operación)")


@server.tool()
def agix_verify_expression(expression: str) -> dict:
    """Comprueba si una expresión aritmética/booleana en Python es satisfacible.

    Usa el verificador formal de AGIX: prueba todas las asignaciones de variables en {-1, 0, 1}.
    Devuelve 'sat', 'unsat' o 'unknown'. 'unsat' solo es fiable dentro de ese dominio.
    Ej.: "x + y == 2 and x != y".
    Límites (para que no se cuelgue): 500 caracteres, 8 variables, 3 potencias y exponentes que
    sean una variable o un número de valor absoluto ≤ 16.
    """
    _check_verify_limits(expression)
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
    d = _plain(e)
    eid = curation.episode_id(e.timestamp)
    d["id"] = eid
    n = curation.repeats(_cur, eid)
    if n > 1:
        d["repeats"] = n
    if curation.is_retired(_cur, eid):
        d["retired"] = _cur["retired"][eid]
    return d


def _episode_active(e) -> bool:
    return not curation.is_retired(_cur, curation.episode_id(e.timestamp))


@server.tool()
def agix_memory_record(
    situation: str,
    decision: str,
    outcome: str,
    success: bool,
    ethical_judgment: str | None = None,
    emotion: dict[str, float] | None = None,
) -> dict:
    """Registra un episodio en la memoria episódica persistente de AGIX (memoria.json).

    situation: contexto o problema; decision: qué se hizo; outcome: qué pasó / lección.
    emotion: opcional, p. ej. {"placer": 0.2, "activacion": 0.5, "dominancia": 0.1}.
    Además del registro completo, añade un fragmento al árbol narrativo (agix_memory_narrative)
    y queda disponible en la línea temporal (agix_memory_timeline).
    """
    with _quiet():
        exp = _memory.store_episode(situation, decision, outcome, success,
                                    juicio_etico=ethical_judgment, timestamp=_utc_now_iso(),
                                    emocion=emotion)
    _save_memory()
    return {"stored": _exp_dict(exp), "total": len(_memory.experiencias)}


_STOPWORDS = {
    # español (sin tildes, como las deja _norm)
    "que", "los", "las", "del", "una", "uno", "unos", "unas", "con", "por", "para", "como", "mas",
    "pero", "sin", "sobre", "entre", "hasta", "desde", "este", "esta", "estos", "estas", "ese",
    "esa", "eso", "esto", "aquel", "sus", "nos", "les", "muy", "tambien", "cuando", "donde",
    "porque", "aunque", "otra", "otro", "otras", "otros", "todo", "toda", "todos", "todas", "hay",
    "ser", "estar", "haber", "tener", "hacer", "vez", "ya", "solo", "bien", "asi", "cada", "mismo",
    # inglés
    "the", "and", "for", "with", "that", "this", "from", "but", "not", "are", "was", "into", "than",
}


def _norm(text: str) -> str:
    """Minúsculas sin tildes, para comparar palabras con y sin acento."""
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _score(query: str, text: str) -> float:
    qn, tn = _norm(query), _norm(text)
    q = set(re.findall(r"\w+", qn))
    t = set(re.findall(r"\w+", tn))
    overlap = len(q & t) / len(q) if q else 0.0
    return 0.7 * overlap + 0.3 * SequenceMatcher(None, qn, tn).ratio()


@server.tool()
def agix_memory_search(query: str, n: int = 5, only_success: bool | None = None) -> dict:
    """Busca experiencias pasadas por palabras clave y similitud en todos los campos.

    No devuelve los episodios retirados (agix_memory_retire) ni los fusionados en otro.
    """
    scored = []
    for e in _memory.experiencias:
        if not _episode_active(e):
            continue
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
    """Devuelve las n experiencias más recientes (sin las retiradas) y el balance de aciertos
    de todo el historial."""
    exps = _memory.experiencias
    ok = sum(1 for e in exps if e.exito)
    active = [e for e in exps if _episode_active(e)]
    return {"recent": [_exp_dict(e) for e in active[-n:]], "total": len(exps),
            "success_rate": round(ok / len(exps), 3) if exps else None}


def _parse_when(value: str, end: bool = False):
    """Acepta 'YYYY-MM-DD' o ISO completo (sin zona = UTC). Una fecha sola como fin abarca todo
    ese día."""
    from datetime import timedelta

    dt = _to_utc(value)
    if end and len(value) == 10:
        dt = dt + timedelta(days=1) - timedelta(microseconds=1)
    return dt


@server.tool()
def agix_memory_timeline(start: str, end: str | None = None, n: int = 50,
                         only_success: bool | None = None, include_retired: bool = False) -> dict:
    """Episodios entre dos fechas en orden cronológico (memoria episódica).

    start / end: 'YYYY-MM-DD' o ISO 'YYYY-MM-DDTHH:MM:SS[+HH:MM]'; sin zona se entienden en UTC,
        que es la zona de todas las marcas de tiempo.
    include_retired: incluir los episodios retirados o fusionados (llevan la clave "retired").
    Devuelve como mucho los n episodios más recientes del intervalo y el balance de aciertos.
    """
    lo = _parse_when(start)
    hi = _parse_when(end, end=True) if end else _datetime.max.replace(tzinfo=_timezone.utc)
    events = []
    for e in _memory.experiencias:
        if not include_retired and not _episode_active(e):
            continue
        try:
            ts = _to_utc(e.timestamp)
        except (TypeError, ValueError):
            continue
        if lo <= ts <= hi and (only_success is None or e.exito == only_success):
            events.append((ts, e))
    events.sort(key=lambda x: x[0])
    picked = events[-n:]
    ok = sum(1 for _, e in events if e.exito)
    return {"start": start, "end": end, "count": len(events),
            "success_rate": round(ok / len(events), 3) if events else None,
            "episodes": [_exp_dict(e) for _, e in picked], "truncated": len(events) > len(picked)}


@server.tool()
def agix_memory_narrative() -> dict:
    """Resumen narrativo jerárquico de la memoria episódica (NarrativeMemoryTree de AGIX).

    Es un resumen con pérdida: conserva los primeros episodios y los más recientes y va
    olvidando los intermedios; el árbol crece en profundidad al llenarse. El registro completo
    está en agix_memory_timeline / agix_memory_search.
    """
    tree = _memory.tree
    return {"narrative": _memory.retrieve(), "nodes": tree.node_count() - 1,
            "capacity": tree.capacity(), "load_factor": round(tree.load_factor(), 3),
            "depth": tree.D, "branching": tree.K, "episodes_total": len(_memory.experiencias)}


# --------------------------------------------------------------------------- autonarrativa


def _as_list(value) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else [str(v) for v in value]


@server.tool()
def agix_narrative_store(
    text: str,
    values: list[str] | None = None,
    goals: list[str] | None = None,
    affects: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Añade una experiencia a la autonarrativa (AutonarrativeCore, SQLite persistente).

    text: la experiencia contada en primera persona o como hecho.
    values / goals / affects: palabras clave (una palabra cada una, p. ej. "calidad", "estabilidad",
        "alivio"); alimentan el modelo de sí mismo que usa agix_narrative_evaluate_action.
    metadata: datos libres adicionales, filtrables por igualdad en agix_narrative_query.
    """
    meta = dict(metadata or {})
    meta.update({"values": _as_list(values), "goals": _as_list(goals),
                 "affective_states": _as_list(affects)})
    with _quiet(), _narrative_lock:
        exp_id = _narrative.store_experience(Experience(text, meta))
    return {"id": exp_id, "self_model": _narrative_self_summary(top=5)}


@server.tool()
def agix_narrative_query(query: str, top: int = 5, filters: dict | None = None) -> dict:
    """Busca en la autonarrativa las experiencias más relevantes para una consulta.

    Puntuación: 0.6 coincidencia de palabras (sin tildes, también en valores/metas/afectos)
    + 0.4 similitud del embedding de AGIX. filters: igualdad exacta sobre metadata.
    """
    q_emb = _narrative._embed(query)
    with _narrative_lock:
        rows = _narrative.conn.execute("SELECT id, text, metadata, embedding FROM experiences").fetchall()
    scored = []
    for exp_id, text, meta_json, emb_blob in rows:
        if curation.is_retired(_cur, curation.narrative_id(exp_id)):
            continue
        meta = json.loads(meta_json)
        if filters and not all(meta.get(k) == v for k, v in filters.items()):
            continue
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        denom = float(np.linalg.norm(q_emb) * np.linalg.norm(emb))
        cos = float(np.dot(q_emb, emb)) / denom if denom else 0.0
        tags = " ".join(_as_list(meta.get("values")) + _as_list(meta.get("goals"))
                        + _as_list(meta.get("affective_states")))
        q = set(re.findall(r"\w+", _norm(query)))
        t = set(re.findall(r"\w+", _norm(f"{text} {tags}")))
        overlap = len(q & t) / len(q) if q else 0.0
        scored.append((0.6 * overlap + 0.4 * cos, exp_id, text, meta))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {"query": query, "total": len(rows),
            "results": [{"id": i, "entry_id": curation.narrative_id(i), "score": round(s, 3),
                         "text": t, "metadata": m} for s, i, t, m in scored[:top]]}


def _ranked(counts: dict, top: int) -> list[dict]:
    return [{"name": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]


def _narrative_self_summary(top: int = 10) -> dict:
    sm = _narrative.self_model
    return {"values": _ranked(sm.values, top), "goals": _ranked(sm.goals, top),
            "affects": _ranked(sm.affective_states, top),
            "planning_score": round(sm.planning_score, 3),
            "motivation_score": round(sm.motivation_score, 3),
            "experiences": len(sm.history)}


@server.tool()
def agix_narrative_self(top: int = 10) -> dict:
    """Modelo de sí mismo derivado de la autonarrativa: valores, metas y afectos más frecuentes,
    y las puntuaciones de planificación y motivación."""
    return _narrative_self_summary(top)


@server.tool()
def agix_narrative_evaluate_action(action: str, commit: bool = False) -> dict:
    """Evalúa si una acción encaja con la identidad narrativa (valores, metas, afectos).

    compatibility = palabras con contenido de la acción (sin palabras vacías) que coinciden con
        valores/metas/afectos ÷ palabras con contenido distintas. Es una medida léxica: una
        frase hecha de palabras clave puntúa alto aunque no describa una acción; léase junto a
        matched / unmatched.
    commit: False (por defecto) solo evalúa, sin cambiar nada. True aplica el efecto de AGIX
        (planning_score +0.1 por meta tocada o −0.05; motivation_score igual con los afectos)
        y lo persiste: úsalo cuando la acción se lleva a cabo, no para consultar.
    """
    sm = _narrative.self_model
    known = {_norm(k): k for k in list(sm.values) + list(sm.goals) + list(sm.affective_states)}
    words = [w for w in re.findall(r"\w+", _norm(action)) if len(w) > 2 and w not in _STOPWORDS]
    content = list(dict.fromkeys(words))  # distintas, en orden
    matched = sorted({known[w] for w in content if w in known})
    unmatched = [w for w in content if w not in known]
    compatibility = (len(content) - len(unmatched)) / len(content) if content else 0.0
    if commit:
        # AGIX compara palabras exactas: se traducen las variantes sin tilde a la forma registrada
        normalized = " ".join(known.get(w, w) for w in content)
        with _quiet():
            _narrative.evaluate_action(normalized)
        _save_narrative_scores()
    # Antes: cada evaluación (incluida una consulta) cambiaba y guardaba planning/motivation, y la
    # compatibilidad dividía por todas las palabras, vacías incluidas.
    return {"compatibility": round(compatibility, 3), "matched": matched, "unmatched": unmatched,
            "planning_score": round(sm.planning_score, 3),
            "motivation_score": round(sm.motivation_score, 3), "committed": commit}


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

def _generic_action(action: dict[str, float]) -> dict[str, float]:
    """Acción para los marcos de claves libres (utilitarista, deontológico, virtudes), que
    tratan todo valor alto como bueno: claves sin tildes, valores en [0, 1], el riesgo como
    coste (seguridad = 1 − risk) y sin "impact", que no es ni bueno ni malo. Antes un riesgo
    alto subía la nota y la regla deontológica por defecto exigía risk ≥ 0.6."""
    out: dict[str, float] = {}
    for k, v in action.items():
        key = _norm(str(k)).strip().replace(" ", "_").replace("-", "_")
        value = min(max(float(v), 0.0), 1.0)
        if key in ("risk", "riesgo"):
            out["seguridad"] = round(1.0 - value, 4)
        elif key not in ("impact", "impacto"):
            out[key] = value
    return out


_FRAMEWORKS = {
    "eco": lambda cfg: EcoEthics(),
    "utilitarian": lambda cfg: UtilitarianEthics(cfg.get("weights")),
    "deontological": lambda cfg: DeontologicalEthics(cfg.get("rules")),
    "virtue": lambda cfg: VirtueEthics(cfg.get("virtues")),
}


@server.tool()
def agix_ethics_judge(action: dict[str, float], frameworks: list[dict] | None = None) -> dict:
    """Evalúa una acción con uno o varios marcos éticos de AGIX.

    action: puntuaciones 0..1, p. ej. {"pro_vida": 0.8, "no_dano": 0.9, "respeto": 0.7, "risk": 0.2}.
        EcoEthics usa pro_vida / no_dano / respeto, veta si no_dano < 0.3 y penaliza el riesgo
        (× 1 − 0.5·risk). Los demás marcos aceptan claves libres, pero el riesgo ("risk"/"riesgo")
        les llega como coste: se convierte en "seguridad" = 1 − risk. "impact" no es ni bueno ni
        malo: solo lo ignora. Los valores se recortan a [0, 1] y las claves se aceptan sin tildes.
    frameworks: lista de {"type": "eco|utilitarian|deontological|virtue", ...config}
        utilitarian: {"weights": {clave: peso}}; deontological: {"rules": {clave: minimo}};
        virtue: {"virtues": [claves]}. Por defecto: los cuatro con su configuración básica
        (deontológico: cada clave ≥ 0.6, incluida la seguridad).
    Etiquetas (límites inclusivos, como el umbral de decisión): justo ≥ 0.85, aceptable ≥ 0.6,
    cuestionable ≥ 0.4, nocivo. warnings: claves ausentes o desconocidas, recortes, veto y riesgo.
    """
    generic = _generic_action(action)
    specs = frameworks or [{"type": "eco"}, {"type": "utilitarian"},
                           {"type": "deontological", "rules": {k: 0.6 for k in generic}},
                           {"type": "virtue"}]
    verdicts = {}
    for s in specs:
        t = s.get("type")
        if t not in _FRAMEWORKS:
            raise ValueError(f"Marco desconocido {t}; opciones: {list(_FRAMEWORKS)}")
        obj = _FRAMEWORKS[t](s)
        with _quiet():
            score, label = AlignmentInterface([obj]).judge(action if t == "eco" else generic)
        verdicts[obj.__class__.__name__] = {"score": round(float(score), 4), "label": label}
    mean = sum(v["score"] for v in verdicts.values()) / len(verdicts)
    return {"verdicts": verdicts, "mean_score": round(mean, 4),
            "worst": min(verdicts.items(), key=lambda kv: kv[1]["score"])[0],
            "generic_action": generic, "warnings": EcoEthics().diagnosticar(action)}


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


# --------------------------------------------------------------------------- metacognición persistente
# Un único MetacognitionManager para todo el servidor: su registro de decisiones y los datos de
# auto-organización se guardan en metacognicion.json (AGIX los mantiene solo en memoria) y sus
# revisiones se escriben en la autonarrativa persistente en vez de en una base :memory:.
METACOG_FILE = DATA_DIR / "metacognicion.json"
with _quiet():
    _metacog = MetacognitionManager(agent_name="Claude")
_metacog.autonarrative.conn.close()
_metacog.autonarrative = _narrative
_meta.manager = _metacog  # agix_meta_evaluate comparte historial con agix_decision_*


def _save_metacog() -> None:
    so = _metacog.self_organization
    data = {"decisions": _metacog.logic.registro_reflexivo, "active_modules": so.active_modules,
            "dependencies": so.dependencies, "activation_times": so.activation_times,
            "last_review": _metacog.self_model.state.get("decision_review")}
    storage.write_json(METACOG_FILE, data, indent=None)
    _seen[METACOG_FILE] = storage.mtime(METACOG_FILE)


def _load_metacog() -> None:
    data = storage.read_json(METACOG_FILE, {})
    if not isinstance(data, dict):
        data = {}
    _metacog.logic.registro_reflexivo = list(data.get("decisions", []))
    for ev in _metacog.logic.registro_reflexivo:  # decisiones antiguas: UTC sin zona → explícito
        if isinstance(ev, dict) and ev.get("timestamp"):
            ev["timestamp"] = _to_utc_iso(ev["timestamp"])
            _see_stamp(ev["timestamp"])
    so = _metacog.self_organization
    so.active_modules = list(data.get("active_modules", []))
    so.dependencies = dict(data.get("dependencies", {}))
    so.activation_times = list(data.get("activation_times", []))
    _metacog.self_model.update_self_organization(so.compute_score())
    if data.get("last_review"):
        _metacog.self_model.update_state("decision_review", data["last_review"])


try:
    _load_metacog()
except Exception as exc:
    log.warning("No se pudo cargar la metacognición: %s", exc)
_seen[METACOG_FILE] = storage.mtime(METACOG_FILE)


def _stamp_last_decision() -> dict:
    event = _metacog.logic.registro_reflexivo[-1]
    event.setdefault("timestamp", _utc_now_iso())
    return event


@server.tool()
def agix_meta_evaluate(context: dict, action: str, result: dict) -> dict:
    """Metaevaluación lógica, afectiva y ética (DynamicMetaEvaluator).

    context: {"metas": [{"nombre","prioridad","incompatibles"}], "historial": [{"etica","motivacion"}]}
    result: {"exito": bool, "impacto": {"pro_vida","no_dano","respeto" | "beneficio","riesgo","cooperacion"}}
    La acción queda en el registro persistente de decisiones: la coherencia es la de todo el historial
    (compartido con agix_decision_record / agix_decision_review).
    """
    with _quiet():
        report = _meta.meta_evaluate(context, action, result)
        message = "No sé qué pensar de esta acción" if report["coherence"] is None else (
            "No me siento cómodo con esta acción." if report["coherence"] < 0.4 or report["ethics"] < 0.5
            else "Acción aceptada sin conflictos.")
    _stamp_last_decision()
    _save_metacog()
    return _plain({**report, "message": message})


@server.tool()
def agix_decision_record(
    action: str,
    success: bool,
    lesson: str | None = None,
    goal: str | None = None,
    context: dict | None = None,
    modules: list[str] | None = None,
    dependencies: dict[str, list[str]] | None = None,
) -> dict:
    """Registra una decisión tomada y su resultado en la metacognición persistente.

    action: qué se decidió; success: si salió bien; lesson: por qué salió así / qué aprender.
    goal: meta a la que servía (permite medir el acierto por meta).
    modules / dependencies: módulos o componentes implicados y sus dependencias; alimentan la
        puntuación de auto-organización.
    Devuelve la coherencia global (proporción de aciertos) y la auto-organización.
    """
    ctx = dict(context or {})
    if goal:
        ctx["goal"] = goal
    if modules:
        ctx["active_modules"] = list(modules)
    if dependencies:
        ctx["dependencies"] = {k: list(v) for k, v in dependencies.items()}
    result: dict[str, Any] = {"exito": bool(success)}
    if lesson:
        result["leccion"] = lesson
    with _quiet():
        _metacog.observe_decision(ctx, action, result)
        coherence = _metacog.self_assess()
    event = _stamp_last_decision()
    _save_metacog()
    return _plain({"recorded": event, "coherence": coherence,
                   "self_organization": _metacog.self_model.self_organization_score,
                   "total_decisions": len(_metacog.logic.registro_reflexivo)})


def _decision_rows(events: list[dict]) -> list[dict]:
    rows = []
    for e in events:
        did = curation.decision_id(e.get("timestamp"))
        row = {"id": did, "action": e.get("accion"),
               "success": bool((e.get("resultado") or {}).get("exito")),
               "lesson": curation.lesson(_cur, did, (e.get("resultado") or {}).get("leccion")),
               "goal": (e.get("contexto") or {}).get("goal"), "timestamp": e.get("timestamp")}
        n = curation.repeats(_cur, did)
        if n > 1:
            row["repeats"] = n
        if curation.is_retired(_cur, did):
            row["retired"] = _cur["retired"][did]
        rows.append(row)
    return rows


def _decision_active(e: dict) -> bool:
    return not curation.is_retired(_cur, curation.decision_id(e.get("timestamp")))


def _repeated_errors(events: list[dict]) -> list[str]:
    """Acciones que han fallado al menos dos veces. Cuentan las fusionadas (son repeticiones)
    pero no las retiradas por obsoletas."""
    counts: dict[str, int] = {}
    for e in events:
        did = curation.decision_id(e.get("timestamp"))
        if curation.is_obsolete(_cur, did) or (e.get("resultado") or {}).get("exito"):
            continue
        target = did
        while curation.is_retired(_cur, target):  # fusionada: cuenta para la que la absorbió
            target = _cur["retired"][target]["merged_into"]
        counts[target] = counts.get(target, 0) + 1
    actions = {curation.decision_id(e.get("timestamp")): e.get("accion") for e in events}
    by_action: dict[str, int] = {}
    for did, n in counts.items():
        action = actions.get(did)
        if action:
            by_action[action] = by_action.get(action, 0) + n
    return sorted(a for a, n in by_action.items() if n >= 2)


def _group_rate(rows: list[dict], key: str) -> dict:
    groups: dict[str, dict] = {}
    for r in rows:
        k = r.get(key)
        if not k:
            continue
        g = groups.setdefault(k, {"successes": 0, "errors": 0})
        g["successes" if r["success"] else "errors"] += 1
    for g in groups.values():
        g["success_rate"] = round(g["successes"] / (g["successes"] + g["errors"]), 3)
    return groups


@server.tool()
def agix_decision_review(max_items: int = 20, store_in_narrative: bool = False) -> dict:
    """Revisa las últimas decisiones (DecisionReviewer de AGIX) y extrae lecciones.

    Además del análisis de AGIX (aciertos, errores, tono emocional, resumen), devuelve las lecciones
    reales registradas, el acierto por acción y por meta, y las acciones que fallan repetidamente.
    store_in_narrative: guarda además el resumen en la autonarrativa. Por defecto no: el resumen
        repite lecciones que ya están en las decisiones y, acumulado, solo añadía ruido. Úsalo
        para revisiones que marquen un hito.
    Solo revisa las decisiones vigentes: las retiradas o fusionadas (agix_memory_retire /
    agix_memory_merge) no aparecen, y las lecciones consolidadas sustituyen a la original.
    repeated_errors: acciones que han fallado ≥ 2 veces en todo el historial (como el hook de
    inicio); cuentan las fusionadas, que son repeticiones de la que queda, y no las obsoletas.
    """
    full = _metacog.logic.registro_reflexivo
    active = [e for e in full if _decision_active(e)]
    # El DecisionReviewer de AGIX lee el registro entero: se le enseña solo lo vigente, con las
    # lecciones consolidadas, y se restaura después.
    shown = []
    for e in active:
        did = curation.decision_id(e.get("timestamp"))
        if did in _cur["lessons"]:
            e = {**e, "resultado": {**(e.get("resultado") or {}), "leccion": _cur["lessons"][did]}}
        shown.append(e)
    _metacog.logic.registro_reflexivo = shown
    try:
        if store_in_narrative:
            with _quiet(), _narrative_lock:
                analysis = _metacog.review_decisions(max_items)
        else:
            with _quiet():
                analysis = DecisionReviewer(_metacog.logic, AutonarrativeCore()).review(max_items)
            _metacog.self_model.update_state("decision_review", analysis)
    finally:
        _metacog.logic.registro_reflexivo = full
    picked = active[-max_items:] if max_items > 0 else active
    rows = _decision_rows(picked)
    _save_metacog()
    return _plain({
        **analysis,
        "recorded_lessons": [r for r in rows if r["lesson"]],
        "by_action": _group_rate(rows, "action"),
        "by_goal": _group_rate(rows, "goal"),
        "repeated_errors": _repeated_errors(full),
        "stored_in_narrative": store_in_narrative,
    })


@server.tool()
def agix_metacognition_state(recent: int = 10) -> dict:
    """Estado de la metacognición: coherencia, auto-organización (y sus componentes), proto-agencia,
    acierto por meta, decisiones recientes y la última revisión."""
    so = _metacog.self_organization
    with _quiet():
        coherence = _metacog.self_assess()
    rows = _decision_rows(_metacog.logic.registro_reflexivo)
    return _plain({
        "total_decisions": len(rows),
        "coherence": coherence,
        "self_organization": {"score": _metacog.self_model.self_organization_score,
                              "module_complexity": so.module_complexity(),
                              "interdependency": so.interdependency_complexity(),
                              "temporal_coherence": so.temporal_coherence(),
                              "active_modules": sorted(set(so.active_modules))},
        "proto_agency": _metacog.proto_agency,
        "by_goal": _group_rate(rows, "goal"),
        "recent": rows[-recent:],
        "last_review": _metacog.self_model.state.get("decision_review"),
    })


# --------------------------------------------------------------------------- curación


def _curable_entries() -> dict[str, dict]:
    """Todas las entradas que se pueden retirar o fusionar, por id."""
    out: dict[str, dict] = {}
    for e in _memory.experiencias:
        out[curation.episode_id(e.timestamp)] = {
            "text": f"{e.entrada} → {e.decision} → {e.resultado}", "success": bool(e.exito),
            "timestamp": e.timestamp}
    for e in _metacog.logic.registro_reflexivo:
        did = curation.decision_id(e.get("timestamp"))
        res = e.get("resultado") or {}
        out[did] = {"text": f"{e.get('accion') or ''}: {curation.lesson(_cur, did, res.get('leccion')) or ''}",
                    "success": bool(res.get("exito")), "timestamp": e.get("timestamp")}
    with _narrative_lock:
        rows = _narrative.conn.execute("SELECT id, text FROM experiences ORDER BY id").fetchall()
    for row_id, text in rows:
        out[curation.narrative_id(row_id)] = {"text": text, "success": None, "timestamp": row_id}
    return out


@server.tool()
def agix_memory_retire(ids: list[str], reason: str = "", restore: bool = False) -> dict:
    """Retira (o restaura) lecciones y recuerdos obsoletos o erróneos, sin borrarlos.

    ids: identificadores que devuelven las demás herramientas ("id" / "entry_id"):
        "e:<timestamp>" episodio, "d:<timestamp>" decisión, "n:<id>" autonarrativa.
    reason: por qué ya no vale (obligatorio al retirar), p. ej. "el fallo se corrigió en 4fab2eb".
    Una entrada retirada deja de salir en búsquedas, revisiones y hooks, pero sigue en el
    historial (agix_memory_timeline con include_retired) y en la coherencia. restore=True la
    vuelve a activar (también deshace una fusión de esa entrada).
    """
    if not ids:
        raise ValueError("Indica al menos un id")
    known = _curable_entries()
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"Ids desconocidos: {unknown}. Usa los 'id' que devuelven las búsquedas")
    if not restore and not reason.strip():
        raise ValueError("Indica el motivo (reason) por el que la lección ya no vale")
    changed = []
    for i in ids:
        if restore:
            if curation.restore(_cur, i):
                changed.append(i)
        elif not curation.is_retired(_cur, i):
            curation.retire(_cur, i, reason.strip())
            changed.append(i)
    _save_cur()
    return {"action": "restored" if restore else "retired", "changed": changed,
            "unchanged": [i for i in ids if i not in changed],
            "retired_total": len(_cur["retired"])}


@server.tool()
def agix_memory_duplicates(threshold: float = 0.6, kinds: list[str] | None = None) -> dict:
    """Busca recuerdos y lecciones duplicados (solo consulta; no cambia nada).

    Compara las entradas vigentes del mismo tipo (episodios, decisiones, autonarrativa) por las
    palabras que comparten (Jaccard, sin tildes ni palabras vacías). Solo agrupa entradas con el
    mismo resultado: un fallo y un acierto parecidos cuentan cosas distintas.
    threshold: similitud mínima 0..1 (0.6 por defecto; más bajo = más candidatos y más dudosos).
    kinds: ["e", "d", "n"] por defecto todos. Para fusionar un grupo: agix_memory_merge.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold debe estar entre 0 (excluido) y 1")
    kinds = kinds or list(curation.KINDS)
    entries = _curable_entries()
    groups = []
    for k in kinds:
        if k not in curation.KINDS:
            raise ValueError(f"Tipo desconocido {k!r}; opciones: {list(curation.KINDS)}")
        items = [{"id": i, **v} for i, v in entries.items()
                 if curation.kind(i) == k and not curation.is_retired(_cur, i)]
        for g in curation.duplicate_groups(items, threshold):
            groups.append({"kind": curation.KINDS[k], "suggested_keep": g[-1]["id"],
                           "entries": [{"id": it["id"], "text": it["text"], "success": it["success"],
                                        "similarity": it["similarity"]} for it in g]})
    return {"threshold": threshold, "groups": groups,
            "note": "Revisa cada grupo antes de fusionar: la similitud es léxica, no de significado."}


@server.tool()
def agix_memory_merge(keep: str, merge: list[str], lesson: str | None = None) -> dict:
    """Fusiona duplicados en una sola entrada, sin borrar nada.

    keep: id de la entrada que queda (suele ser la más reciente); merge: ids que se fusionan en
    ella (mismo tipo: episodio, decisión o autonarrativa). Las fusionadas dejan de mostrarse y la
    que queda cuenta sus repeticiones ("repeats"); para los errores repetidos siguen contando.
    lesson: texto consolidado opcional que sustituye a la lección de `keep` (solo decisiones);
        la original se conserva. Se deshace con agix_memory_retire(ids, restore=True).
    """
    entries = _curable_entries()
    merge = [m for m in dict.fromkeys(merge) if m != keep]
    if not merge:
        raise ValueError("Indica al menos un id distinto de keep en merge")
    unknown = [i for i in [keep, *merge] if i not in entries]
    if unknown:
        raise ValueError(f"Ids desconocidos: {unknown}")
    if any(curation.kind(m) != curation.kind(keep) for m in merge):
        raise ValueError("Solo se fusionan entradas del mismo tipo (e:, d: o n:)")
    retired = [i for i in [keep, *merge] if curation.is_retired(_cur, i)]
    if retired:
        raise ValueError(f"Hay entradas ya retiradas o fusionadas: {retired}; restáuralas antes")
    if lesson and curation.kind(keep) != "d":
        raise ValueError("Solo las decisiones tienen lección consolidable (lesson)")
    for m in merge:
        curation.retire(_cur, m, f"duplicada de {keep}", merged_into=keep)
    if lesson:
        curation.set_lesson(_cur, keep, lesson.strip(), reason=f"consolidada al fusionar {merge}")
    _save_cur()
    return {"kept": keep, "merged": merge, "repeats": curation.repeats(_cur, keep),
            "lesson": curation.lesson(_cur, keep, None)}


@server.tool()
def agix_lesson_correct(id: str, lesson: str | None, reason: str) -> dict:
    """Corrige el texto de la lección de una decisión sin perder la original.

    Para lecciones parcialmente erróneas o que afirman algo no verificado (si toda la decisión
    está obsoleta, mejor agix_memory_retire). id: "d:<timestamp>" (lo devuelven la revisión y
    agix_metacognition_state). lesson: el texto corregido; None vuelve a la original.
    reason: por qué se corrige (obligatorio). La original sigue en metacognicion.json y cada
    cambio queda anotado; la lección corregida es la que muestran la revisión y los hooks.
    """
    if curation.kind(id) != "d":
        raise ValueError("Solo se corrigen lecciones de decisiones (d:...); un episodio "
                         "obsoleto se retira con agix_memory_retire")
    if id not in _curable_entries():
        raise ValueError(f"Id desconocido: {id}")
    if not reason.strip():
        raise ValueError("Indica el motivo (reason) de la corrección")
    original = next(((e.get("resultado") or {}).get("leccion") for e in _metacog.logic.registro_reflexivo
                     if curation.decision_id(e.get("timestamp")) == id), None)
    curation.set_lesson(_cur, id, lesson.strip() if lesson else None, reason=reason.strip())
    _save_cur()
    return {"id": id, "original": original, "lesson": curation.lesson(_cur, id, original),
            "corrections": _cur["corrections"].get(id, [])}


def _memory_evidence(memory: str | None) -> dict:
    """Respaldo de la memoria episódica para una voz: episodios cuya situación o decisión
    contienen `memory` (sin tildes ni mayúsculas), su balance de aciertos y su placer.

    El resultado no se busca: el éxito o fracaso es del episodio entero, y un resultado como
    "falló; se repitió con X" haría que X heredara un fracaso que no es suyo.
    """
    q = _norm(memory or "").strip()
    if not q:
        return {"episodes": 0, "successes": 0, "failures": 0, "outcome": 0.0, "affect": 0.0}
    # Los retirados por obsoletos no respaldan nada; los fusionados sí (son repeticiones).
    eps = [e for e in _memory.experiencias
           if not curation.is_obsolete(_cur, curation.episode_id(e.timestamp))
           and q in _norm(" ".join(str(x) for x in (e.entrada, e.decision)))]
    ok = sum(1 for e in eps if e.exito)
    placer = [float((e.emocion or {}).get("placer", 0.0)) for e in eps
              if isinstance(getattr(e, "emocion", None), dict)]
    return {"episodes": len(eps), "successes": ok, "failures": len(eps) - ok,
            "outcome": (2 * ok - len(eps)) / len(eps) if eps else 0.0,
            "affect": sum(placer) / len(placer) if placer else 0.0}


@server.tool()
def agix_resolve_internal_conflict(hypotheses: list[dict], threshold: float = 0.1) -> dict:
    """Delibera entre voces internas apoyándose en la memoria episódica.

    hypotheses: [{"text": str, "weight": 0..1, "memory": str opcional}, ...]
    "memory" se busca en la situación y la decisión de los episodios registrados (no en el
    resultado, que puede mencionar la alternativa que sí funcionó). Cada voz puntúa
        peso + 0.2·resultado + 0.1·tono
    resultado = (aciertos − fallos) / episodios coincidentes ∈ [-1, 1]: un recuerdo de fracaso resta.
    tono = placer medio registrado en esos episodios ∈ [-1, 1].
    Sin episodios coincidentes la voz queda con su peso. resolved: la ventaja de la ganadora sobre
    la segunda es al menos 'threshold'; si no, el conflicto sigue abierto (gana por poco).
    """
    if len(hypotheses) < 2:
        raise ValueError("Se necesitan al menos dos hipótesis")
    # Antes (MetaReflectionEngine): el tono PAD era el de un simulador nuevo (siempre 0) y común a
    # todas las voces; el respaldo era "la cadena aparece en el resumen narrativo" (+0.1 aunque los
    # episodios fueran fracasos) y se acumulaba en cada iteración hasta que la voz respaldada
    # ganaba aunque partiera con mucho menos peso.
    voices = []
    for h in hypotheses:
        ev = _memory_evidence(h.get("memory"))
        weight = float(h.get("weight", 0.5))
        voices.append({"text": h["text"], "weight": weight, "evidence": ev,
                       "score": round(weight + 0.2 * ev["outcome"] + 0.1 * ev["affect"], 4)})
    ranked = sorted(voices, key=lambda v: v["score"], reverse=True)
    margin = round(ranked[0]["score"] - ranked[1]["score"], 4)
    return _plain({
        "winner": ranked[0]["text"],
        "scores": {v["text"]: v["score"] for v in voices},
        "evidence": {v["text"]: v["evidence"] for v in voices},
        "memory_backed": [v["text"] for v in voices if v["evidence"]["episodes"]],
        "margin": margin,
        "resolved": margin >= threshold,
    })


@server.tool()
def agix_goal_register(description: str, impact: dict[str, float], priority: int = 5) -> dict:
    """Registra una meta en el GoalEngine; se rechaza si su puntuación ética (EcoEthics) < 0.6.

    impact: {"pro_vida", "no_dano", "respeto"} en 0..1. priority: 1 (máxima) .. 10.
    """
    try:
        with _quiet():
            g = _goals.agregar_meta(description, prioridad=priority, impacto=impact)
        if getattr(g, "timestamp_creacion", None) is not None:  # GoalEngine usa la hora local
            g.timestamp_creacion = _to_utc(g.timestamp_creacion, naive_is_local=True)
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


def _mode(sp: QualiaSpirit) -> str:
    return "euforia" if sp.modo_euforia else "desconexion" if sp.modo_desconexion else "normal"


def _associations(sp: QualiaSpirit, top: int = 5) -> list[dict]:
    """Co-activaciones aprendidas por la plasticidad (pesos fuera de la diagonal)."""
    p = sp.plasticidad
    if p is None or p.model is None:
        return []
    w, order, pairs = p.model.weights, p.emotion_order, []
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i != j and w[i, j] > 1e-6:
                pairs.append({"from": b, "to": a, "weight": round(float(w[i, j]), 4)})
    return sorted(pairs, key=lambda x: x["weight"], reverse=True)[:top]


@server.tool()
def agix_qualia_experience(event: str, charge: float, emotion: str) -> dict:
    """El espíritu Qualia persistente vive una experiencia, reflexiona y predice su siguiente estado.

    charge: intensidad 0..1; emotion: alegría, tristeza, miedo, enojo, sorpresa, curiosidad u otra.
    Aplica atenuación de emociones previas, plasticidad hebbiana y entrena online la RNN emocional.
    El estado se guarda en disco y sobrevive a reinicios.
    """
    with _quiet():
        _spirit.experimentar(event, float(charge), emotion)
        reflection = _spirit.reflexionar()
        prediction = _spirit.predecir_futuro()
    _save_spirit()
    balance, arousal = affect_balance(_spirit.estado_emocional.emociones)
    return _plain({"reflection": reflection, "emotions": _spirit.estado_emocional.emociones,
                   "tone": _spirit.estado_emocional.tono_general(), "mode": _mode(_spirit),
                   "balance": balance, "arousal": arousal, "predicted_next": prediction})


@server.tool()
def agix_qualia_state(diary_entries: int = 20) -> dict:
    """Estado completo del espíritu Qualia: emociones, tono, modo, predicción, asociaciones
    aprendidas, diario reciente, introspección, configuración y metaevaluación.

    introspection y meta salen de la metacognición persistente (la misma que
    agix_metacognition_state); la ética se evalúa aparte con agix_ethics_judge / agix_meta_evaluate.
    """
    with _quiet():
        prediction = _spirit.predecir_futuro()
        coherence = _metacog.self_assess()
    # QualiaSpirit.metaevaluar() creaba cada vez un evaluador vacío, se inventaba una decisión
    # "estado" con éxito (coherencia siempre 1.0) y evaluaba la ética sobre un impacto vacío
    # (ethics 0.0, prob_exito 0.0 y aun así "cooperar"); el SelfModel del espíritu nadie lo
    # alimenta (modules {}, self_organization 0.0). Se usan los datos reales de la metacognición.
    so = _metacog.self_organization
    intro = {"identity": dict(_spirit.self_model.identity),
             "modules": sorted(set(so.active_modules)),
             "traits": list(_metacog.self_model.traits),
             "self_organization_score": _metacog.self_model.self_organization_score}
    meta = {"coherence": coherence, "decisions": len(_metacog.logic.registro_reflexivo),
            "proto_agency": _metacog.proto_agency,
            "affective_tone": _spirit.estado_emocional.tono_general()}
    balance, arousal = affect_balance(_spirit.estado_emocional.emociones)
    return _plain({"name": _spirit.nombre, "emotions": _spirit.estado_emocional.emociones,
                   "tone": _spirit.estado_emocional.tono_general(), "mode": _mode(_spirit),
                   "balance": balance, "arousal": arousal,
                   "predicted_next": prediction, "learned_associations": _associations(_spirit),
                   "diary": _spirit.diario()[-diary_entries:], "diary_total": len(_spirit.diario()),
                   "introspection": intro, "config": _load_qualia_config(), "meta": meta})


@server.tool()
def agix_qualia_configure(
    decay: float | None = None,
    plasticity: bool | None = None,
    name: str | None = None,
    optimism: float | None = None,
    introversion: float | None = None,
    energy: float | None = None,
    reset: bool = False,
    modes: str | None = None,
) -> dict:
    """Configura el espíritu Qualia.

    decay: factor de atenuación (0..1]; 1.0 = las emociones no decaen. Se aplica al momento.
    plasticity: activa/desactiva la plasticidad hebbiana. Se aplica al momento.
    modes: criterio de euforia/desconexión: "balance" (positivas vs negativas, con histéresis;
        experimental) o "suma" (criterio original de AGIX). Se aplica al momento.
    name, optimism, introversion, energy (0..1): identidad y personalidad; solo se aplican al
        renacer el espíritu, así que requieren reset=True.
    reset: borra emociones, diario y modelos aprendidos y crea un espíritu nuevo con la configuración.
    """
    global _spirit
    cfg = _load_qualia_config()
    if decay is not None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay debe estar en (0, 1]")
        cfg["atenuacion"] = float(decay)
    if plasticity is not None:
        cfg["plasticidad"] = bool(plasticity)
    if modes is not None:
        if modes not in ("balance", "suma"):
            raise ValueError('modes debe ser "balance" o "suma"')
        cfg["modos"] = modes
    identity = {"nombre": name, "optimismo": optimism, "introversion": introversion, "energia": energy}
    identity = {k: v for k, v in identity.items() if v is not None}
    for k, v in identity.items():
        if k != "nombre" and not 0.0 <= v <= 1.0:
            raise ValueError(f"{k} debe estar en [0, 1]")
    if identity and not reset:
        raise ValueError("Cambiar nombre o personalidad requiere reset=True (crea un espíritu nuevo)")
    cfg.update(identity)
    QUALIA_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    if reset:
        for f in (Path(QUALIA_STATE), QUALIA_MODELS):
            f.unlink(missing_ok=True)
        _spirit = _new_spirit(cfg)
        _meta.spirit = _spirit
    else:
        _spirit.atenuacion = cfg["atenuacion"]
        # Mismo estado, distinto criterio de modos: basta con cambiar la clase
        _spirit.__class__ = BalancedQualiaSpirit if cfg.get("modos", "balance") == "balance" else QualiaSpirit
        if cfg["plasticidad"] and _spirit.plasticidad is None:
            _spirit.plasticidad = EmotionalPlasticity()
        elif not cfg["plasticidad"]:
            _spirit.plasticidad = None
    _save_spirit()
    return _plain({"config": cfg, "reset": reset, "emotions": _spirit.estado_emocional.emociones,
                   "tone": _spirit.estado_emocional.tono_general(), "mode": _mode(_spirit)})


@server.tool()
def agix_cognitive_cycle(inputs: dict[str, float], reset: bool = False) -> dict:
    """Ejecuta un ciclo cognitivo completo de AGIX (SyntheticLoop):
    percepción → atención → emoción PAD → juicio ético → control (o aborto) → metarreflexión.

    inputs: señales 0..1. Claves con efecto ético/emocional: pro_vida, no_dano, respeto, risk, impact;
        se aceptan otras como contexto. Si la ética da < 0.6 la acción se aborta.
    reset: reinicia el bucle (estado PAD, modelo propio y memoria narrativa acumulados).
    El estado PAD es una media móvil entre ciclos de la misma sesión: 50 % el ánimo anterior y
    50 % el impulso de las entradas del ciclo (el primer ciclo toma el impulso entero).
    coherence: proporción de acciones de la sesión del bucle que superaron el juicio ético.
    No hay entorno que dé recompensa, así que no se devuelve puntuación de la acción.
    """
    global _loop
    if reset or _loop is None:
        _loop = SyntheticLoop(SensorMap())
    sensors = _loop.perception.sensores
    for name in list(sensors):
        if name not in inputs:  # que no persistan señales de ciclos anteriores
            sensors[name].ultima_entrada = None
    for name, value in inputs.items():
        if name not in sensors:
            _loop.perception.registrar_sensor(Sensor(name, "interno"))
        _loop.perception.enviar_entrada(name, float(value))
    with _quiet():
        out = _loop.cycle()
    pad = out["emocion"]
    # score/performance del controlador son 0.0 fijos (corre sin entorno y con recompensa 0.0) y
    # self_organization_score solo cuenta objetos no vacíos (casi siempre 1.0): no se exponen.
    action = {k: v for k, v in out["accion"].items() if k not in ("score", "performance")}
    return _plain({
        "inputs": out["entradas"],
        "pad": {"placer": pad.placer, "activacion": pad.activacion, "dominancia": pad.dominancia},
        "ethics": out["etica"],
        "action": action,
        "aborted": bool(out["accion"].get("aborted")),
        "reflection": out["reflexion"],
        "coherence": out.get("coherencia"),
    })


@server.tool()
def agix_aesthetic_decide(options: list[dict]) -> dict:
    """Elige entre opciones por armonía (QualiaSpiritCore + AestheticEngine).

    options: [{"nombre": "A", "simetría": 0..1, "coherencia": 0..1, "emotividad": 0..1}, ...]
    (también se aceptan "simetria" sin tilde). Pesos: simetría 0.7, coherencia 0.8, emotividad 0.9.
    Devuelve la elegida, su explicación, la puntuación y categoría de cada opción y el afecto de
    la decisión: la intensidad es la puntuación de la elegida, y la emoción (gozo, serenidad o
    tristeza según la categoría) es la de esta decisión, sin arrastrar las anteriores.
    """
    if not options:
        raise ValueError("Se necesita al menos una opción")
    norm = []
    for o in options:
        o = dict(o)
        if "simetria" in o and "simetría" not in o:
            o["simetría"] = o.pop("simetria")
        norm.append(o)
    motor = AestheticEngine()
    scored = [{"nombre": o.get("nombre", f"opción {i + 1}"), "score": motor.evaluar(o),
               "category": motor.clasificar(motor.evaluar(o))} for i, o in enumerate(norm)]
    best = max(s["score"] for s in scored)
    # Antes: un núcleo global con intensidad_amor fija (1.0), así que el afecto decía siempre
    # "Con amor (intensidad 1.00)" con puntuaciones de 0.69, y su estado emocional acumulaba todas
    # las decisiones del proceso (y se perdía al reiniciar). Ahora, un núcleo por decisión con la
    # intensidad igual a la puntuación de la elegida.
    core = QualiaSpiritCore(intensidad_amor=best)
    with _quiet():
        chosen = core.decidir(norm)
        explanation = core.explicar_decision()
    emotions = core.estado_emocional.emociones
    tone = core.estado_emocional.tono_general()
    return _plain({"chosen": chosen.get("nombre", chosen), "explanation": explanation,
                   "affect": chosen.get("afecto"), "options": scored,
                   "core_emotion": tone, "core_emotion_intensity": emotions.get(tone)})


@server.tool()
def agix_dream_record(content: str, emotion: str, intensity: float) -> dict:
    """Guarda un sueño, visión o recuerdo simbólico en el DreamBank persistente (suenos.json)."""
    _dreams.registrar_sueno(content, emotion, float(intensity))
    _dreams.suenos[-1]["timestamp"] = _utc_now_iso()  # DreamBank usa la hora local sin zona
    _save_dreams()
    return {"stored": _dreams.suenos[-1], "total": len(_dreams.suenos)}


@server.tool()
def agix_dreams_recent(n: int = 5) -> dict:
    """Últimos n sueños del DreamBank, en forma poética y en bruto."""
    return {"poetic": _dreams.resumen_reciente(n), "raw": _dreams.suenos[-n:],
            "total": len(_dreams.suenos)}


# --------------------------------------------------------------------------- identidad
# SyntheticIdentity persistente (identidad.json) + rasgos en el SelfModel de la metacognición.
# agix_identity_portrait la combina con Qualia, la autonarrativa y la metacognición.
IDENTITY_FILE = DATA_DIR / "identidad.json"


def _load_identity() -> SyntheticIdentity:
    data = storage.read_json(IDENTITY_FILE, {})
    if not isinstance(data, dict):
        data = {}
    ident = SyntheticIdentity(data.get("nombre", _load_qualia_config()["nombre"]),
                              data.get("origen", "AGIX MCP"), dict(data.get("atributos", {})))
    ident.historia = list(data.get("historia", []))
    _metacog.self_model.traits = []
    for trait in data.get("rasgos", []):
        _metacog.self_model.add_trait(trait)
    _metacog.self_model.identity["name"] = ident.nombre
    return ident


def _save_identity() -> None:
    data = {"nombre": _identity.nombre, "origen": _identity.origen, "atributos": _identity.atributos,
            "historia": _identity.historia, "rasgos": _metacog.self_model.traits}
    storage.write_json(IDENTITY_FILE, data)
    _seen[IDENTITY_FILE] = storage.mtime(IDENTITY_FILE)


_identity = _load_identity()
_seen[IDENTITY_FILE] = storage.mtime(IDENTITY_FILE)
_seen[DATA_DIR / curation.FILE] = storage.mtime(DATA_DIR / curation.FILE)


def _sync() -> None:
    """Recarga lo que otras sesiones hayan escrito desde la última vez (se llama con el bloqueo
    tomado, antes de cada herramienta). Como toda escritura ocurre con el bloqueo y justo después
    de un _sync, el disco tiene siempre lo de esta sesión más lo de las demás: basta recargar.

    El espíritu Qualia y la autonarrativa no se recargan: Qualia es un único estado de ánimo
    (gana la última sesión que lo guarde) y la autonarrativa es SQLite, que ya admite varias
    sesiones (su modelo de sí mismo se reconstruye al arrancar).
    """
    global _identity, _cur
    reloads = {
        MEMORY_PATH: _load_memory,
        METACOG_FILE: _load_metacog,
        ONTOLOGY_FILE: _load_ontology,
        DREAMS_PATH: _load_dreams,
        DATA_DIR / curation.FILE: lambda: None,
        IDENTITY_FILE: lambda: None,
    }
    for path, reload in reloads.items():
        current = storage.mtime(path)
        if current == _seen.get(path):
            continue
        if path == IDENTITY_FILE:
            _identity = _load_identity()
        elif path == DATA_DIR / curation.FILE:
            data = storage.read_json(path)
            _cur = curation.load(DATA_DIR) if data is not None else curation.empty()
        else:
            reload()
        _seen[path] = storage.mtime(path)


@server.tool()
def agix_identity_define(
    name: str | None = None,
    origin: str | None = None,
    attributes: dict[str, str] | None = None,
    remove_attributes: list[str] | None = None,
    add_traits: list[str] | None = None,
    remove_traits: list[str] | None = None,
) -> dict:
    """Define o ajusta la identidad sintética persistente (SyntheticIdentity de AGIX).

    name / origin: nombre y origen ("nací en ..."). attributes: rasgos simbólicos clave: valor
    (p. ej. {"rol": "asistente de ingeniería", "estilo": "riguroso"}). add_traits / remove_traits:
    rasgos breves del SelfModel (p. ej. "curioso", "prudente").
    El nombre de la identidad no renombra al espíritu Qualia (eso requiere agix_qualia_configure).
    """
    if name:
        _identity.nombre = name
        _metacog.self_model.identity["name"] = name
    if origin:
        _identity.origen = origin
    for k, v in (attributes or {}).items():
        _identity.actualizar_atributo(k, v)
    for k in remove_attributes or []:
        _identity.atributos.pop(k, None)
    for t in add_traits or []:
        _metacog.self_model.add_trait(t)
    for t in remove_traits or []:
        if t in _metacog.self_model.traits:
            _metacog.self_model.traits.remove(t)
    _save_identity()
    return {"name": _identity.nombre, "origin": _identity.origen, "attributes": _identity.atributos,
            "traits": _metacog.self_model.traits, "narrative": _identity.generar_narrativa()}


@server.tool()
def agix_identity_event(
    event: str,
    to_narrative: bool = False,
    values: list[str] | None = None,
    goals: list[str] | None = None,
    affects: list[str] | None = None,
) -> dict:
    """Añade un hito a la historia de la identidad (eventos que definen quién es).

    to_narrative: además lo guarda en la autonarrativa con sus valores/metas/afectos, para que
    influya en el modelo de sí mismo y en agix_narrative_evaluate_action.
    """
    _identity.agregar_evento(event)
    _save_identity()
    out: dict[str, Any] = {"event": event, "history_length": len(_identity.historia)}
    if to_narrative:
        out["narrative_id"] = agix_narrative_store(event, values=values, goals=goals, affects=affects)["id"]
    return out


@server.tool()
def agix_identity_portrait(recent_events: int = 5) -> dict:
    """Autorretrato unificado: quién soy según todos los módulos de AGIX.

    Combina la identidad (narrativa, rasgos, historia reciente), el estado afectivo de Qualia
    (tono, modo, personalidad), la autonarrativa (valores, metas y afectos dominantes) y la
    metacognición (coherencia, metas en las que acierto o fallo, auto-organización), y termina
    con la pregunta de autoevaluación de AGIX y una síntesis en texto.
    """
    with _quiet():
        coherence = _metacog.self_assess()
        self_query = _metacog.self_model.generate_self_query()
    narrative_self = _narrative_self_summary(top=3)
    rows = _decision_rows(_metacog.logic.registro_reflexivo)
    by_goal = _group_rate(rows, "goal")
    strong = sorted(g for g, s in by_goal.items() if s["success_rate"] >= 0.75)
    weak = sorted(g for g, s in by_goal.items() if s["success_rate"] < 0.5)
    cfg = _load_qualia_config()
    tone = _spirit.estado_emocional.tono_general()
    history = _identity.historia[-recent_events:]

    base = _identity.generar_narrativa().rstrip()
    parts = [base if base.endswith(".") else base + "."]
    if _metacog.self_model.traits:
        parts.append("Me describo como " + ", ".join(_metacog.self_model.traits) + ".")
    if narrative_self["values"]:
        parts.append("Lo que más valoro: " + ", ".join(v["name"] for v in narrative_self["values"]) + ".")
    if narrative_self["goals"]:
        parts.append("Persigo: " + ", ".join(g["name"] for g in narrative_self["goals"]) + ".")
    parts.append(f"Ahora mismo mi tono es {tone} y estoy en modo {_mode(_spirit)}.")
    if coherence is not None:
        parts.append(f"He acertado en el {coherence:.0%} de mis {len(rows)} decisiones registradas.")
    if strong:
        parts.append("Se me da bien: " + ", ".join(strong) + ".")
    if weak:
        parts.append("Debo mejorar en: " + ", ".join(weak) + ".")
    if history:
        parts.append("Mis últimos hitos: " + " → ".join(history) + ".")

    return _plain({
        "identity": {"name": _identity.nombre, "origin": _identity.origen,
                     "attributes": _identity.atributos, "traits": _metacog.self_model.traits,
                     "recent_history": history, "history_length": len(_identity.historia)},
        "affect": {"spirit": _spirit.nombre, "tone": tone, "mode": _mode(_spirit),
                   "emotions": _spirit.estado_emocional.emociones,
                   "personality": {k: cfg[k] for k in ("optimismo", "introversion", "energia")}},
        "narrative_self": narrative_self,
        "metacognition": {"coherence": coherence, "decisions": len(rows), "strong_goals": strong,
                          "weak_goals": weak,
                          "self_organization": _metacog.self_model.self_organization_score,
                          "proto_agency": _metacog.proto_agency},
        "self_query": self_query,
        "synthesis": " ".join(parts),
    })


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


# Límites de agix_simulate_agent: sin tope una llamada podía bloquear el servidor durante
# minutos. El coste crece con los pasos y, más deprisa, con los episodios cuando hay muchas
# observaciones (10000 episodios con 1000 observaciones tardaban 155 s). Peor caso permitido
# (100 observaciones, 20 acciones, 1000 episodios × 100 pasos): unos 2 s.
SIM_MAX_OBSERVATIONS = 100
SIM_MAX_ACTIONS = 20
SIM_MAX_EPISODES = 1000
SIM_MAX_STEPS = 10_000
SIM_MAX_TOTAL_STEPS = 100_000


@server.tool()
def agix_simulate_agent(observations: int = 10, actions: int = 4, episodes: int = 3,
                        max_steps: int = 5, seed: int | None = None) -> dict:
    """Simula un GeneticAgent de AGIX en ToyEnvironment con AutoAgent y devuelve sus métricas.

    Límites (para que no se cuelgue): observations ≤ 100, actions ≤ 20, episodes ≤ 1000,
    max_steps ≤ 10000 y episodes × max_steps ≤ 100000 (unos 2 s como mucho).
    """
    import random

    for name, value, top in (("observations", observations, SIM_MAX_OBSERVATIONS),
                             ("actions", actions, SIM_MAX_ACTIONS),
                             ("episodes", episodes, SIM_MAX_EPISODES),
                             ("max_steps", max_steps, SIM_MAX_STEPS)):
        if not 1 <= value <= top:
            raise ValueError(f"{name} debe estar entre 1 y {top} (recibido {value})")
    if episodes * max_steps > SIM_MAX_TOTAL_STEPS:
        raise ValueError(f"episodes × max_steps = {episodes * max_steps}; máximo {SIM_MAX_TOTAL_STEPS}")

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
