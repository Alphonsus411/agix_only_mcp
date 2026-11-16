# agix-mcp

Servidor MCP (stdio) que expone [AGIX](https://pypi.org/project/agix/) como herramientas para Claude.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
claude mcp add --scope user agix -e PYTHONIOENCODING=utf-8 -- "%CD%\.venv\Scripts\python.exe" "%CD%\server.py"
```

Datos persistentes en `%USERPROFILE%\.agix-mcp` (cambiable con `AGIX_MCP_DATA`):
`memoria.json` (experiencias), `ontologia.json` (conceptos), `qualia_config.json`,
`qualia_estado.json` + `qualia_modelos.json` (espíritu Qualia y sus pesos aprendidos), `suenos.json`
`autonarrativa.db` + `autonarrativa_scores.json`, `metacognicion.json` e `identidad.json`.

## Hooks y skill (uso automático)

- **`hooks/agix_context.py`**: solo lee `~/.agix-mcp` (no importa agix, ~150 ms) y no imprime nada
  si no hay datos útiles.
  - `session` (SessionStart): identidad, tono de Qualia, coherencia, errores repetidos, últimas
    lecciones, valores/metas de la autonarrativa.
  - `prompt` (UserPromptSubmit): hasta 3 episodios, decisiones o entradas narrativas que comparten
    al menos 2 palabras clave (sin tildes) con el mensaje.
  - Para activarlos, añade el bloque de `hooks/settings-snippet.json` a `~/.claude/settings.json`
    (ajusta las rutas si clonas el repo en otro sitio). Se cargan en las sesiones nuevas.
- **`skill/agix/SKILL.md`**: cuándo usar cada herramienta (consultar al empezar, verificar al razonar,
  registrar al terminar, qué no registrar). Instalación: copiar la carpeta a `~/.claude/skills/agix/`.

## Herramientas

| Herramienta | Base en AGIX | Uso |
|---|---|---|
| `agix_logic_infer` | `Fact` + motor propio | Encadenamiento hacia delante hasta punto fijo, variables, negación por fallo, `transitive`/`symmetric`, `query`, traza de derivación |
| `agix_verify_expression` | `evaluation.formal.verifier` | sat/unsat en dominio {-1,0,1} |
| `agix_fuzzy_infer` | scikit-fuzzy | Mamdani con términos bajo/medio/alto |
| `agix_nash_equilibria` | nashpy | Equilibrios de Nash de cualquier juego bimatricial |
| `agix_memory_record/search/recent` | `EpisodicMemory` (+ `GestorDeMemoria`) | Memoria episódica persistente (búsqueda por palabras clave sin tildes + similitud) |
| `agix_memory_timeline` | `EpisodicMemory` | Episodios entre fechas (UTC), en orden y con balance de aciertos |
| `agix_memory_narrative` | `NarrativeMemoryTree` | Resumen narrativo jerárquico con pérdida (se reconstruye al arrancar) |
| `agix_narrative_store/query` | `AutonarrativeCore` | Autonarrativa en SQLite con valores, metas y afectos; búsqueda top-k puntuada |
| `agix_narrative_self` / `agix_narrative_evaluate_action` | `AutonarrativeCore.SelfModel` | Modelo de sí mismo y encaje de una acción con esa identidad |
| `agix_ontology_add/query` | `memory.symbolic.Ontology` | Grafo persistente, cierre transitivo de relaciones y herencia de atributos |
| `agix_ethics_judge` | `AlignmentInterface` + marcos | Eco, utilitarista, deontológico, virtudes |
| `agix_intention_evaluate` | `IntentionEvaluator` | Consecuencias, conflicto de metas, conducta |
| `agix_meta_evaluate` | `DynamicMetaEvaluator` | Coherencia, ética, tono afectivo (historial persistente compartido) |
| `agix_decision_record` | `MetacognitionManager` | Decisión + resultado + lección + meta + módulos implicados |
| `agix_decision_review` | `DecisionReviewer` | Aciertos/errores, lecciones reales, acierto por acción y meta, errores repetidos; se guarda en la autonarrativa |
| `agix_metacognition_state` | `MetacognitionManager` + `SelfOrganizationMonitor` | Coherencia, auto-organización y componentes, proto-agencia |
| `agix_identity_define` / `agix_identity_event` | `SyntheticIdentity` + `SelfModel` | Identidad persistente: nombre, origen, atributos, rasgos e hitos (opcionalmente a la autonarrativa) |
| `agix_identity_portrait` | Todos | Autorretrato unificado: identidad + Qualia + autonarrativa + metacognición, con síntesis en texto |
| `agix_resolve_internal_conflict` | `MetaReflectionEngine` | Deliberación entre hipótesis reforzando las respaldadas por la memoria episódica |
| `agix_goal_register` | `GoalEngine` | Metas con filtro ético (< 0.6 se rechaza) |
| `agix_emotion_update` | `EmotionSimulator` | Estado PAD puntual |
| `agix_qualia_experience/state` | `QualiaSpirit` | Espíritu persistente: emociones, modo, predicción RNN, asociaciones hebbianas, diario, introspección |
| `agix_qualia_configure` | `QualiaSpirit` + `PersonalityProfile` | Atenuación y plasticidad en caliente; nombre/personalidad con `reset` |
| `agix_cognitive_cycle` | `SyntheticLoop` | Percepción → emoción PAD → ética → control/aborto → metarreflexión |
| `agix_aesthetic_decide` | `QualiaSpiritCore` + `AestheticEngine` | Decisión por armonía con explicación |
| `agix_dream_record` / `agix_dreams_recent` | `DreamBank` | Banco de sueños persistente |
| `agix_fagi_index`, `agix_harmony`, `agix_simulate_agent` | `evaluation`, `AutoAgent` | Métricas y simulación |

## Correcciones locales de AGIX (solo en este `.venv`)

La copia de agix instalada en `.venv` lleva correcciones que **no están en el repositorio de AGIX**.
Un `pip install --force-reinstall agix` las borraría; para reaplicarlas, usa `patches/qualia-fixes.patch`
(rutas `src/agix/...`, pensado para aplicarse en agi_core). Originales en `patches/orig/`.
Tests: `.venv\Scripts\python -m pytest tests -q`.

- `qualia/neuro_plastic.py`: la plasticidad parte de la identidad (antes pesos a 0 → anulaba todas las
  emociones tras la primera experiencia), normaliza por filas y admite emociones nuevas.
- `qualia/emotional_rnn.py` + `qualia/spirit.py`: `EmotionalRNN.expand_inputs` y crecimiento del predictor
  al aparecer emociones nuevas (antes se perdían).
- `orchestrator/synthetic_loop.py`: pesos PAD por defecto (`DEFAULT_PAD_WEIGHTS`, parámetro
  `emotion_simulator`), estado `qualia` persistente, las entradas ya no se aplican hasta 3 veces por ciclo
  y el foco de atención no se acumula como modulador.

- `qualia/spirit.py`: nuevo parámetro `atenuacion` (por defecto 0.9). Antes de cada experiencia las
  emociones previas se multiplican por ese factor, así que el espíritu sale de la desconexión
  (con 0.9, tras ~7 experiencias positivas; 0.85 → 4; 0.95 → 14). `atenuacion=1.0` mantiene el
  comportamiento original, en el que tristeza + miedo > 0.8 dejaba la desconexión permanente.

## Notas sobre AGIX 1.11.0 detectadas al integrarlo

- `agix razonar` solo deriva un hecho por llamada (no alcanza el cierre); por eso `agix_logic_infer` usa su propio bucle.
- Importación circular: hay que importar `agix.memory` antes de `agix.reasoning.neuro_symbolic`.
- `agix.memory.experiential.PADState` es un stub y `guardar()` falla si `emocion` es un `PADState`; aquí se pasa como dict.
- `NeuroSymbolicBridge`/`ConceptClassifier` usan vectores aleatorios (no semánticos); `QualiaEngine` requiere torch/jax.
- `EthicalDilemmaSimulator.simulate_scenario` ignora el texto de entrada.
- `GestorDeMemoria.buscar_similar` usa difflib sobre la cadena completa: casi no encuentra consultas cortas.
- `AutonarrativeCore` persiste las experiencias en SQLite pero no su `SelfModel`: el MCP lo reconstruye
  desde la base de datos al arrancar. Su embedding son frecuencias de letras ASCII (ignora ñ y tildes) y
  `query_experiences` devuelve todo sin puntuación; el MCP añade coincidencia de palabras y top-k.
  La conexión se abre en el hilo de importación; el MCP la reabre con `check_same_thread=False` y un lock.
- `MetacognitionManager` guarda sus decisiones (`ReflexiveLogic`) y la auto-organización solo en
  memoria, y su `AutonarrativeCore` es `:memory:`; el MCP persiste ambos y conecta la autonarrativa real.
  `DecisionReviewer` genera lecciones genéricas ("Mejorar acción 'X'"); el MCP conserva la lección real.
- `NarrativeMemoryTree` no se persiste y descarta episodios intermedios al llenarse (K=2, D=3 conserva
  los primeros y los últimos); el MCP lo reconstruye reinsertando los episodios en orden.
