"""Marcas de tiempo: todas en UTC con la zona explícita, también las ya guardadas."""
import json
from datetime import datetime, timedelta, timezone


def _utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    assert dt.utcoffset() == timedelta(0), ts
    return dt


def test_todo_lo_nuevo_en_utc_explicito_y_coherente(server):
    antes = datetime.now(timezone.utc)
    ep = server.agix_memory_record("s", "d", "r", True)["stored"]["timestamp"]
    dec = server.agix_decision_record("a", True)["recorded"]["timestamp"]
    sueno = server.agix_dream_record("un sueño", "calma", 0.5)["stored"]["timestamp"]
    meta = server.agix_goal_register("meta", {"pro_vida": 0.9, "no_dano": 0.9, "respeto": 0.9})
    creada = meta["goal"]["timestamp_creacion"]
    despues = datetime.now(timezone.utc)
    for ts in (ep, dec, sueno, creada):
        assert antes - timedelta(seconds=1) <= _utc(ts) <= despues + timedelta(seconds=1)


def test_suenos_antiguos_en_hora_local_se_convierten(server, restart):
    server.agix_dream_record("viejo", "calma", 0.5)
    local = datetime(2026, 11, 16, 9, 6, 32)  # como lo guardaba DreamBank: hora local sin zona
    data = json.loads(open("suenos.json", encoding="utf-8").read())
    data[0]["timestamp"] = local.isoformat()
    open("suenos.json", "w", encoding="utf-8").write(json.dumps(data))
    s2 = restart(server)
    ts = s2.agix_dreams_recent()["raw"][0]["timestamp"]
    assert _utc(ts) == local.astimezone(timezone.utc)
    # y queda persistido ya convertido
    assert json.loads(open("suenos.json", encoding="utf-8").read())[0]["timestamp"] == ts


def test_episodios_y_decisiones_antiguos_sin_zona_se_leen_como_utc(server, restart):
    server.agix_memory_record("antiguo", "d", "r", True)
    server.agix_decision_record("antigua", True)
    mem = json.loads(open("memoria.json", encoding="utf-8").read())
    registros = mem if isinstance(mem, list) else mem.get("experiencias", mem)
    registros[0]["timestamp"] = "2026-11-16T08:05:48.594647"
    open("memoria.json", "w", encoding="utf-8").write(json.dumps(mem))
    meta = json.loads(open("metacognicion.json", encoding="utf-8").read())
    meta["decisions"][0]["timestamp"] = "2026-11-16T08:06:12.185835"
    open("metacognicion.json", "w", encoding="utf-8").write(json.dumps(meta))
    s2 = restart(server)
    s2.agix_memory_record("nuevo", "d", "r", True)  # mezcla antiguo + nuevo sin TypeError
    tl = s2.agix_memory_timeline("2026-11-16", "2026-11-16")
    assert [e["entrada"] for e in tl["episodes"]][0] == "antiguo"
    assert tl["episodes"][0]["timestamp"] == "2026-11-16T08:05:48.594647+00:00"
    assert s2.agix_metacognition_state()["recent"][0]["timestamp"] == "2026-11-16T08:06:12.185835+00:00"
    # un intervalo con zona explícita también funciona
    tl = s2.agix_memory_timeline("2026-11-16T09:00:00+01:00", "2026-11-16T09:10:00+01:00")
    assert tl["count"] == 1
