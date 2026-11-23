"""Almacenamiento seguro de la memoria de AGIX entre varias sesiones a la vez.

Cada sesión de Claude Code arranca su propio servidor, y todos escriben en la misma carpeta.
Antes, cada uno guardaba su copia en memoria y reescribía el archivo entero: el último en
escribir borraba lo que habían registrado los demás. Aquí:

- lock(): bloqueo entre procesos (y reentrante dentro del proceso) sobre .agix.lock.
- write_text() / write_json(): escritura atómica (temporal + os.replace), con reintentos porque
  en Windows el reemplazo falla mientras otro proceso (p. ej. un hook) tiene el archivo abierto.
- read_json(): si el archivo está corrupto lo aparta como <nombre>.corrupto-<fecha> y lo
  anota en problems; nunca se sobrescribe un archivo ilegible.
- daily_backup(): copia diaria de los datos en backups/AAAA-MM-DD (se conservan 7).

Sin dependencias fuera de la biblioteca estándar.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOCK_FILE = ".agix.lock"
LOCK_TIMEOUT = 15.0
BACKUP_DIR = "backups"
BACKUP_KEEP = 7

problems: list[str] = []  # avisos de esta ejecución (archivos apartados, copias fallidas...)

_rlock = threading.RLock()
_local = threading.local()


class StorageBusy(RuntimeError):
    pass


def _try_lock(fh) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fh) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def lock(data_dir: Path, timeout: float = LOCK_TIMEOUT):
    """Bloqueo exclusivo de una carpeta de datos entre procesos; reentrante por carpeta en el hilo.

    Cada carpeta tiene su propio bloqueo: tomar el de la memoria del proyecto y, dentro, el de
    la Qualia global bloquea las dos. Orden fijo para no provocar interbloqueos: proyecto y
    después Qualia, nunca al revés."""
    with _rlock:
        held = getattr(_local, "held", None)
        if held is None:
            held = _local.held = {}
        key = os.path.normcase(str(Path(data_dir).resolve()))
        if held.get(key):
            held[key] += 1
            try:
                yield
            finally:
                held[key] -= 1
            return
        path = Path(data_dir) / LOCK_FILE
        fh = open(path, "a+b")
        try:
            deadline = time.monotonic() + timeout
            while not _try_lock(fh):
                if time.monotonic() > deadline:
                    raise StorageBusy("La memoria de AGIX está ocupada por otra sesión; reinténtalo")
                time.sleep(0.05)
            held[key] = 1
            try:
                yield
            finally:
                held[key] = 0
                _unlock(fh)
        finally:
            fh.close()


def write_text(path: Path, text: str, retries: int = 40) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: alguien tiene el archivo abierto ahora mismo
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def write_json(path: Path, data, indent: int | None = 2) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=indent, default=str))


def quarantine(path: Path, why: str) -> Path | None:
    path = Path(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    target = path.with_name(f"{path.name}.corrupto-{stamp}")
    try:
        os.replace(path, target)
    except OSError as exc:
        problems.append(f"{path.name} está corrupto ({why}) y no se pudo apartar: {exc}")
        return None
    problems.append(f"{path.name} estaba corrupto ({why}); apartado como {target.name}. "
                    "Puedes recuperarlo de backups/.")
    return target


def read_json(path: Path, default=None):
    """Lee un JSON; si no existe devuelve default; si está corrupto lo aparta y devuelve default."""
    path = Path(path)
    if not path.exists():
        return default
    for attempt in range(20):
        try:
            text = path.read_text(encoding="utf-8")
            break
        except PermissionError:  # Windows: se está reemplazando en este instante
            time.sleep(0.05)
    else:
        raise StorageBusy(f"No se pudo leer {path.name}")
    try:
        return json.loads(text)
    except ValueError as exc:
        quarantine(path, str(exc).split(":")[0])
        return default


def mtime(path: Path) -> float | None:
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return None


def daily_backup(data_dir: Path, keep: int = BACKUP_KEEP) -> Path | None:
    """Copia los datos en backups/AAAA-MM-DD si aún no hay copia de hoy; conserva `keep` días."""
    data_dir = Path(data_dir)
    root = data_dir / BACKUP_DIR
    today = root / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today.exists():
        return None
    files = [p for p in data_dir.iterdir() if p.is_file() and p.name != LOCK_FILE
             and not p.name.endswith(".tmp") and ".corrupto-" not in p.name]

    def valid(p: Path) -> bool:  # una copia de un JSON roto no sirve: se queda fuera
        if p.suffix != ".json":
            return True
        try:
            json.loads(p.read_text(encoding="utf-8"))
            return True
        except (OSError, ValueError):
            return False

    files = [p for p in files if valid(p)]
    if not files:
        return None
    partial = today.with_name(today.name + ".parcial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        for p in files:
            if p.suffix == ".db":  # copia coherente de SQLite aunque esté abierta
                src = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
                dst = sqlite3.connect(partial / p.name)
                try:
                    src.backup(dst)
                finally:
                    src.close()
                    dst.close()
            else:
                shutil.copy2(p, partial / p.name)
        os.replace(partial, today)
    except Exception as exc:
        problems.append(f"No se pudo hacer la copia de seguridad diaria: {exc}")
        shutil.rmtree(partial, ignore_errors=True)
        return None
    days = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith(".parcial"))
    for old in days[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return today
