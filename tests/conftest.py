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


@pytest.fixture()
def restart():
    """Simula un reinicio del servidor MCP con el mismo directorio de datos."""
    return importlib.reload
