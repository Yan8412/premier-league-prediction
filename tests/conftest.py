import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_real_token(monkeypatch, tmp_path):
    """Tests never hit the network with a real token and never touch data/epl.db."""
    monkeypatch.delenv("SPORTMONKS_API_TOKEN", raising=False)
    monkeypatch.setenv("EPL_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr("epl_prediction.config.REPO_ROOT", tmp_path)
