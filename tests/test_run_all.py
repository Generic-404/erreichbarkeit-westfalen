"""Prüft Reihenfolge, gemeinsame Konfiguration und Fehlerabbruch des Gesamtskripts."""

import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner():
    """Lädt das Gesamtskript, ohne seinen Kommandozeilenaufruf auszuführen."""
    return runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_all.py"))


def test_runs_phases_with_shared_configuration(runner, monkeypatch, tmp_path):
    """Prüft Phasenreihenfolge und gezielte Weitergabe von Heapgröße und Kartenbeschriftung."""
    execute = Mock()
    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.chdir(tmp_path)
    runner["main"](["--config", "run.yaml", "--max-memory", "2G", "--area-label", "Testgebiet"])
    commands = [call.args[0] for call in execute.call_args_list]
    assert [command[3] for command in commands] == [
        "prepare",
        "route",
        "sources",
        "transfer",
        "score",
        "export",
        "evaluate",
    ]
    for command, call in zip(commands, execute.call_args_list, strict=True):
        assert command[:3] == [sys.executable, "-m", "accessibility"]
        assert command[4:6] == ["--config", str(tmp_path / "run.yaml")]
        assert call.kwargs == {"check": True}
        if command[3] == "route":
            assert command[6:] == ["--max-memory", "2G"]
        elif command[3] == "evaluate":
            assert command[6:] == ["--area-label", "Testgebiet"]
        else:
            assert len(command) == 6


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [(subprocess.CalledProcessError(2, "route"), 2), (KeyboardInterrupt(), 130)],
)
def test_failure_stops_before_later_phases(runner, monkeypatch, tmp_path, error, exit_code):
    """Prüft den Abbruch der Phasenfolge bei Prozessfehlern und Unterbrechungen."""
    execute = Mock(side_effect=[None, error])
    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(SystemExit) as result:
        runner["main"](["--config", str(tmp_path / "run.yaml")])
    assert result.value.code == exit_code
    assert [call.args[0][3] for call in execute.call_args_list] == ["prepare", "route"]
    assert execute.call_args.args[0][-2:] == ["--max-memory", "70%"]
