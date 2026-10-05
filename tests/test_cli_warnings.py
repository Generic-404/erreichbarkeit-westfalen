"""Prüft die gezielte Warnungsfilterung beim Start einer Routingphase."""

import sys
import warnings

import pytest

from accessibility import cli


@pytest.mark.parametrize("phase", ["route", "prepare"])
def test_only_routing_window_warning_is_filtered(tiny_project, monkeypatch, phase):
    """Andere Meldungen bleiben sichtbar; Konfiguration und globale Filter bleiben erhalten."""
    message = (
        "The provided departure time window is below 5 minutes. "
        "This may cause adverse effects with routing."
    )
    emitted = [
        (message, RuntimeWarning, "r5py.r5.regional_task"),
        ("Andere Routingwarnung", RuntimeWarning, "r5py.r5.regional_task"),
        (message, RuntimeWarning, "anderes_modul"),
        (message, UserWarning, "r5py.r5.regional_task"),
    ]
    received = []

    def fake_step(config, step):
        """Erzeugt Warnungen wie beim Routing, ohne eine Berechnung auszuführen."""
        received.append((config, step))
        for text, category, module in emitted:
            warnings.warn_explicit(
                text, category, filename="regional_task.py", lineno=250, module=module
            )
        return config.paths.output_directory

    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: tiny_project)
    monkeypatch.setattr(cli, "run_step", fake_step)
    original_argv = sys.argv[:]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        original_filters = warnings.filters[:]
        cli.main([phase, "--config", "config.yaml"])
        assert warnings.filters == original_filters
    expected = emitted[1:] if phase == "route" else emitted
    assert [(str(item.message), item.category) for item in captured] == [
        (text, category) for text, category, _ in expected
    ]
    assert received == [(tiny_project, phase)]
    assert sys.argv == original_argv
