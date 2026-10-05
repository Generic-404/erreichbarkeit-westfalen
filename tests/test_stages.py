from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from accessibility.cli import _parser
from accessibility.config import load_config
from accessibility.exceptions import DataValidationError
from accessibility.osm import matches_filters
from accessibility.pipeline import run_step
from accessibility.routing import _nearest_candidate_ids, hilbert_blocks
from accessibility.stages import STEPS, StageStore


def test_cli_requires_single_explicit_step():
    """Prüft, dass die Kommandozeile die Auswahl einer einzelnen Phase verlangt."""
    assert _parser().parse_args(["score", "--config", "x.yaml"]).command == "score"
    with pytest.raises(SystemExit):
        _parser().parse_args(["calculate", "--config", "x.yaml"])


def test_swimming_filters_and_healthcare_alias():
    """Prüft die konfigurierten Schwimmbadfilter und alternativen Gesundheitstags."""
    config = load_config(
        Path(__file__).parents[1] / "config/westfalen.yaml", validate_input_files=False
    )
    rules = {c.id: c.osm_filters for c in config.poi_categories}
    assert matches_filters(
        {"leisure": "sports_hall", "sport": "fitness;swimming"}, rules["swimming"]
    )
    assert matches_filters({"leisure": "water_park"}, rules["swimming"])
    assert not matches_filters({"leisure": "swimming_pool"}, rules["swimming"])
    assert not matches_filters({"leisure": "sports_hall", "sport": "fitness"}, rules["swimming"])
    assert matches_filters({"healthcare": "pharmacy"}, rules["pharmacies"])


def test_hilbert_blocks_have_25_stops_and_stable_order():
    """Prüft Blockgrößen und stabile räumliche Reihenfolge der Haltestellen."""
    stops = gpd.GeoDataFrame(
        {"stop_group_id": [str(i) for i in range(61)]},
        geometry=[Point(i % 9, i // 9) for i in range(61)],
        crs="EPSG:25832",
    )
    first = hilbert_blocks(stops, 25)
    second = hilbert_blocks(stops.sample(frac=1, random_state=9), 25)
    assert first.groupby("block_id").size().tolist() == [25, 25, 11]
    assert first.stop_group_id.tolist() == second.stop_group_id.tolist()
    assert first.hilbert_order.is_monotonic_increasing


def test_single_stop_gets_multiple_global_candidates():
    """Prüft die Auswahl mehrerer Kandidaten auch bei nur einer Haltestelle."""
    stops = gpd.GeoDataFrame({"stop_group_id": ["s"]}, geometry=[Point(0, 0)], crs="EPSG:25832")
    pois = gpd.GeoDataFrame(
        {"poi_id": ["a", "b", "c"], "category_id": ["x"] * 3},
        geometry=[Point(2, 0), Point(1, 0), Point(3, 0)],
        crs=stops.crs,
    )
    candidates, _ = _nearest_candidate_ids(stops, pois, {"x": 2})
    assert candidates["s"] == {"a", "b"}


def _complete_placeholder(store, step):
    """Schreibt eine minimale erfolgreiche Phasenausgabe für Tests der Abhängigkeiten."""
    with store.transaction(step) as directory:
        (directory / "result.txt").write_text(step)


def test_order_dependency_invalidation_and_no_automatic_steps(tiny_project):
    """Prüft die Phasenabhängigkeiten und Statusänderungen bei angepassten Einstellungen."""
    store = StageStore(tiny_project)
    with pytest.raises(DataValidationError, match="benötigt.*prepare"):
        with store.transaction("route"):
            pass
    assert not store.directory("prepare").exists()
    for step in STEPS:
        _complete_placeholder(store, step)
    assert all(s["status"] == "complete" for s in store.inspect())
    categories = tuple(
        replace(c, weight=60 if i == 0 else 40) for i, c in enumerate(tiny_project.poi_categories)
    )
    changed = StageStore(replace(tiny_project, poi_categories=categories))
    states = {s["step"]: s["status"] for s in changed.inspect()}
    assert [states[s] for s in STEPS[:4]] == ["complete"] * 4
    assert states["score"] == states["export"] == "stale"
    changed = StageStore(
        replace(tiny_project, surface=replace(tiny_project.surface, airline_detour_factor=1.4))
    )
    states = {s["step"]: s["status"] for s in changed.inspect()}
    assert states["route"] == states["sources"] == "complete"
    assert states["transfer"] == "stale"
    _complete_placeholder(store, "route")
    assert next(s for s in store.inspect() if s["step"] == "sources")["status"] == "stale"


def test_partial_failure_not_marked_complete(tiny_project):
    """Prüft, dass fehlgeschlagene Phasen kein gültiges Abschlussmanifest erhalten."""
    store = StageStore(tiny_project)
    with pytest.raises(RuntimeError):
        with store.transaction("prepare") as folder:
            (folder / "partial").write_text("incomplete")
            raise RuntimeError("simulated interruption")
    assert not store.directory("prepare").exists()
    assert store.inspect()[0]["last_attempt"]["status"] == "failed"


def test_prepare_is_separate_and_keeps_all_grid_cells(tiny_project):
    """Prüft die Vorbereitung als einzelne Phase einschließlich aller Rasterzellen."""
    folder = run_step(tiny_project, "prepare")
    assert len(gpd.read_file(folder / "pois.gpkg")) == 2
    assert len(gpd.read_file(folder / "stops.gpkg")) == 2
    assert len(gpd.read_file(folder / "grid.gpkg")) > 1
    assert StageStore(tiny_project).inspect()[1]["status"] == "missing"


def test_routing_failure_never_becomes_zero_score(tiny_project, monkeypatch):
    """Prüft, dass Routingfehler abbrechen und keine scheinbar gültigen Nullscores erzeugen."""
    run_step(tiny_project, "prepare")

    def failure(*args, **kwargs):
        """Simuliert einen Fehler des Routingdienstes."""
        raise RuntimeError("JVM error")

    monkeypatch.setattr("accessibility.pipeline.route_stop_to_pois", failure)
    with pytest.raises(RuntimeError, match="JVM error"):
        run_step(tiny_project, "route")
    with pytest.raises(DataValidationError, match="benötigt.*route"):
        run_step(tiny_project, "sources")
    assert not StageStore(tiny_project).directory("score").exists()
