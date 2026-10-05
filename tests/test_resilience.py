from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from accessibility.config import RoutingConfig
from accessibility.routing import _batched_matrix, _filter_candidates, _load_checkpoint
from accessibility.stages import STEPS, StageStore
from accessibility.timepoints import ConcreteTimepoint


def _routing_config() -> RoutingConfig:
    """Erzeugt Routingparameter für kleine Tests der Blockverarbeitung."""
    return RoutingConfig(90, 15, 15, 4.5, True)


def test_completed_routing_batches_are_reused(tmp_path: Path, monkeypatch) -> None:
    """Prüft die Wiederverwendung vollständiger und passender Routingblöcke."""
    origins = gpd.GeoDataFrame(
        {"id": ["a", "b", "c"]},
        geometry=[Point(7.0, 51.0), Point(7.1, 51.1), Point(7.2, 51.2)],
        crs="EPSG:4326",
    )
    destinations = gpd.GeoDataFrame(
        {"id": ["x", "y"]}, geometry=[Point(7.3, 51.3), Point(7.4, 51.4)], crs="EPSG:4326"
    )
    point = ConcreteTimepoint(
        "2026-09-09T1200", "Mittwoch", date(2026, 9, 9), datetime(2026, 9, 9, 12), 1.0
    )
    calls: list[tuple[str, ...]] = []

    def fake_matrix(_r5py, _network, origin_batch, destination_batch, *_args):
        """Ersetzt R5 durch vorhersehbare Testreisezeiten und zählt die Blockaufrufe."""
        calls.append(tuple(origin_batch.id))
        return pd.DataFrame(
            [
                {"from_id": origin, "to_id": destination, "travel_time_minutes": 12.0}
                for origin in origin_batch.id
                for destination in destination_batch.id
            ]
        )

    monkeypatch.setattr("accessibility.routing._matrix", fake_matrix)
    kwargs = {
        "r5py": None,
        "network": None,
        "origins": origins,
        "destinations": destinations,
        "point": point,
        "routing": _routing_config(),
        "modes": ["TRANSIT", "WALK"],
        "departure_time_window_minutes": 5,
        "batch_size": 2,
        "checkpoint_directory": tmp_path,
        "checkpoint_signature": "matching-inputs",
    }
    first = _batched_matrix(**kwargs)
    second = _batched_matrix(**kwargs)

    assert calls == [("a", "b"), ("c",)]
    assert len(first) == len(second) == 6
    assert len(list(tmp_path.rglob("*.csv"))) == 2


def test_output_is_replaced_only_after_complete_export(tiny_project):
    """Prüft den Erhalt des bisherigen Exports bei Fehlern während eines neuen Exports."""
    store = StageStore(tiny_project)
    for step in STEPS:
        with store.transaction(step) as folder:
            (folder / "result.txt").write_text("original")
    with pytest.raises(RuntimeError):
        with store.transaction("export") as folder:
            (folder / "result.txt").write_text("partial")
            raise RuntimeError("export failed")
    assert (store.output / "result.txt").read_text() == "original"
    with store.transaction("export") as folder:
        (folder / "result.txt").write_text("new")
    assert (store.output / "result.txt").read_text() == "new"
    backups = list(store.output.parent.glob(".out.previous-*"))
    assert len(backups) == 1
    assert (backups[0] / "result.txt").read_text() == "original"


def test_matrix_union_does_not_add_foreign_candidates():
    """Prüft die Beschränkung zusammengeführter Reisezeiten auf erlaubte Kandidaten."""
    matrix = pd.DataFrame(
        {
            "from_id": ["a", "a", "b", "b"],
            "to_id": ["x", "y", "x", "y"],
            "travel_time_minutes": [30, 1, 1, 30],
        }
    )
    selected = _filter_candidates(matrix, {"a": {"x"}, "b": {"y"}})
    assert len(selected) == 2
    assert selected.travel_time_minutes.tolist() == [30, 30]


def test_partial_checkpoint_is_not_a_complete_result(tmp_path):
    """Prüft, dass ein unvollständiger Routingcache nicht als gültiges Ergebnis gilt."""
    path = tmp_path / "partial.csv"
    pd.DataFrame(
        {"from_id": ["a", "b"], "to_id": ["x", "x"], "travel_time_minutes": [10, 20]}
    ).to_csv(path, index=False)
    assert _load_checkpoint(path, {"a", "b"}, {"x", "y"}) is None


def test_r5_inputs_with_same_basename_remain_distinct_and_immutable(tmp_path):
    """Prüft getrennte R5-Kopien gleichnamiger Eingabedateien und deren Unveränderlichkeit."""
    from accessibility.routing import _r5_input_copy

    first, second = tmp_path / "city" / "network.osm.pbf", tmp_path / "region" / "network.osm.pbf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"city network")
    second.write_bytes(b"regional network")
    cache = tmp_path / "inputs"
    a = _r5_input_copy(first, cache, ".osm.pbf")
    b = _r5_input_copy(second, cache, ".osm.pbf")
    assert a.name != b.name
    assert a.read_bytes() == b"city network"
    assert b.read_bytes() == b"regional network"
    assert _r5_input_copy(first, cache, ".osm.pbf") == a
    first.write_bytes(b"updated network")
    assert _r5_input_copy(first, cache, ".osm.pbf") != a
    assert a.read_bytes() == b"city network"
