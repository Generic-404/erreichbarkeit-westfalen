"""Explizit starten: ACCESSIBILITY_R5_TEST=1 python -m pytest tests/test_r5_integration.py -q."""

import os
from dataclasses import replace
from datetime import datetime, timedelta

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from accessibility.pipeline import run_step
from accessibility.r5_adapter import minimum_transit_matrix
from accessibility.stages import STEPS, StageStore

pytestmark = pytest.mark.skipif(
    os.environ.get("ACCESSIBILITY_R5_TEST") != "1",
    reason="R5/JVM-Integration ausdrücklich aktivieren",
)


def test_transit_minimum_and_no_walk_fallback(tiny_project):
    """Prüft echte R5-Intervallminima mit Wartezeit und ohne Ersatz durch direkte Gehwege."""
    import r5py

    network = r5py.TransportNetwork(str(tiny_project.paths.osm_pbf), [str(tiny_project.paths.gtfs)])
    origins = gpd.GeoDataFrame({"id": ["start"]}, geometry=[Point(7.002, 52.002)], crs="EPSG:4326")
    destinations = gpd.GeoDataFrame(
        {"id": ["poi"]}, geometry=[Point(7.010, 52.002)], crs="EPSG:4326"
    )
    arguments = dict(
        origins=origins,
        destinations=destinations,
        snap_to_network=True,
        departure=datetime(2026, 9, 9, 10),
        departure_time_window=timedelta(hours=4),
        percentiles=[50],
        transport_modes=["TRANSIT"],
        access_modes=["WALK"],
        egress_modes=["WALK"],
        max_time=timedelta(minutes=61),
        max_time_walking=timedelta(minutes=15),
        speed_walking=4.5,
    )
    result = minimum_transit_matrix(r5py, network, **arguments)
    value = result.travel_time.item()
    # Die Busfahrt im Testfahrplan dauert 8 Minuten; direkte Gehwege sind ausgeschlossen.
    assert 8 <= value <= 12
    # Start genau 10:00: der Bus fährt erst 10:10. Die Wartezeit zählt zur Reisezeit.
    fixed_start = minimum_transit_matrix(
        r5py, network, **{**arguments, "departure_time_window": timedelta(minutes=1)}
    )
    assert 18 <= fixed_start.travel_time.item() <= 22
    walks = r5py.TravelTimeMatrix(network, **{**arguments, "transport_modes": ["WALK"]})
    assert walks.travel_time.item() < value
    later = minimum_transit_matrix(
        r5py, network, **{**arguments, "departure": datetime(2026, 9, 9, 15)}
    )
    assert pd.isna(
        later.travel_time.item()
    )  # Zu Fuß weiterhin erreichbar, aber kein Bus im Intervall.
    # Vergleich mit einzeln ausgewerteten Abfahrtsminuten bestätigt die Minimumbildung.
    minute_values = []
    for minute in (8, 9, 10):
        one = minimum_transit_matrix(
            r5py,
            network,
            **{
                **arguments,
                "departure": datetime(2026, 9, 9, 10, minute),
                "departure_time_window": timedelta(minutes=1),
            },
        )
        minute_values.extend(one.travel_time.dropna().tolist())
    assert value == min(minute_values)


def test_all_six_steps_and_scoring_without_rerouting(tiny_project, monkeypatch):
    """Prüft die gesamte Rechenkette und erneutes Scoring ohne erneutes Routing."""
    for step in STEPS:
        run_step(tiny_project, step)
    store = StageStore(tiny_project)
    assert all(s["status"] == "complete" for s in store.inspect())
    grid = pd.read_csv(store.output / "grid_scores.csv")
    assert grid.status.eq("complete").all()
    assert grid.overall_score.between(0, 100).all()
    html = (store.output / "accessibility_map.html").read_text()
    assert "Gesamtscore" in html
    import accessibility.pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "route_stop_to_pois", lambda *args, **kwargs: pytest.fail("Unerwartetes Routing")
    )
    new_categories = tuple(
        replace(c, weight=60 if i == 0 else 40) for i, c in enumerate(tiny_project.poi_categories)
    )
    config = replace(tiny_project, poi_categories=new_categories)
    run_step(config, "score")
    assert StageStore(config).inspect()[-1]["status"] == "stale"
    run_step(config, "export")
    assert StageStore(config).inspect()[-1]["status"] == "complete"
