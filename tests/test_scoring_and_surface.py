import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from accessibility.census import (
    GRID_CRS,
    attach_census_attributes,
    generate_census_grid,
)
from accessibility.exceptions import DataValidationError
from accessibility.pipeline import _routing_stops
from accessibility.scoring import score_cells, score_from_minutes
from accessibility.surface import SOURCE_COLUMNS, build_sources, transfer_times


def test_exponential_boundaries_and_unknown():
    """Prüft die Scorefunktion an ihren Grenzen und bei unbekannter Reisezeit."""
    assert score_from_minutes(0) == 100
    assert score_from_minutes(9.999) == 100
    assert score_from_minutes(10) == pytest.approx(146.11 * math.exp(-0.39))
    assert score_from_minutes(60) == pytest.approx(146.11 * math.exp(-2.34))
    assert score_from_minutes(60.001) == 0
    assert score_from_minutes(float("inf")) == 0
    assert math.isnan(score_from_minutes(None))
    with pytest.raises(DataValidationError):
        score_from_minutes(-1)


def test_missing_category_is_not_renormalized():
    """Prüft feste Kategoriegewichte bei fehlenden Ergebnissen einzelner Kategorien."""
    times = pd.DataFrame(
        {
            "grid_id": ["a", "a", "b", "b"],
            "category_id": ["x", "y", "x", "y"],
            "travel_time_minutes": [0, np.inf, 0, np.nan],
            "status": ["ok", "no_source_within_limits", "ok", "unknown_data"],
        }
    )
    _, summary = score_cells(times, {"x": 25, "y": 75})
    assert summary.loc[summary.grid_id.eq("a"), "overall_score"].item() == 25
    assert pd.isna(summary.loc[summary.grid_id.eq("b"), "overall_score"].item())
    with pytest.raises(DataValidationError):
        score_cells(times, {"x": 25, "y": 70})


def test_grid_uses_global_zensus_alignment_and_covers_edge():
    """Prüft Zensusausrichtung, Zellkennungen und die Aufnahme von Randzellen."""
    boundary = gpd.GeoDataFrame(geometry=[box(10, 10, 240, 190)], crs=GRID_CRS)
    grid = generate_census_grid(boundary)
    assert len(grid) == 6
    assert (grid.geometry.area == 10_000).all()
    assert "CRS3035RES100mN0E0" in set(grid.grid_id)
    assert (grid.center_x % 100 == 50).all()


def test_grid_matches_positive_area_reference_with_holes_and_touching_edges():
    """Vergleicht das Raster mit einer Flächenschnittreferenz einschließlich Löchern."""
    shape = MultiPolygon(
        [
            Polygon(
                [(0, 0), (500, 0), (400, 500), (0, 500)],
                holes=[[(100, 100), (100, 300), (300, 300), (300, 100)]],
            ),
            box(600, 200, 700, 400),
        ]
    )
    boundary = gpd.GeoDataFrame(geometry=[shape], crs=GRID_CRS)
    grid = generate_census_grid(boundary)
    expected = {
        f"CRS3035RES100mN{y}E{x}"
        for x in range(0, 700, 100)
        for y in range(0, 500, 100)
        if box(x, y, x + 100, y + 100).intersection(shape).area > 0
    }
    assert set(grid.grid_id) == expected
    assert (grid.geometry.area == 10_000).all()


def test_only_stops_whose_source_cells_reach_the_output_grid_are_routed():
    """Prüft die Auswahl der Haltestellen anhand erreichbarer Ergebniszellen."""
    grid = generate_census_grid(gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=GRID_CRS))
    stops = gpd.GeoDataFrame(
        {"stop_group_id": ["own", "edge", "outside"]},
        geometry=[Point(50, 50), Point(950, 50), Point(1050, 50)],
        crs=GRID_CRS,
    )
    assert _routing_stops(stops, grid, 900).stop_group_id.tolist() == ["own", "edge"]


def _grid():
    """Erzeugt eine Reihe benachbarter Rasterzellen für die Reisezeitübertragung."""
    return generate_census_grid(gpd.GeoDataFrame(geometry=[box(0, 0, 1100, 100)], crs=GRID_CRS))


def test_poi_without_stop_is_original_source_and_stops_can_improve():
    """Prüft direkte Gehquellen und mögliche Verbesserungen durch Haltestellenquellen."""
    stops = gpd.GeoDataFrame({"stop_group_id": ["slow"]}, geometry=[Point(151, 51)], crs=GRID_CRS)
    pois = gpd.GeoDataFrame(
        {"poi_id": ["p"], "category_id": ["x"]}, geometry=[Point(51, 51)], crs=GRID_CRS
    )
    times = pd.DataFrame(
        {
            "stop_group_id": ["slow"],
            "category_id": ["x"],
            "travel_time_minutes": [30],
            "poi_id": ["p"],
        }
    )
    sources = build_sources(stops, pois, times)
    result = pd.concat(
        transfer_times(_grid(), sources, {"x": "available"}, 15, 4.5, 1.25)
    ).set_index("grid_id")
    assert result.loc["CRS3035RES100mN0E0", "travel_time_minutes"] == 0
    assert result.loc["CRS3035RES100mN0E100", "source_type"] == "direct_walk"
    assert result.loc["CRS3035RES100mN0E100", "travel_time_minutes"] == pytest.approx(100 / 60)
    assert result.loc["CRS3035RES100mN0E900", "access_minutes"] == 15
    # Nach 900 m endet die Gehquelle; die ursprüngliche Haltestelle bleibt zulässig.
    assert result.loc["CRS3035RES100mN0E1000", "source_type"] == "transit"


def test_no_recursive_propagation():
    """Prüft, dass übertragene Reisezeiten nicht selbst zu neuen Quellen werden."""
    source = pd.DataFrame(
        [["CRS3035RES100mN0E0", "x", 0, "direct_walk", "", "p", 50, 50]], columns=SOURCE_COLUMNS
    )
    result = pd.concat(transfer_times(_grid(), source, {"x": "available"}, 15, 4.5, 1.25))
    assert np.isinf(
        result.loc[result.grid_id.eq("CRS3035RES100mN0E1000"), "travel_time_minutes"].item()
    )


def test_best_not_nearest_source_and_unknown_data():
    """Prüft die Auswahl der schnellsten Quelle und den Erhalt unbekannter Kategorien."""
    sources = pd.DataFrame(
        [
            ["slow", "x", 30, "transit", "s", "p", 50, 50],
            ["fast", "x", 5, "transit", "f", "p", 250, 50],
        ],
        columns=SOURCE_COLUMNS,
    )
    result = pd.concat(
        transfer_times(_grid().iloc[:1], sources, {"x": "available", "y": "unknown"}, 15, 4.5, 1.25)
    )
    assert result.loc[result.category_id.eq("x"), "source_grid_id"].item() == "fast"
    assert pd.isna(result.loc[result.category_id.eq("y"), "travel_time_minutes"].item())


def test_census_join_keeps_cells_without_demographics(tmp_path):
    """Prüft, dass fehlende Zensusattribute keine Rasterzellen entfernen."""
    path = tmp_path / "census.csv"
    path.write_text("GITTER_ID_100m;x_mp_100m;y_mp_100m;Alter\nCRS3035RES100mN0E0;50;50;42,5\n")
    grid, report = attach_census_attributes(_grid(), (path,))
    assert len(grid) == 11
    assert report[0]["matched_cells"] == 1
    path.write_text("GITTER_ID_100m;x_mp_100m;y_mp_100m\nCRS3035RES100mN0E0;0;0\n")
    with pytest.raises(DataValidationError, match="Mittelpunkte"):
        attach_census_attributes(_grid(), (path,))


def test_empty_sources_preserve_unknown_vs_known_absence():
    """Prüft die Unterscheidung fehlender Daten und bekannter Nichterreichbarkeit."""
    stops = gpd.GeoDataFrame({"stop_group_id": []}, geometry=[], crs=GRID_CRS)
    pois = gpd.GeoDataFrame({"poi_id": [], "category_id": []}, geometry=[], crs=GRID_CRS)
    travel = pd.DataFrame(columns=["stop_group_id", "category_id", "travel_time_minutes", "poi_id"])
    sources = build_sources(stops, pois, travel)
    result = pd.concat(
        transfer_times(_grid(), sources, {"x": "available", "y": "unknown"}, 15, 4.5, 1.25)
    )
    assert result.loc[result.category_id.eq("x"), "travel_time_minutes"].eq(np.inf).all()
    assert result.loc[result.category_id.eq("y"), "travel_time_minutes"].isna().all()
