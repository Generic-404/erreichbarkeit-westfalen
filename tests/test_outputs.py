import json
import re

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from accessibility.outputs import _map_grid, write_stage_map


def test_map_shows_every_small_area_cell_and_aggregates_large_areas_without_holes():
    """Prüft die lückenlose Rasterdarstellung kleiner und großer Gebiete."""
    grid = gpd.GeoDataFrame(
        {
            "grid_id": ["a", "b", "c"],
            "center_x": [50, 150, 50],
            "center_y": [50, 50, 150],
            "status": ["complete"] * 3,
            "overall_score": [20, 40, 60],
        },
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100), box(0, 100, 100, 200)],
        crs="EPSG:3035",
    )

    full, cell_size = _map_grid(grid, ["overall_score"], max_cells=2)
    assert cell_size == 100
    assert len(full) == len(grid)

    grouped, cell_size = _map_grid(grid, ["overall_score"], max_cells=1, full_detail_limit=2)
    assert cell_size == 200
    assert len(grouped) == 1
    assert grouped.represented_cells.item() == 3
    assert grouped.overall_score.item() == pytest.approx(40)
    assert grouped.to_crs("EPSG:3035").geometry.area.item() == pytest.approx(30_000)


def test_map_can_switch_scores_census_and_individual_poi_categories(tmp_path):
    """Prüft Merkmalsauswahl, Punktkategorien und JavaScript-Syntax der Basiskarte."""
    grid = gpd.GeoDataFrame(
        {
            "grid_id": ["a", "b"],
            "center_x": [50, 150],
            "center_y": [50, 50],
            "status": ["complete", "complete"],
            "overall_score": [100.0, 0.0],
            "score_schools": [100.0, 0.0],
            "score_hospitals": [50.0, 0.0],
            "zensus_1_Durchschnittsalter": ["37,50", None],
            "zensus_2_AnteilUeber65": ["20,00", "–"],
            "zensus_3_AnteilUnter18": ["15,00", None],
            "zensus_4_Einwohner": ["42", None],
        },
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs="EPSG:3035",
    )
    stops = gpd.GeoDataFrame(
        {"stop_name": ["Haltestelle"]}, geometry=[Point(50, 50)], crs="EPSG:3035"
    )
    pois = gpd.GeoDataFrame(
        {
            "name": ["Schule", "Klinik"],
            "category_name": ["Schulen", "Krankenhäuser"],
            "category_id": ["schools", "hospitals"],
        },
        geometry=[Point(50, 50), Point(150, 50)],
        crs="EPSG:3035",
    )
    path = tmp_path / "map.html"

    report = write_stage_map(
        path, grid, stops, pois, {"schools": "Schulen", "hospitals": "Krankenhäuser"}
    )
    html = path.read_text()

    assert report["displayed_cells"] == 2
    assert report["poi_category_layers"] == 2
    assert isinstance(report["poi_category_layers"], int)
    assert len(report["census_layers"]) == 4
    assert report["population_color_saturation"] >= 10
    assert "POI: Schulen" in html and "POI: Krankenh\\u00e4user" in html
    assert "rasterPane" in html and "stopPane" in html and "poiPane" in html
    assert re.search(
        r'pointToLayer\(feature, latlng\)\s*\{\s*var opts = \{[^}]*"pane": "stopPane"', html
    )
    assert re.search(
        r'pointToLayer\(feature, latlng\)\s*\{\s*var opts = \{[^}]*"pane": "poiPane"', html
    )
    assert "Zensus: Durchschnittsalter (Jahre)" in html
    assert "Zensus: Anteil ab 65 (%)" in html
    assert "Zensus: Einwohner je 100-m-Zelle" in html
    assert "37.5" in html and "keine Angabe" in html
    assert "colors[index]" in html and "#081d58" in html


def test_large_map_loads_original_cells_from_tiles_on_zoom(tmp_path):
    """Prüft den Export vollständiger Detailzellen zum Nachladen bei großen Karten."""
    grid = gpd.GeoDataFrame(
        {
            "grid_id": ["west", "east"],
            "center_x": [50, 150],
            "center_y": [50, 50],
            "status": ["complete", "complete"],
            "overall_score": [100.0, 0.0],
            "score_schools": [75.0, 25.0],
            "zensus_4_Einwohner": ["42", None],
        },
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs="EPSG:3035",
    )
    empty = gpd.GeoDataFrame({"category_id": []}, geometry=[], crs="EPSG:3035")
    path = tmp_path / "accessibility_map.html"

    report = write_stage_map(
        path, grid, empty, empty, {"schools": "Schulen"}, max_cells=1, full_detail_limit=1
    )
    tiles = list((tmp_path / "grid_tiles").rglob("*.geojson"))
    cells = [feature for tile in tiles for feature in json.loads(tile.read_text())["features"]]

    assert report["display_aggregation"] and report["detail_tile_count"] == len(tiles) > 0
    assert report["detail_switch_zoom"] == 13
    assert {cell["properties"]["grid_id"] for cell in cells} == {"west", "east"}
    assert {cell["properties"]["overall_score"] for cell in cells} == {0.0, 100.0}
    assert {cell["properties"]["score_schools"] for cell in cells} == {25.0, 75.0}
    assert {cell["properties"]["zensus_4_Einwohner"] for cell in cells} == {42.0, None}
    html = path.read_text()
    assert "fetch('grid_tiles/'" in html and "makeCellDetails" in html
    assert "pane: 'rasterPane'" in html
