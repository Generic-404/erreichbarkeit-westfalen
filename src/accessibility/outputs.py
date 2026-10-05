"""CSV-, GeoPackage- und Folium-Ausgabe."""

from __future__ import annotations

import json
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from branca.colormap import linear
from branca.element import MacroElement, Template
from pyproj import Transformer

DETAIL_TILE_ZOOM = 13
DETAIL_SWITCH_ZOOM = 13


def _point_marker(radius: int, pane: str, **options) -> folium.CircleMarker:
    """Erzeugt einen Kreismarker auf der angegebenen Leaflet-Zeichenebene."""
    marker = folium.CircleMarker(radius=radius, **options)
    # Die Zeichenebene wird als pane-Option direkt an den Leaflet-Marker übergeben.
    marker.options["pane"] = pane
    return marker


def _write_detail_tiles(output_path: Path, grid: gpd.GeoDataFrame, fields: list[str]) -> list[str]:
    """Vollständige 100-m-Zellen in kleine, bei Bedarf abrufbare Kartendateien teilen."""
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(grid.center_x.to_numpy(), grid.center_y.to_numpy())
    latitude = np.clip(latitude, -85.05112878, 85.05112878)
    count = 1 << DETAIL_TILE_ZOOM
    tile_x = np.clip(np.floor((longitude + 180) / 360 * count).astype(np.int32), 0, count - 1)
    tile_y = np.clip(
        np.floor((1 - np.arcsinh(np.tan(np.deg2rad(latitude))) / np.pi) / 2 * count).astype(
            np.int32
        ),
        0,
        count - 1,
    )
    # Detaildateien enthalten die vollständigen ursprünglichen Zellen.
    # Die Kachelzuordnung dient nur dem gezielten Nachladen im Browser.
    full = grid[["grid_id", "status", *fields, "geometry"]].to_crs("EPSG:4326")
    groups = pd.DataFrame({"x": tile_x, "y": tile_y, "row": np.arange(len(grid))}).groupby(
        ["x", "y"], sort=False
    )
    tile_root = output_path.parent / "grid_tiles" / str(DETAIL_TILE_ZOOM)
    keys = []
    for (x, y), rows in groups:
        tile = full.iloc[rows.row.to_numpy()]
        features = list(tile.iterfeatures(drop_id=True, na="null"))
        directory = tile_root / str(x)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{y}.geojson").open("w", encoding="utf-8") as handle:
            json.dump(
                {"type": "FeatureCollection", "features": features},
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        keys.append(f"{x}/{y}")
    return keys


def _map_grid(
    grid: gpd.GeoDataFrame, score_fields: list[str], max_cells: int, full_detail_limit: int = 50_000
) -> tuple[gpd.GeoDataFrame, int]:
    """Kleine Gebiete vollständig, große Gebiete lückenlos zusammengefasst."""
    if len(grid) <= full_detail_limit:
        return grid.to_crs("EPSG:4326"), 100

    projected = grid.to_crs("EPSG:3035")
    # Große Gebiete erhalten eine gröbere Übersicht als Vielfaches von 100 m.
    # Die Zielanzahl steuert deren Auflösung; die Detaildaten bleiben separat.
    tile_size = 100 * max(2, int(np.ceil(np.sqrt(len(grid) / max_cells))))
    east = (np.floor((projected.center_x.to_numpy() - 50) / tile_size) * tile_size).astype(np.int64)
    north = (np.floor((projected.center_y.to_numpy() - 50) / tile_size) * tile_size).astype(
        np.int64
    )
    attributes = pd.DataFrame(
        {
            "tile_e": east,
            "tile_n": north,
            "status": projected.status.to_numpy(),
            **{field: projected[field].to_numpy() for field in score_fields},
        }
    )
    grouped = attributes.groupby(["tile_e", "tile_n"], sort=False)
    values = grouped[score_fields].mean()
    values["status"] = grouped.status.agg(
        lambda statuses: "complete" if statuses.eq("complete").all() else "incomplete"
    )
    values["represented_cells"] = grouped.size()
    geometries = projected.geometry.to_numpy()
    # Vereint werden nur die tatsächlich vorhandenen Zellpolygone, damit
    # am Gebietsrand und in Aussparungen keine zusätzlichen Flächen entstehen.
    values["geometry"] = [
        shapely.union_all(geometries[grouped.indices[key]]) for key in values.index
    ]
    values["grid_id"] = [f"Kartenfläche E{e} N{n}" for e, n in values.index]
    return gpd.GeoDataFrame(
        values.reset_index(drop=True), geometry="geometry", crs="EPSG:3035"
    ).to_crs("EPSG:4326"), tile_size


def write_stage_map(
    output_path, grid, stops, pois, category_names, max_cells=20_000, full_detail_limit=50_000
) -> dict:
    """Ein gemeinsamer Rasterlayer; die HTML-Auswahl wechselt das Scoreattribut."""
    census_labels = {
        "zensus_1_Durchschnittsalter": "Zensus: Durchschnittsalter (Jahre)",
        "zensus_2_AnteilUeber65": "Zensus: Anteil ab 65 (%)",
        "zensus_3_AnteilUnter18": "Zensus: Anteil unter 18 (%)",
        "zensus_4_Einwohner": "Zensus: Einwohner je 100-m-Zelle",
    }
    census_fields = {
        column: label for column, label in census_labels.items() if column in grid.columns
    }
    # Darstellungswerte werden auf einer Kopie aufbereitet, sodass
    # die eingelesenen Scores und Zensusattribute unverändert bleiben.
    grid_for_map = grid.copy()
    for column in census_fields:
        grid_for_map[column] = pd.to_numeric(
            grid_for_map[column].astype("string").str.replace(",", ".", regex=False),
            errors="coerce",
        ).astype(float)
    fields = {
        "overall_score": "Gesamtscore",
        **{f"score_{key}": f"Score: {label}" for key, label in category_names.items()},
        **census_fields,
    }
    scales = {field: 100 for field in fields}
    population_field = "zensus_4_Einwohner"
    if population_field in census_fields:
        population = grid_for_map[population_field].dropna()
        if not population.empty:
            scales[population_field] = max(10, int(np.ceil(population.quantile(0.95) / 10) * 10))
    displayed, tile_size = _map_grid(grid_for_map, list(fields), max_cells, full_detail_limit)
    detail_tiles = (
        _write_detail_tiles(Path(output_path), grid_for_map, list(fields))
        if tile_size > 100
        else []
    )
    center = displayed.geometry.union_all().centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=10, tiles=None)
    folium.TileLayer("OpenStreetMap", referrerPolicy="strict-origin-when-cross-origin").add_to(fmap)
    folium.map.CustomPane("rasterPane", z_index=400).add_to(fmap)
    folium.map.CustomPane("stopPane", z_index=650).add_to(fmap)
    folium.map.CustomPane("poiPane", z_index=660).add_to(fmap)
    tooltip_fields = ["grid_id", "status", *fields]
    tooltip_labels = [
        "Zelle" if tile_size == 100 else "Kartenfläche",
        "Datenstatus",
        *fields.values(),
    ]
    if tile_size > 100:
        tooltip_fields.append("represented_cells")
        tooltip_labels.append("100-m-Zellen")
    data = displayed[[*tooltip_fields, "geometry"]].copy()
    for field in fields:
        data[field] = data[field].astype(object).where(data[field].notna(), None)
    palette = linear.YlGnBu_09.scale(0, 100)
    colors = [palette(value)[:7] for value in range(101)]
    layer = folium.GeoJson(
        data.__geo_interface__,
        name="100-m-Zellen" if tile_size == 100 else f"Kartenflächen ({tile_size} m)",
        style_function=lambda feature: {
            "fillColor": "#999999"
            if feature["properties"]["overall_score"] is None
            else palette(feature["properties"]["overall_score"]),
            "fillOpacity": 0.8,
            "weight": 0.1,
            "color": "#ffffff",
        },
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_labels),
        control=False,
        pane="rasterPane",
    ).add_to(fmap)
    menu = MacroElement()
    menu._template = Template(
        (Path(__file__).parent / "templates" / "stage_map.js.j2").read_text(encoding="utf-8")
    )
    # Die Vorlage erhält Daten und Beschriftungen als JSON. Ein gemeinsamer
    # Rasterlayer wechselt danach im Browser das dargestellte Merkmal.
    menu.fields = json.dumps(fields, ensure_ascii=False)
    menu.layer = layer.get_name()
    menu.colors = json.dumps(colors)
    menu.scales = json.dumps(scales)
    menu.gradient = json.dumps(
        "linear-gradient(to right, "
        + ", ".join(palette(value)[:7] for value in (0, 25, 50, 75, 100))
        + ")"
    )
    menu.map = fmap.get_name()
    menu.tile_zoom = DETAIL_TILE_ZOOM
    menu.switch_zoom = DETAIL_SWITCH_ZOOM
    menu.tile_keys = json.dumps(detail_tiles)
    menu.note = json.dumps(
        f"Übersicht: {len(displayed):,} Flächen mit {tile_size} m. Ab Zoomstufe {DETAIL_SWITCH_ZOOM}: vollständige 100-m-Zellen im sichtbaren Ausschnitt."
        if tile_size > 100
        else f"Alle {len(grid):,} Rasterzellen."
    )
    fmap.add_child(menu)
    if not stops.empty:
        folium.GeoJson(
            stops.to_crs("EPSG:4326")[["stop_name", "geometry"]].__geo_interface__,
            name="Haltestellen",
            show=False,
            marker=_point_marker(radius=3, pane="stopPane"),
            pane="stopPane",
            tooltip=folium.GeoJsonTooltip(fields=["stop_name"]),
        ).add_to(fmap)
    poi_colors = (
        "#d73027",
        "#fc8d59",
        "#7b3294",
        "#008837",
        "#e6ab02",
        "#a6761d",
        "#1b9e77",
        "#e7298a",
        "#7570b3",
        "#e41a1c",
    )
    for index, (category, label) in enumerate(category_names.items()):
        subset = pois.loc[pois.category_id.eq(category)]
        if not subset.empty:
            folium.GeoJson(
                subset.to_crs("EPSG:4326")[["name", "category_name", "geometry"]].__geo_interface__,
                name=f"POI: {label}",
                show=False,
                marker=_point_marker(
                    radius=4,
                    pane="poiPane",
                    color=poi_colors[index % len(poi_colors)],
                    fill=True,
                    fill_opacity=0.9,
                ),
                pane="poiPane",
                tooltip=folium.GeoJsonTooltip(fields=["name", "category_name"]),
            ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.save(str(output_path))
    return {
        "total_cells": len(grid),
        "displayed_cells": len(displayed),
        "map_resolution_meters": tile_size,
        "display_aggregation": tile_size > 100,
        "poi_category_layers": int(
            sum(bool(pois.category_id.eq(category).any()) for category in category_names)
        ),
        "census_layers": list(census_fields),
        "population_color_saturation": scales.get(population_field),
        "full_resolution_export": True,
        "detail_tile_count": len(detail_tiles),
        "detail_switch_zoom": DETAIL_SWITCH_ZOOM if detail_tiles else None,
    }
