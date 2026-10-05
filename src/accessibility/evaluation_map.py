"""Separate Auswertungskarte; bestehender Export und seine Cache-Signaturen bleiben unverändert."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from uuid import uuid4

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from branca.colormap import linear
from branca.element import Element, MacroElement, Template

from .outputs import DETAIL_TILE_ZOOM


def evaluation_overview(grid, fields, max_cells=20_000, full_detail_limit=50_000):
    """Fasst bekannte 100-m-Werte als ungewichtete Mittelwerte in festen 1-km-Zellen zusammen.

    Die Zusammenfassung dient der Anzeige; fehlende Werte gehen nicht als Null ein.
    """
    tile_size = 1000
    projected = grid.to_crs("EPSG:3035")
    # Die Südwestecke der 100-m-Zelle bestimmt ihre feste 1-km-Gruppe.
    # So bleibt die Übersicht am selben globalen Zensusgitter ausgerichtet.
    east = (np.floor((projected.center_x.to_numpy() - 50) / tile_size) * tile_size).astype(np.int64)
    north = (np.floor((projected.center_y.to_numpy() - 50) / tile_size) * tile_size).astype(
        np.int64
    )
    values = pd.DataFrame({"e": east, "n": north, **{f: grid[f].to_numpy() for f in fields}})
    grouped = values.groupby(["e", "n"], sort=False)
    # Mittelwerte dienen nur der Anzeige. Auch Klassen- und Prioritätscodes
    # werden gemittelt; es werden keine neuen regionalen Grenzen berechnet.
    displayed = grouped[fields].mean()
    displayed["represented_cells"] = grouped.size()
    displayed["status"] = "Zusammenfassung bekannter 100-m-Werte"
    geometries = projected.geometry.to_numpy()
    displayed["geometry"] = [
        shapely.union_all(geometries[grouped.indices[key]]) for key in displayed.index
    ]
    displayed["grid_id"] = [f"1-km-Kachel E{e} N{n}" for e, n in displayed.index]
    # Zusätzlich zu Mittelwerten bleiben die Anzahlen ursprünglicher
    # Prioritäten und unbekannter Werte für die Zellinformationen erhalten.
    for field in fields:
        if field.startswith("priority_"):
            group = field.removeprefix("priority_")
            for code in range(4):
                values["count"] = values[field].eq(code).astype(int)
                displayed[f"overview_{group}_{code}"] = (
                    values.groupby(["e", "n"], sort=False)["count"].sum().to_numpy()
                )
            values["count"] = values[field].isna().astype(int)
            displayed[f"overview_{group}_unknown"] = (
                values.groupby(["e", "n"], sort=False)["count"].sum().to_numpy()
            )
    for age in ("under18", "65plus"):
        flag = f"population_{age}_uncertain"
        if flag in grid:
            values["count"] = grid[flag].fillna(False).astype(bool).astype(int).to_numpy()
            counts = values.groupby(["e", "n"], sort=False)["count"].sum().to_numpy()
            displayed[flag] = counts > 0
            displayed[f"overview_{age}_uncertain_count"] = counts
    return gpd.GeoDataFrame(
        displayed.reset_index(drop=True), geometry="geometry", crs="EPSG:3035"
    ).to_crs("EPSG:4326"), tile_size


def _write_canvas_data(path, grid, fields):
    """Schreibt Zellattribute und Geometrien als kompakte Arrays für die Canvas-Karte."""
    projected = grid.to_crs("EPSG:3857")
    # Fehlende Werte werden als JSON-null übertragen. Die gemeinsame
    # Feldliste vermeidet wiederholte Attributnamen in jeder einzelnen Zelle.
    values = grid[fields].astype(object).where(grid[fields].notna(), None).to_numpy().tolist()
    # Web-Mercator-Meter werden in Weltpixel der Zoomstufe 0 umgerechnet.
    # Die y-Achse zeigt im Browser nach unten; höhere Zoomstufen skalieren
    # diese Koordinaten anschließend im Canvas-Renderer.
    circumference = 40075016.68557849
    cells = []
    for geometry, attributes in zip(projected.geometry, values, strict=True):
        polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        rings = []
        for polygon in polygons:
            for ring in [polygon.exterior, *polygon.interiors]:
                rings.append(
                    [
                        [
                            round(256 * (0.5 + x / circumference), 9),
                            round(256 * (0.5 - y / circumference), 9),
                        ]
                        for x, y in ring.coords
                    ]
                )
        cells.append([rings, attributes])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"fields": fields, "cells": cells},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _write_canvas_tiles(root, grid, fields):
    """Teilt die 100-m-Zellen nach Webkartenkacheln auf und schreibt deren JSON-Dateien."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(grid.center_x.to_numpy(), grid.center_y.to_numpy())
    count = 1 << DETAIL_TILE_ZOOM
    x = np.floor((lon + 180) / 360 * count).astype(int)
    y = np.floor((1 - np.arcsinh(np.tan(np.deg2rad(lat))) / np.pi) / 2 * count).astype(int)
    # Der Zellmittelpunkt entscheidet über die Datei. Die gespeicherte
    # Geometrie bleibt vollständig, auch wenn sie eine Kachelgrenze schneidet.
    groups = pd.DataFrame({"x": x, "y": y, "row": np.arange(len(grid))}).groupby(
        ["x", "y"], sort=False
    )
    keys = []
    for (east, north), rows in groups:
        key = f"{east}/{north}"
        _write_canvas_data(root / (key + ".json"), grid.iloc[rows.row.to_numpy()], fields)
        keys.append(key)
    return keys


def write_evaluation_map(
    output_path,
    grid,
    stops,
    pois,
    category_names,
    report,
    max_cells=20_000,
    full_detail_limit=50_000,
) -> dict:
    """Ein gemeinsamer Rasterlayer; die HTML-Auswahl wechselt das Scoreattribut."""
    from .evaluation import GROUPS, census_number

    grid_for_map = grid.copy()
    suffix_labels = {
        "Durchschnittsalter": "Zensus: Durchschnittsalter (Jahre)",
        "AnteilUeber65": "Zensus: Anteil ab 65 (%)",
        "AnteilUnter18": "Zensus: Anteil unter 18 (%)",
    }
    census_fields = {
        c: label
        for c in grid
        for suffix, label in suffix_labels.items()
        if c.startswith("zensus_") and c.endswith("_" + suffix)
    }
    for column in census_fields:
        grid_for_map[column] = census_number(grid_for_map[column], maximum=100)
    fields = {
        **{f"priority_{g}": "Untersuchungspriorität: " + info[0] for g, info in GROUPS.items()},
        "overall_score": "Gesamtscore",
        "score_education": "Bildungsscore",
        "score_health": "Gesundheitsscore",
        **{
            f"score_{key}": f"Score: {label}"
            for key, label in category_names.items()
            if f"score_{key}" in grid
        },
        "population_total": "Einwohner",
        "population_under18": "Unter 18 (rechnerisch ermittelt)",
        "population_65plus": "Ab 65 (rechnerisch ermittelt)",
        **{f"score_class_{g}": "Erreichbarkeitsklasse: " + info[0] for g, info in GROUPS.items()},
        **{f"population_class_{g}": "Bevölkerungsklasse: " + info[0] for g, info in GROUPS.items()},
        **census_fields,
    }
    extra = [f"population_{age}_uncertain" for age in ("under18", "65plus")]
    extra += [f"evaluation_{g}_status" for g in GROUPS]
    displayed, tile_size = evaluation_overview(
        grid_for_map, list(fields), max_cells, full_detail_limit
    )
    # Ein eigener Datenordner pro Karte verhindert, dass bereits geöffnete
    # Karten beim Neuschreiben plötzlich andere Kacheldateien nachladen.
    assets = Path(output_path).parent / ("map_assets-" + uuid4().hex[:12])
    assets.mkdir()
    detail_tiles = (
        _write_canvas_tiles(assets / "tiles", grid_for_map, ["grid_id", "status", *fields, *extra])
        if tile_size > 100
        else []
    )
    shutil.copyfile(Path(__file__).with_name("evaluation_canvas.js"), assets / "raster.js")
    if tile_size == 100:
        displayed[extra] = grid_for_map[extra].to_numpy()
    scales = {field: 100 for field in fields}
    detail_scales = dict(scales)
    # Das 95-%-Quantil steuert ausschließlich die Farbsättigung der
    # Personenwerte; die gespeicherten Werte und Klassen bleiben erhalten.
    for field in ("population_total", "population_under18", "population_65plus"):
        for frame, target in ((grid_for_map, detail_scales), (displayed, scales)):
            positive = frame.loc[frame[field].gt(0), field]
            target[field] = max(1, float(positive.quantile(0.95))) if not positive.empty else 1
    center = displayed.geometry.total_bounds
    center_y, center_x = (center[1] + center[3]) / 2, (center[0] + center[2]) / 2
    fmap = folium.Map(location=[center_y, center_x], zoom_start=10, tiles=None)
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
    data = displayed[
        [
            *dict.fromkeys(
                [
                    *tooltip_fields,
                    *[c for c in displayed if c.startswith("overview_") or c in extra],
                ]
            ),
            "geometry",
        ]
    ].copy()
    for field in fields:
        data[field] = data[field].astype(object).where(data[field].notna(), None)
    palette = linear.YlGnBu_09.scale(0, 100)
    colors = [palette(value)[:7] for value in range(101)]
    # Kollineare Punkte nur für die Darstellung entfernen; Exportgeometrien bleiben erhalten.
    data = data.to_crs("EPSG:3035")
    data.geometry = data.geometry.simplify(0.01, preserve_topology=True)
    _write_canvas_data(assets / "overview.json", data, [c for c in data if c != "geometry"])
    fmap.get_root().header.add_child(Element(f'<script src="{assets.name}/raster.js"></script>'))
    fmap.get_root().header.add_child(
        Element(
            (Path(__file__).parent / "templates" / "map_style.html").read_text(encoding="utf-8")
        )
    )
    menu = MacroElement()
    menu._template = Template(
        (Path(__file__).parent / "templates" / "evaluation_map.js.j2").read_text(encoding="utf-8")
    )
    # Die Vorlage verbindet die gespeicherten Daten mit der Browseransicht.
    # json.dumps überträgt Beschriftungen und Werte als JavaScript-Literale.
    menu.fields = json.dumps(fields, ensure_ascii=False)
    menu.overview_url = json.dumps(f"{assets.name}/overview.json")
    menu.tile_root = json.dumps(f"{assets.name}/tiles")
    menu.colors = json.dumps(colors)
    menu.scales = json.dumps(scales)
    menu.detail_scales = json.dumps(detail_scales)
    menu.thresholds = json.dumps(report["thresholds"])
    menu.reference_area = json.dumps(report["reference_area"])
    menu.gradient = json.dumps(
        "linear-gradient(to right, "
        + ", ".join(palette(value)[:7] for value in (0, 25, 50, 75, 100))
        + ")"
    )
    menu.map = fmap.get_name()
    menu.tile_zoom = DETAIL_TILE_ZOOM
    menu.tile_keys = json.dumps(detail_tiles)
    menu.note = json.dumps(
        f"Auswertungsgebiet: {report['analysis_area']}. Rasterauflösung unabhängig vom Zoom wählbar. 100 m lädt bei großen Kartenausschnitten mehr Daten. Mittelwerte können lokale Unterschiede verdecken."
    )
    fmap.add_child(menu)
    # Punktdaten erst beim Einschalten der jeweiligen Kartenebene laden.
    point_sources = []
    if not stops.empty:
        point_sources.append(("Haltestellen", stops, ["stop_name"], "stopPane", "#3388ff", 3))
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
    for i, (category, label) in enumerate(category_names.items()):
        subset = pois.loc[pois.category_id.eq(category)]
        if not subset.empty:
            point_sources.append(
                (
                    "POI: " + label,
                    subset,
                    ["name", "category_name"],
                    "poiPane",
                    poi_colors[i % len(poi_colors)],
                    4,
                )
            )
    for i, (label, points, attributes, pane, color, radius) in enumerate(point_sources):
        target = assets / f"points-{i}.json"
        target.write_text(
            points.to_crs("EPSG:4326")[[*attributes, "geometry"]].to_json(drop_id=True),
            encoding="utf-8",
        )
        group = folium.FeatureGroup(name=label, show=False).add_to(fmap)
        loader = MacroElement()
        loader._template = Template(
            (Path(__file__).parent / "templates" / "point_layer.js.j2").read_text(encoding="utf-8")
        )
        loader.group, loader.map = group.get_name(), fmap.get_name()
        loader.url, loader.pane = json.dumps(f"{assets.name}/{target.name}"), json.dumps(pane)
        loader.radius, loader.color, loader.attributes = (
            radius,
            json.dumps(color),
            json.dumps(attributes),
        )
        fmap.add_child(loader)
    folium.LayerControl().add_to(fmap)
    fmap.save(str(output_path))
    return {
        "renderer": "canvas",
        "assets_directory": assets.name,
        "initial_html_bytes": Path(output_path).stat().st_size,
        "total_cells": len(grid),
        "displayed_cells": len(displayed),
        "map_resolution_meters": tile_size,
        "display_aggregation": tile_size > 100,
        "poi_category_layers": int(
            sum(bool(pois.category_id.eq(category).any()) for category in category_names)
        ),
        "census_layers": list(census_fields),
        "population_color_saturation": detail_scales.get("population_total"),
        "reference_area": report["reference_area"],
        "overview_priority_rule": "arithmetic mean of known 100m priority codes, including zero",
        "overview_value_rule": "arithmetic mean of known 100m values per field; missing values excluded",
        "resolution_selection": "manual",
        "available_resolutions_meters": [1000, 100],
        "full_resolution_export": True,
        "detail_tile_count": len(detail_tiles),
        "detail_switch_zoom": None,
    }


def rebuild_evaluation_map(config):
    """Erstellt die Karte aus fertigen Kartenklassen mit eigenen, unveränderlichen Datendateien."""
    import fcntl
    import tempfile

    import geopandas as gpd

    from .evaluation import _markdown_report, _validate_export
    from .exceptions import DataValidationError
    from .stages import StageStore, identity, read_json, write_json

    store = StageStore(config)
    folder = store.output / "evaluation"
    if (
        not (folder / "evaluation_report.json").is_file()
        or not (folder / "evaluation.gpkg").is_file()
    ):
        raise DataValidationError(
            "evaluate-map benötigt eine fertige Auswertung. Es werden keine Vorgänger gestartet."
        )
    store.root.mkdir(parents=True, exist_ok=True)
    with (store.root / "process.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DataValidationError(
                "Für diesen Ergebnisordner läuft bereits ein Verarbeitungsschritt."
            ) from exc
        manifest = _validate_export(store.output)
        report = read_json(folder / "evaluation_report.json")
        if report["source_export_run_id"] != manifest["run_id"]:
            raise DataValidationError(
                "Auswertung gehört nicht zum aktuellen Export. Zuerst evaluate ausführen."
            )
        initial = {
            name: identity(folder / name)
            for name in ("evaluation_report.json", "evaluation.gpkg", "thresholds.json")
        }
        metadata = read_json(store.output / "run_metadata.json")
        logging.getLogger(__name__).info(
            "Erzeuge ausschließlich Kartendateien aus der vorhandenen Auswertung."
        )
        grid = gpd.read_file(folder / "evaluation.gpkg", layer="grid")
        stops = gpd.read_file(store.output / "accessibility.gpkg", layer="stops")
        pois = gpd.read_file(store.output / "accessibility.gpkg", layer="pois")
        names = {c["id"]: c["name"] for c in metadata["config"]["poi_categories"]}
        with tempfile.TemporaryDirectory(prefix=".map-", dir=folder) as temporary:
            destination = Path(temporary)
            result = write_evaluation_map(
                destination / "accessibility_map.html", grid, stops, pois, names, report
            )
            write_json(destination / "map_report.json", result)
            (destination / "AUSWERTUNG.md").write_text(_markdown_report(report), encoding="utf-8")
            if initial != {name: identity(folder / name) for name in initial}:
                raise DataValidationError(
                    "Auswertungsdaten wurden während des Kartenaufbaus geändert."
                )
            if _validate_export(store.output)["run_id"] != manifest["run_id"]:
                raise DataValidationError("Export wurde während des Kartenaufbaus geändert.")
            assets = result["assets_directory"]
            (destination / assets).replace(folder / assets)
            (destination / "map_report.json").replace(folder / "map_report.json")
            (destination / "AUSWERTUNG.md").replace(folder / "AUSWERTUNG.md")
            (destination / "accessibility_map.html").replace(folder / "accessibility_map.html")
        return folder / "accessibility_map.html"
