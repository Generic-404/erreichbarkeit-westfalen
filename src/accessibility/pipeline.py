"""Sechs explizite Prozesse; ein Aufruf führt genau einen Schritt aus."""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .census import attach_census_attributes, cell_coordinates, generate_census_grid
from .config import AppConfig, validate_paths
from .exceptions import ConfigurationError, DataValidationError
from .osm import extract_pois
from .outputs import write_stage_map
from .progress import timed
from .routing import _best_by_category, route_stop_to_pois
from .scoring import score_cells
from .stages import STEPS, StageStore, read_json, write_json
from .stops import group_stops, read_gtfs_stops, select_stops_near_boundary
from .surface import build_sources, transfer_times
from .timepoints import derive_timepoints, validate_gtfs_timepoints

LOGGER = logging.getLogger(__name__)
# Kennungen bleiben beim CSV-Einlesen Text, auch wenn sie nur Ziffern
# enthalten. So bleibt die Zuordnung zwischen den Phasentabellen erhalten.
ID_TYPES = {
    name: str
    for name in (
        "grid_id",
        "category_id",
        "stop_group_id",
        "poi_id",
        "from_id",
        "to_id",
        "source_grid_id",
    )
}


def _csv(path: Path, **kwargs):
    """Liest eine Ergebnistabelle und bewahrt Identifikationsspalten als Zeichenketten."""
    return pd.read_csv(path, dtype=ID_TYPES, **kwargs)


def _read_boundary(config: AppConfig) -> tuple[gpd.GeoDataFrame, object, object]:
    """Lädt und prüft die Gebietsgeometrie und erzeugt den konfigurierten Außenpuffer."""
    boundary = gpd.read_file(config.paths.boundary)
    if boundary.empty or boundary.crs is None:
        raise DataValidationError("Die Gebietsgrenze ist leer oder besitzt kein CRS.")
    # Der Puffer wird im metrischen Analysekoordinatensystem gebildet.
    # Er erweitert das Suchgebiet für Angebote außerhalb der Gebietsgrenze.
    boundary = boundary.to_crs(config.area.analysis_crs)
    shape = boundary.geometry.union_all()
    if shape.is_empty or not shape.is_valid or shape.geom_type not in ("Polygon", "MultiPolygon"):
        raise DataValidationError("Die Gebietsgrenze muss ein gültiges Polygon sein.")
    return boundary, shape, shape.buffer(config.area.buffer_km * 1000)


def _validate_method(config: AppConfig) -> None:
    """Prüft die für diese Berechnung vorausgesetzten Modellparameter."""
    if (
        config.surface.grid_size_meters != 100
        or not config.surface.enabled
        or not config.surface.include_direct_walk_to_poi
    ):
        raise ConfigurationError(
            "Die Methode benötigt surface.enabled=true, grid_size_meters=100 und include_direct_walk_to_poi=true."
        )
    if config.surface.airline_detour_factor < 1:
        raise ConfigurationError("Der Umwegfaktor muss mindestens 1 sein.")
    if len(config.analysis.times) != 1 or derive_timepoints(config.analysis)[0].date.weekday() > 4:
        raise ConfigurationError(
            "Genau einen Analysezeitpunkt an einem Montag bis Freitag konfigurieren."
        )
    if config.routing.maximum_travel_time_minutes < 61:
        raise ConfigurationError(
            "Der R5-Suchhorizont muss mindestens 61 Minuten betragen, damit 60 Minuten trotz interner strikter Grenze enthalten sind."
        )
    if config.routing.compute_walk_only_matrix:
        raise ConfigurationError(
            "compute_walk_only_matrix muss false sein; Gehquellen entstehen im sources-Schritt."
        )
    if not config.paths.census_csvs:
        raise ConfigurationError(
            "Mindestens eine bereitgestellte 100-m-Zensus-CSV unter paths.census_csvs angeben."
        )
    if not math.isclose(sum(c.weight for c in config.poi_categories), 100.0, abs_tol=1e-8):
        raise ConfigurationError("Die Kategoriegewichte müssen insgesamt 100 Prozent ergeben.")


def _osm_extent_report(path: Path, buffered: object, crs: str) -> dict:
    """Vergleicht das Suchgebiet mit dem im PBF-Header angegebenen Begrenzungsrechteck."""
    import osmium
    from shapely.geometry import box

    with osmium.io.Reader(str(path)) as reader:
        bounds = reader.header().box()
        if not bounds.valid():
            return {"header_extent": "unknown", "coverage": "Nicht aus dem PBF-Header bestimmbar."}
        extent = (
            gpd.GeoSeries(
                [
                    box(
                        bounds.bottom_left.lon,
                        bounds.bottom_left.lat,
                        bounds.top_right.lon,
                        bounds.top_right.lat,
                    )
                ],
                crs="EPSG:4326",
            )
            .to_crs(crs)
            .iloc[0]
        )
    return {
        "buffer_inside_header_bbox": bool(extent.covers(buffered)),
        "coverage": "Verglichen werden das Begrenzungsrechteck aus dem PBF-Header und das gepufferte Untersuchungsgebiet.",
    }


def prepare(
    config: AppConfig,
    destination: Path,
    *,
    boundary_file: Path,
    osm_pbf: Path,
    gtfs_zip: Path,
    census_csvs: tuple[Path, ...],
) -> None:
    """Schreibt Raster, POIs und Haltestellen aus dem fertigen regionalen Datenbestand."""
    # Die Pfade gelten für diesen Phasenaufruf; replace erzeugt dafür eine
    # neue Konfiguration und lässt das übergebene Objekt unverändert.
    config = replace(
        config,
        paths=replace(
            config.paths,
            boundary=boundary_file,
            osm_pbf=osm_pbf,
            gtfs=gtfs_zip,
            census_csvs=census_csvs,
        ),
    )
    if config.area.crop_osm:
        raise ConfigurationError(
            "Die Westfalen-Berechnung benötigt einen fertigen OSM-Ausschnitt und crop_osm=false."
        )
    timings = {}
    boundary, _, buffered = _read_boundary(config)
    extent_report = _osm_extent_report(config.paths.osm_pbf, buffered, config.area.analysis_crs)
    osm_path = config.paths.osm_pbf.resolve()
    crop_backend = "none"
    # POIs und Haltestellen werden im gepufferten Gebiet berücksichtigt,
    # damit auch Angebote jenseits der Grenze erreichbar sein können.
    with timed("prepare 2/5: POIs extrahieren", timings):
        pois = extract_pois(osm_path, config.poi_categories, buffered, config.area.analysis_crs)
        LOGGER.info("POI-Extraktion: %s Einrichtungen.", f"{len(pois):,}")
    with timed("prepare 3/5: GTFS-Haltestellen aufbereiten", timings):
        all_stops = read_gtfs_stops(config.paths.gtfs)
        # Die Vorauswahl bewahrt vollständige Parent-Gruppen, bevor deren
        # gemeinsame Haltestellenkoordinaten bestimmt werden.
        nearby = select_stops_near_boundary(
            all_stops,
            config.area.analysis_crs,
            buffered,
            config.stops.fallback_grouping_distance_meters,
        )
        if nearby.empty:
            stops = gpd.GeoDataFrame(
                columns=["stop_group_id", "stop_name", "source_stop_ids", "geometry"],
                geometry="geometry",
                crs=config.area.analysis_crs,
            )
            stop_report = {"original_gtfs_stops": len(all_stops), "stop_groups": 0}
        else:
            stops, report = group_stops(
                nearby,
                config.area.analysis_crs,
                buffered,
                config.stops.fallback_grouping_distance_meters,
                len(all_stops),
            )
            stop_report = report.as_dict()
    # Das Ergebnisraster gehört zur ungepufferten Gebietsgrenze.
    # Zensusmerkmale werden über die Zellkennung angefügt.
    with timed("prepare 4/5: Zensusraster und Attribute", timings):
        with timed("Rastergeometrie erzeugen", timings):
            grid = generate_census_grid(boundary)
        with timed("Zensusattribute einlesen", timings):
            grid, census_report = attach_census_attributes(grid, config.paths.census_csvs)
        LOGGER.info("Zensusraster: %s Zellen.", f"{len(grid):,}")
    # Jede Datenart erhält eine eigene Datei, die spätere Phasen direkt
    # einlesen können, ohne die Vorbereitung erneut auszuführen.
    with timed("prepare 5/5: Ergebnisse speichern", timings):
        for name, frame in (
            ("stops", stops),
            ("pois", pois),
            ("grid", grid),
            ("boundary", boundary),
        ):
            frame.to_file(destination / f"{name}.gpkg", layer=name, driver="GPKG")
    counts = {
        category.id: int(pois.category_id.eq(category.id).sum())
        for category in config.poi_categories
    }
    write_json(
        destination / "report.json",
        {
            "stops": len(stops),
            "pois": len(pois),
            "grid_cells": len(grid),
            "poi_counts": counts,
            "stop_grouping": stop_report,
            "census": census_report,
            "osm_coverage": extent_report,
            "crop_backend": crop_backend,
            "timings_seconds": timings,
            "routing_osm": str(osm_path),
        },
    )


def route(
    config: AppConfig,
    destination: Path,
    *,
    stops_file: Path,
    pois_file: Path,
    grid_file: Path,
    osm_pbf: Path,
    gtfs_zip: Path,
    checkpoint_directory: Path | None,
) -> None:
    """Berechnet Reisezeiten von Haltestellen zu den ausgewählten POI-Kandidaten."""
    point = derive_timepoints(config.analysis)[0]
    validate_gtfs_timepoints(gtfs_zip, [point])
    stops, pois = (gpd.read_file(path) for path in (stops_file, pois_file))
    grid = gpd.read_file(grid_file)
    # Aus Gehzeit, Geschwindigkeit und Umwegfaktor folgt der maximale
    # Luftlinienabstand zwischen Quellzelle und Ergebniszelle.
    radius = (
        config.routing.maximum_grid_access_walking_time_minutes
        * config.routing.walking_speed_kmh
        * 1000
        / 60
        / config.surface.airline_detour_factor
    )
    routed_stops = _routing_stops(stops, grid, radius)
    LOGGER.info(
        "Routing-Ursprünge: %d von %d Haltestellengruppen können eine Ergebniszelle erreichen (%.0f m).",
        len(routed_stops),
        len(stops),
        radius,
    )
    # R5 berechnet zunächst einzelne Haltestellen-POI-Paare. Die Auswahl
    # des besten Angebots je Kategorie erfolgt erst in sources.
    pairs = route_stop_to_pois(
        routed_stops,
        pois,
        osm_pbf,
        gtfs_zip,
        point,
        config.routing,
        [c.id for c in config.poi_categories],
        config.analysis.departure_time_window_minutes,
        checkpoint_directory=checkpoint_directory,
        plan_directory=destination,
        timezone=config.analysis.timezone,
    )
    pairs.to_csv(destination / "travel_times.csv", index=False)
    write_json(
        destination / "report.json",
        {
            "point": asdict(point),
            "window_minutes": config.analysis.departure_time_window_minutes,
            "routed_stop_groups": len(routed_stops),
            "prepared_stop_groups": len(stops),
            "grid_access_radius_meters": radius,
            "aggregation": "minimum over R5 departure-minute iterations before time-independent egress",
            "transit_required": True,
            "selected_pairs": len(pairs),
            "finite_pairs": int(pairs.travel_time_minutes.notna().sum()),
            "walking_limit_semantics": "R5 limits access, egress and transfers separately, not their total sum.",
        },
    )


def _routing_stops(
    stops: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, radius: float
) -> gpd.GeoDataFrame:
    """Nur Haltestellenzellen mit mindestens einer Ergebniszelle im Gehkreis routen."""
    from scipy.spatial import cKDTree

    if stops.empty:
        return stops.copy()
    grid_centers = grid[["center_x", "center_y"]].to_numpy(float)
    # Maßgeblich ist der Mittelpunkt der Haltestellenzelle, weil transfer
    # später ebenfalls Entfernungen zwischen Zellmittelpunkten verwendet.
    stop_centers = cell_coordinates(stops)[["center_x", "center_y"]].to_numpy(float)
    distances = cKDTree(grid_centers).query(stop_centers, k=1)[0]
    return stops.iloc[np.flatnonzero(distances <= radius + 1e-8)].copy()


def sources(
    config: AppConfig,
    destination: Path,
    *,
    stops_file: Path,
    pois_file: Path,
    travel_times_file: Path,
    prepare_report_file: Path,
) -> None:
    """Wählt die ursprünglichen ÖPNV- und Gehquellen je Zelle und Kategorie."""
    stops, pois = (gpd.read_file(path) for path in (stops_file, pois_file))
    pairs = _csv(travel_times_file)
    # Pro Haltestelle und Kategorie bleibt die kleinste geroutete
    # Reisezeit erhalten; anschließend werden daraus Rasterquellen.
    best = _best_by_category(
        pairs,
        pois,
        stops,
        derive_timepoints(config.analysis)[0],
        [c.id for c in config.poi_categories],
    )
    best.to_csv(destination / "stop_category_times.csv", index=False)
    counts = read_json(prepare_report_file)["poi_counts"]
    # Im automatischen Modus bedeuten fehlende POI-Treffer eine ungeklärte
    # Datenlage. Eine explizite Statuseinstellung hat Vorrang.
    status = {
        c.id: ("available" if counts[c.id] else "unknown")
        if c.data_status == "auto"
        else c.data_status
        for c in config.poi_categories
    }
    # Neben ÖPNV-Angeboten liefern POIs direkte Gehquellen. Für Kategorien
    # mit unbekannter Datenlage werden keine Quellen weitergegeben.
    values = build_sources(stops, pois, best)
    values = values.loc[values.category_id.map(status).eq("available")]
    values.to_csv(destination / "sources.csv", index=False)
    write_json(
        destination / "report.json",
        {
            "category_status": status,
            "source_count": len(values),
            "note": "auto setzt den Datenstatus bei vorhandenen OSM-POIs auf available und bei null Treffern auf unknown.",
        },
    )


def transfer(
    config: AppConfig,
    destination: Path,
    *,
    grid_file: Path,
    sources_file: Path,
    sources_report_file: Path,
) -> None:
    """Überträgt Reisezeiten einmalig von ursprünglichen Quellen auf das Raster."""
    grid = gpd.read_file(grid_file, columns=["grid_id", "center_x", "center_y"])
    values = _csv(sources_file)
    status = read_json(sources_report_file)["category_status"]
    rows = 0
    # Die Quellen werden einmal eingelesen. Der Generator verarbeitet
    # das große Zielraster in Blöcken und erzeugt daraus keine neuen Quellen.
    for block in transfer_times(
        grid,
        values,
        status,
        config.routing.maximum_grid_access_walking_time_minutes,
        config.routing.walking_speed_kmh,
        config.surface.airline_detour_factor,
    ):
        # Nur der erste Block schreibt die Kopfzeile; weitere Blöcke
        # werden angehängt, ohne die Gesamttabelle im Speicher zu halten.
        block.to_csv(
            destination / "grid_times.csv",
            mode="w" if rows == 0 else "a",
            header=rows == 0,
            index=False,
        )
        rows += len(block)
        LOGGER.info("Reisezeiten übertragen: %d Zellen.", rows // len(status))
    write_json(
        destination / "report.json",
        {
            "rows": rows,
            "grid_cells": len(grid),
            "category_count": len(status),
            "walking_radius_m": config.routing.maximum_grid_access_walking_time_minutes
            * config.routing.walking_speed_kmh
            * 1000
            / 60
            / config.surface.airline_detour_factor,
        },
    )


def score(config: AppConfig, destination: Path, *, grid_times_file: Path) -> None:
    """Berechnet Kategorie- und Gesamtscores aus gespeicherten Rasterreisezeiten."""
    weights = {c.id: c.weight for c in config.poi_categories}
    rows = complete = cells = 0
    # transfer schreibt die Kategorien jeder Zelle zusammenhängend.
    # Die Blockgröße umfasst deshalb ein Vielfaches der Kategorienzahl.
    for block in _csv(grid_times_file, chunksize=10_000 * len(weights)):
        detail, summary = score_cells(block, weights)
        for name, frame in (("category_scores", detail), ("grid_scores", summary)):
            frame.to_csv(
                destination / f"{name}.csv",
                mode="w" if rows == 0 else "a",
                header=rows == 0,
                index=False,
            )
        rows += len(detail)
        cells += len(summary)
        complete += int(summary.status.eq("complete").sum())
    write_json(
        destination / "report.json",
        {
            "rows": rows,
            "cells": cells,
            "complete_cells": complete,
            "incomplete_cells": cells - complete,
            "weights_percent": weights,
        },
    )


def export(
    config: AppConfig,
    destination: Path,
    *,
    prepare_directory: Path,
    route_directory: Path,
    sources_directory: Path,
    transfer_directory: Path,
    score_directory: Path,
) -> None:
    """Exportiert die fünf angegebenen Phasenergebnisse einschließlich Basiskarte."""
    directories = {
        "prepare": prepare_directory,
        "route": route_directory,
        "sources": sources_directory,
        "transfer": transfer_directory,
        "score": score_directory,
    }
    inputs = prepare_directory
    # Der linke Join erhält alle vorbereiteten Zellen, auch wenn für
    # einzelne Zellen kein vollständiger Gesamtscore vorliegt.
    grid = gpd.read_file(inputs / "grid.gpkg").merge(
        _csv(score_directory / "grid_scores.csv"), on="grid_id", how="left", validate="one_to_one"
    )
    stops, pois = (gpd.read_file(inputs / f"{name}.gpkg") for name in ("stops", "pois"))
    for name, frame in (("grid", grid), ("stops", stops), ("pois", pois)):
        frame.to_file(destination / "accessibility.gpkg", layer=name, driver="GPKG")
    for step, files in {
        "route": ("travel_times.csv", "blocks.csv", "candidates.csv", "matrix_sizes.csv"),
        "sources": ("sources.csv", "stop_category_times.csv"),
        "score": ("category_scores.csv", "grid_scores.csv"),
    }.items():
        for name in files:
            if (directories[step] / name).exists():
                shutil.copy2(directories[step] / name, destination / name)
    # Berichte beschreiben die Berechnung; Manifeste weisen die konkret
    # verwendeten Vorgängerläufe nach. Beides wird im Export mitgeführt.
    reports = {step: read_json(directories[step] / "report.json") for step in STEPS[:-1]}
    provenance = {step: read_json(directories[step] / "manifest.json") for step in STEPS[:-1]}
    map_report = write_stage_map(
        destination / "accessibility_map.html",
        grid,
        stops,
        pois,
        {c.id: c.name for c in config.poi_categories},
    )
    limitations = [
        "Je Ursprung-Ziel-Paar wird die geringste Reisezeit im Abfahrtsintervall verwendet. R5 verarbeitet Abfahrtsminuten und liefert Reisezeiten in ganzen Minuten.",
        "Die Kandidatenauswahl umfasst je Haltestelle und Kategorie die zehn luftliniennächsten POIs.",
        "Direkte POIs erhalten in ihrer 100-m-Zelle null Minuten. Die Gehzeitübertragung verwendet Zellmittelpunkte, Gehgeschwindigkeit und Umwegfaktor; Fußwegbarrieren werden dabei nicht eingelesen.",
        "Der Rasterzugang wird zur Reisezeit der Quellzelle addiert. Die R5-Abfahrtszeit bleibt dabei konstant.",
        "Die 15-Minuten-Grenze für den Rasterzugang gilt getrennt von den R5-Gehgrenzen für Zugang, Abgang und Umstiege.",
        "Der Datenstatus folgt den POI-Treffern und der Konfiguration. Der räumliche Headervergleich verwendet das Begrenzungsrechteck des PBF.",
        "Bildungs- und Gesundheitsscores verbinden jeweils zwei Kategorien mit ihren relativen Konfigurationsgewichten.",
        "Der Reisezeitscore beträgt unter 10 Minuten 100, von 10 bis einschließlich 60 Minuten 146.11 * exp(-0.039 * t) und darüber 0.",
    ]
    metadata = {
        "config": asdict(config),
        "reports": reports,
        "provenance": provenance,
        "map": map_report,
        "limitations": limitations,
    }
    write_json(destination / "run_metadata.json", metadata)
    lines = [
        "# Prüfbericht",
        "",
        f"Rasterzellen: {len(grid)}",
        f"Haltestellen: {len(stops)}",
        f"POIs: {len(pois)}",
        "",
        "## Datenlage",
        "",
        *[f"- {k}: {v}" for k, v in reports["sources"]["category_status"].items()],
        "",
        "## Verarbeitung",
        "",
        *[f"- {s}: {m['duration_seconds']} s; Lauf {m['run_id']}" for s, m in provenance.items()],
        "",
        "## Berechnungsverfahren",
        "",
        *[f"- {value}" for value in limitations],
        "",
        "Details zu Datenständen, Matrixgrößen und Parametern: run_metadata.json und matrix_sizes.csv.",
    ]
    (destination / "PRUEFBERICHT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step(config: AppConfig, step: str) -> Path:
    """Führt genau eine Phase mit geprüften Vorgängern und atomarer Ausgabe aus."""
    if step not in STEPS:
        raise ConfigurationError(f"Unbekannter Verarbeitungsschritt: {step}")
    validate_paths(config.paths)
    _validate_method(config)
    store = StageStore(config)
    store.root.mkdir(parents=True, exist_ok=True)
    file_log = logging.FileHandler(store.root / f"{step}.log", encoding="utf-8")
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_log)
    try:
        # Die Phase schreibt zunächst in einen temporären Ordner.
        # StageStore übernimmt ihn erst nach vollständigem Erfolg.
        with store.transaction(step) as destination:
            run_with_directories(config, step, store, destination)
    finally:
        logging.getLogger().removeHandler(file_log)
        file_log.close()
    return store.directory(step)


def run_with_directories(
    config: AppConfig, step: str, store: StageStore, destination: Path
) -> None:
    """Ordnet einer Phase ihre Eingabedateien aus den verwalteten Arbeitsordnern zu."""
    # Die YAML-Konfiguration liefert Rohdatenpfade und Ergebnisordner.
    # Zwischenprodukte folgen festen Dateinamen innerhalb der Phasenordner.
    prepared = store.directory("prepare")
    if step == "prepare":
        prepare(
            config,
            destination,
            boundary_file=config.paths.boundary,
            osm_pbf=config.paths.osm_pbf,
            gtfs_zip=config.paths.gtfs,
            census_csvs=config.paths.census_csvs,
        )
    elif step == "route":
        route(
            config,
            destination,
            stops_file=prepared / "stops.gpkg",
            pois_file=prepared / "pois.gpkg",
            grid_file=prepared / "grid.gpkg",
            osm_pbf=config.paths.osm_pbf,
            gtfs_zip=config.paths.gtfs,
            checkpoint_directory=store.root / "routing-checkpoints",
        )
    elif step == "sources":
        sources(
            config,
            destination,
            stops_file=prepared / "stops.gpkg",
            pois_file=prepared / "pois.gpkg",
            travel_times_file=store.directory("route") / "travel_times.csv",
            prepare_report_file=prepared / "report.json",
        )
    elif step == "transfer":
        transfer(
            config,
            destination,
            grid_file=prepared / "grid.gpkg",
            sources_file=store.directory("sources") / "sources.csv",
            sources_report_file=store.directory("sources") / "report.json",
        )
    elif step == "score":
        score(config, destination, grid_times_file=store.directory("transfer") / "grid_times.csv")
    elif step == "export":
        export(
            config,
            destination,
            **{f"{name}_directory": store.directory(name) for name in STEPS[:-1]},
        )
