"""Ursprüngliche Quellen und einmalige Reisezeitübertragung; kein Routing ab Zellen."""

from __future__ import annotations

from collections.abc import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .census import GRID_CRS, cell_coordinates
from .exceptions import DataValidationError

SOURCE_COLUMNS = [
    "grid_id",
    "category_id",
    "base_minutes",
    "source_type",
    "stop_group_id",
    "poi_id",
    "center_x",
    "center_y",
]


def walking_minutes(
    distance_meters: float, detour_factor: float, walking_speed_kmh: float
) -> float:
    """Rechnet Luftlinienentfernungen mit Umwegfaktor und Gehgeschwindigkeit in Minuten um."""
    return distance_meters * detour_factor / (walking_speed_kmh * 1000 / 60)


def build_sources(
    stops: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, travel: pd.DataFrame
) -> pd.DataFrame:
    """Ein bestes ursprüngliches Angebot je Quellzelle/Kategorie, einschließlich Geh-POIs."""
    # Die geroutete Reisezeit wird dem Mittelpunkt der Haltestellenzelle
    # als Basiswert zugeordnet; unbekannte Reisezeiten bilden keine Quelle.
    stop_cells = cell_coordinates(stops).assign(stop_group_id=stops.stop_group_id.astype(str))
    transit = travel.loc[travel.travel_time_minutes.notna()].copy()
    transit["stop_group_id"] = transit.stop_group_id.astype(str)
    transit = transit.merge(stop_cells, on="stop_group_id", validate="many_to_one")
    transit = transit.rename(columns={"travel_time_minutes": "base_minutes"}).assign(
        source_type="transit"
    )
    # Ein direkt erreichbarer POI startet in seiner eigenen Zelle mit
    # null Minuten. Gehzeit zu anderen Zellen wird erst später addiert.
    direct = cell_coordinates(pois).assign(
        category_id=pois.category_id.to_numpy(),
        poi_id=pois.poi_id.to_numpy(),
        base_minutes=0.0,
        source_type="direct_walk",
        stop_group_id="",
    )
    combined = pd.concat(
        [transit.reindex(columns=SOURCE_COLUMNS), direct.reindex(columns=SOURCE_COLUMNS)],
        ignore_index=True,
    )
    combined["base_minutes"] = pd.to_numeric(combined.base_minutes, errors="raise")
    if combined.base_minutes.lt(0).any() or not np.isfinite(combined.base_minutes).all():
        raise DataValidationError("Ungültige Basisreisezeit einer Erreichbarkeitsquelle.")
    # Mehrere Angebote derselben Kategorie in einer Zelle werden auf das
    # schnellste reduziert. Weitere Sortierschlüssel entscheiden Gleichstände.
    return (
        combined.sort_values(
            ["base_minutes", "source_type", "poi_id", "stop_group_id"], kind="stable"
        )
        .drop_duplicates(["grid_id", "category_id"])
        .reset_index(drop=True)
    )


def transfer_times(
    grid: gpd.GeoDataFrame,
    sources: pd.DataFrame,
    category_status: dict[str, str],
    walking_limit: float,
    walking_speed: float,
    detour_factor: float,
    chunk_size: int = 10_000,
) -> Iterator[pd.DataFrame]:
    """Liefert Zellblöcke; Quellendaten bleiben während der Übertragung unverändert."""
    if str(grid.crs) != GRID_CRS:
        raise DataValidationError("Die Übertragung erwartet Zensuszellen in EPSG:3035.")
    radius = walking_limit * walking_speed * 1000 / 60 / detour_factor
    # Ein räumlicher Suchbaum je Kategorie begrenzt die Suche auf Quellen
    # innerhalb der Gehreichweite. Diese Quellmenge bleibt für alle Blöcke fest.
    indexed = {}
    for category in category_status:
        rows = sources.loc[sources.category_id.eq(category)].reset_index(drop=True)
        coords = rows[["center_x", "center_y"]].to_numpy(float)
        indexed[category] = (rows, coords, cKDTree(coords) if len(rows) else None)
    for offset in range(0, len(grid), chunk_size):
        block = grid.iloc[offset : offset + chunk_size]
        coords = block[["center_x", "center_y"]].to_numpy(float)
        output = []
        for category, status in category_status.items():
            # NaN steht für unbekannte Daten, unendlich für keine Quelle
            # innerhalb der Grenzen bei bekannter Datenlage. Das Scoring
            # unterscheidet diese Fälle später von einer Reisezeit von null.
            result = pd.DataFrame(
                {
                    "grid_id": block.grid_id.to_numpy(),
                    "category_id": category,
                    "travel_time_minutes": np.nan if status == "unknown" else np.inf,
                    "status": "unknown_data" if status == "unknown" else "no_source_within_limits",
                    "source_grid_id": "",
                    "source_type": "",
                    "stop_group_id": "",
                    "poi_id": "",
                    "base_minutes": np.nan,
                    "access_minutes": np.nan,
                }
            )
            rows, source_coords, tree = indexed[category]
            if status != "unknown" and tree is not None:
                neighbors = tree.query_ball_point(coords, radius + 1e-8)
                lengths = np.fromiter(map(len, neighbors), dtype=np.int64, count=len(neighbors))
                if lengths.sum():
                    # Die Nachbarlisten werden zu Zell-Quellen-Paaren
                    # aufgefächert, deren Gehzeiten gemeinsam berechnet werden.
                    cell_indexes = np.repeat(np.arange(len(block)), lengths)
                    source_indexes = np.concatenate(
                        [np.asarray(n, dtype=np.int64) for n in neighbors if len(n)]
                    )
                    delta = coords[cell_indexes] - source_coords[source_indexes]
                    walks = (
                        np.hypot(delta[:, 0], delta[:, 1])
                        * detour_factor
                        / (walking_speed * 1000 / 60)
                    )
                    # Entscheidend ist Basisreisezeit plus Rasterzugang,
                    # nicht allein die Entfernung zur nächsten Quelle.
                    costs = rows.base_minutes.to_numpy(float)[source_indexes] + walks
                    # lexsort priorisiert den letzten Schlüssel: erst Zelle,
                    # dann Gesamtreisezeit, bei Gleichstand Quellenindex.
                    order = np.lexsort((source_indexes, costs, cell_indexes))
                    ordered_cells = cell_indexes[order]
                    first = np.r_[True, ordered_cells[1:] != ordered_cells[:-1]]
                    selected = order[first]
                    target = cell_indexes[selected]
                    chosen = rows.iloc[source_indexes[selected]]
                    result.loc[target, "travel_time_minutes"] = costs[selected]
                    result.loc[target, "status"] = "ok"
                    # Die gewählte Quelle und ihre Zeitanteile bleiben im
                    # Ergebnis erhalten, damit der Zellwert nachvollziehbar ist.
                    for destination, original in (
                        ("source_grid_id", "grid_id"),
                        ("source_type", "source_type"),
                        ("stop_group_id", "stop_group_id"),
                        ("poi_id", "poi_id"),
                        ("base_minutes", "base_minutes"),
                    ):
                        result.loc[target, destination] = chosen[original].to_numpy()
                    result.loc[target, "access_minutes"] = walks[selected]
            output.append(result)
        yield pd.concat(output, ignore_index=True).sort_values(
            ["grid_id", "category_id"], kind="stable"
        )
