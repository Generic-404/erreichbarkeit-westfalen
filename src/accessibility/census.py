"""100-m-Zensusgitter: EPSG:3035, Zellkennung bezeichnet die Südwestecke."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .exceptions import DataValidationError

LOGGER = logging.getLogger(__name__)
GRID_CRS = "EPSG:3035"
CELL_SIZE = 100


def cell_coordinates(points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Ordnet Punkten die Kennung und den Mittelpunkt ihrer 100-m-Zensuszelle zu."""
    projected = points.to_crs(GRID_CRS)
    # Abrunden auf das globale 100-m-Gitter liefert die Südwestecke.
    # Der für Gehentfernungen verwendete Mittelpunkt liegt 50 m versetzt.
    east = (np.floor(projected.geometry.x.to_numpy() / CELL_SIZE) * CELL_SIZE).astype(np.int64)
    north = (np.floor(projected.geometry.y.to_numpy() / CELL_SIZE) * CELL_SIZE).astype(np.int64)
    return pd.DataFrame(
        {
            "grid_id": [f"CRS3035RES100mN{n}E{e}" for e, n in zip(east, north, strict=True)],
            "center_x": east + 50.0,
            "center_y": north + 50.0,
        },
        index=points.index,
    )


def generate_census_grid(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Erzeugt alle 100-m-Zellen mit positiver Schnittfläche zum Untersuchungsgebiet."""
    shape = boundary.to_crs(GRID_CRS).geometry.union_all()
    shapely.prepare(shape)
    x0, y0, x1, y1 = shape.bounds
    # Die Rasterlinien bleiben am Zensusgitter ausgerichtet und werden
    # nicht an die untere linke Ecke des Untersuchungsgebiets verschoben.
    east = np.arange(np.floor(x0 / 100) * 100, np.ceil(x1 / 100) * 100, 100, dtype=np.int64)
    north = np.arange(np.floor(y0 / 100) * 100, np.ceil(y1 / 100) * 100, 100, dtype=np.int64)
    pieces = []
    # Zeilenweise Teilraster begrenzen den Speicher für die Geometrieprüfung.
    rows_per_piece = max(1, 100_000 // max(1, len(east)))
    total_positions = len(east) * len(north)
    selected_count = 0
    for offset in range(0, len(north), rows_per_piece):
        xx, yy = np.meshgrid(east, north[offset : offset + rows_per_piece])
        xx, yy = xx.ravel(), yy.ravel()
        cells = shapely.box(xx, yy, xx + 100, yy + 100)
        intersects = shapely.intersects(shape, cells)
        xx, yy, cells = xx[intersects], yy[intersects], cells[intersects]
        # Innenliegende Zellen werden direkt übernommen. Für Randzellen
        # entscheidet die positive Schnittfläche über die Aufnahme.
        positive_area = shapely.contains_properly(shape, cells)
        edge = ~positive_area
        positive_area[edge] = shapely.area(shapely.intersection(cells[edge], shape)) > 0
        xx, yy, cells = xx[positive_area], yy[positive_area], cells[positive_area]
        # Randzellen werden vollständig gespeichert, sobald ihre Schnittfläche
        # mit dem Gebiet positiv ist; die Zellpolygone werden nicht beschnitten.
        pieces.append(
            gpd.GeoDataFrame(
                {
                    "grid_id": [f"CRS3035RES100mN{n}E{e}" for e, n in zip(xx, yy, strict=True)],
                    "center_x": xx + 50.0,
                    "center_y": yy + 50.0,
                },
                geometry=cells,
                crs=GRID_CRS,
            )
        )
        selected_count += len(cells)
        processed = min(offset + rows_per_piece, len(north)) * len(east)
        LOGGER.info(
            "Raster: %.1f %% der Positionen geprüft (%s/%s), %s Zellen übernommen.",
            100 * processed / total_positions,
            f"{processed:,}",
            f"{total_positions:,}",
            f"{selected_count:,}",
        )
    if not pieces:
        raise DataValidationError("Das Untersuchungsgebiet enthält keine Zensuszellen.")
    return gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=GRID_CRS)


def attach_census_attributes(
    grid: gpd.GeoDataFrame, paths: tuple[Path, ...]
) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """Deutschland-CSV stückweise lesen; fehlende Attribute entfernen keine Zelle."""
    reports = []
    wanted = set(grid.grid_id)
    for file_number, path in enumerate(paths, start=1):
        selected = []
        checked = 0
        try:
            # Die bundesweiten Dateien werden blockweise als Text gelesen.
            # Dezimalkommas und Zensus-Sonderzeichen bleiben zunächst erhalten.
            for chunk in pd.read_csv(
                path, sep=";", dtype=str, encoding="utf-8-sig", chunksize=100_000
            ):
                required = {"GITTER_ID_100m", "x_mp_100m", "y_mp_100m"}
                if not required <= set(chunk):
                    raise DataValidationError(
                        f"{path}: erwartet Zensus-100-m-Spalten {sorted(required)}."
                    )
                relevant = chunk.loc[chunk.GITTER_ID_100m.isin(wanted)]
                # Die ersten Datensätze und alle regional passenden Zeilen
                # werden auf Zellkennung, Mittelpunkt und Gitterlage geprüft.
                sample = pd.concat([chunk.head(100) if checked == 0 else chunk.iloc[:0], relevant])
                parts = sample.GITTER_ID_100m.str.extract(r"^CRS3035RES100mN(\d+)E(\d+)$")
                x = pd.to_numeric(sample.x_mp_100m, errors="coerce")
                y = pd.to_numeric(sample.y_mp_100m, errors="coerce")
                n = pd.to_numeric(parts[0], errors="coerce")
                e = pd.to_numeric(parts[1], errors="coerce")
                if not (
                    parts.notna().all().all()
                    and ((x - e) == 50).all()
                    and ((y - n) == 50).all()
                    and (e % 100 == 0).all()
                    and (n % 100 == 0).all()
                ):
                    raise DataValidationError(
                        f"{path}: Zellkennungen oder Mittelpunkte passen nicht zum EPSG:3035-100-m-Gitter."
                    )
                checked += len(chunk)
                if not relevant.empty:
                    selected.append(relevant)
        except (ValueError, pd.errors.ParserError) as exc:
            raise DataValidationError(f"Zensus-CSV nicht lesbar: {path}: {exc}") from exc
        if checked == 0:
            raise DataValidationError(f"Zensus-CSV ist leer: {path}")
        matched = 0
        if selected:
            values = pd.concat(selected, ignore_index=True)
            if values.GITTER_ID_100m.duplicated().any():
                raise DataValidationError(f"Doppelte Zellkennungen in {path}.")
            values = values.drop(columns=["x_mp_100m", "y_mp_100m"]).rename(
                columns={"GITTER_ID_100m": "grid_id"}
            )
            # Die Position in census_csvs wird Teil des Spaltennamens.
            # Dadurch bleiben gleichnamige Attribute verschiedener Dateien
            # unterscheidbar und ihrer Eingabedatei zuordenbar.
            values = values.rename(
                columns={
                    column: f"zensus_{file_number}_{column}"
                    for column in values
                    if column != "grid_id"
                }
            )
            matched = len(values)
            # Fehlende Zensusangaben entfernen keine Zelle aus dem Raster.
            grid = grid.merge(values, on="grid_id", how="left", validate="one_to_one")
        reports.append(
            {
                "path": str(path),
                "rows_read": checked,
                "matched_cells": matched,
                "attribute_prefix": f"zensus_{file_number}_",
                "crs": GRID_CRS,
                "cell_id_origin": "southwest",
                "center_offset_m": 50,
            }
        )
        LOGGER.info("Zensus %s: %d passende von %d Datensätzen.", path.name, matched, checked)
    return grid, reports
