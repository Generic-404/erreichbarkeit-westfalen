"""GTFS-Haltestellengruppierung nach Parent-Station, Name und Entfernung."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from .exceptions import DataValidationError
from .timepoints import read_gtfs_csv

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StopGroupingReport:
    original_gtfs_stops: int
    stop_groups: int
    stops_inside_boundary: int
    implausible_stops: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        """Stellt die Kennzahlen der Haltestellengruppierung für den JSON-Bericht bereit."""
        return {
            "original_gtfs_stops": self.original_gtfs_stops,
            "stop_groups": self.stop_groups,
            "stops_inside_boundary": self.stops_inside_boundary,
            "implausible_stops": list(self.implausible_stops),
        }


def normalize_stop_name(value: object) -> str:
    """Vereinheitlicht Großschreibung und Leerzeichen für den Namensvergleich."""
    return re.sub(r"\s+", " ", str(value or "").casefold().strip())


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        """Legt zunächst für jede Haltestelle eine eigene Zusammenhangsgruppe an."""
        self.parents = {value: value for value in values}

    def find(self, value: str) -> str:
        """Ermittelt den Gruppenvertreter und verkürzt dabei die Suchkette."""
        if self.parents[value] != value:
            self.parents[value] = self.find(self.parents[value])
        return self.parents[value]

    def union(self, first: str, second: str) -> None:
        """Vereinigt zwei Zusammenhangsgruppen unter einem gemeinsamen Vertreter."""
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parents[second_root] = first_root


def read_gtfs_stops(gtfs_zip: str | Path) -> pd.DataFrame:
    """Liest Haltestellen aus dem GTFS-Archiv und prüft die erforderlichen Spalten."""
    rows = read_gtfs_csv(gtfs_zip, "stops.txt")
    if not rows:
        raise DataValidationError("Die GTFS-ZIP enthält keine stops.txt oder sie ist leer.")
    return pd.DataFrame(rows)


def select_stops_near_boundary(
    stops: pd.DataFrame,
    analysis_crs: str,
    boundary: object,
    fallback_distance_meters: float,
) -> pd.DataFrame:
    """Reduziert GTFS auf die für die Analyse und Fallback-Gruppierung relevanten Stopps.

    Nicht bewertete Stopps weit außerhalb der Gebietsgrenze können weder einen
    Analysepunkt bilden noch innerhalb der Fallback-Distanz in dessen Gruppe
    fallen. Parent-Stationen und alle Plattformgeschwister ausgewählter Gruppen
    bleiben dagegen erhalten.
    """
    required = {"stop_id", "stop_lat", "stop_lon"}
    if missing := required - set(stops.columns):
        raise DataValidationError(f"stops.txt fehlen Pflichtspalten: {', '.join(sorted(missing))}")
    data = stops.copy()
    data["stop_id"] = data.stop_id.fillna("").astype(str)
    data["parent_station"] = (
        data["parent_station"].fillna("").astype(str) if "parent_station" in data else ""
    )
    data["stop_lat"] = pd.to_numeric(data.stop_lat, errors="coerce")
    data["stop_lon"] = pd.to_numeric(data.stop_lon, errors="coerce")
    valid = data.loc[data.stop_id.ne("") & data.stop_lat.notna() & data.stop_lon.notna()].copy()
    if valid.empty:
        return valid
    points = gpd.GeoDataFrame(
        valid, geometry=gpd.points_from_xy(valid.stop_lon, valid.stop_lat), crs="EPSG:4326"
    ).to_crs(analysis_crs)
    # Der zusätzliche Rand berücksichtigt gleichnamige Plattformen, die
    # knapp außerhalb liegen, aber noch zur Fallback-Gruppierung gehören.
    nearby_ids = set(
        points.loc[points.geometry.intersects(boundary.buffer(fallback_distance_meters)), "stop_id"]
    )
    parent_ids = set(data.loc[data.stop_id.isin(nearby_ids), "parent_station"]) - {""}
    # Parent und sämtliche Plattformen der Parent-Gruppe werden zur
    # Bestimmung der gemeinsamen Gruppenkoordinate übernommen.
    group_ids = nearby_ids | parent_ids
    selected = data.loc[data.stop_id.isin(group_ids) | data.parent_station.isin(parent_ids)].copy()
    LOGGER.info("GTFS räumlich vorgefiltert: %d von %d Stopps relevant.", len(selected), len(data))
    return selected


def group_stops(
    stops: pd.DataFrame,
    analysis_crs: str,
    boundary: object,
    fallback_distance_meters: float,
    original_gtfs_stops_count: int | None = None,
) -> tuple[gpd.GeoDataFrame, StopGroupingReport]:
    """Gruppiert Parent-Stationen, dann gleichnamige Plattformen in räumlicher Nähe."""
    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = required - set(stops.columns)
    if missing:
        raise DataValidationError(f"stops.txt fehlen Pflichtspalten: {', '.join(sorted(missing))}")
    original_count = (
        original_gtfs_stops_count if original_gtfs_stops_count is not None else len(stops)
    )
    data = stops.copy()
    data["stop_id"] = data["stop_id"].fillna("").astype(str)
    data["stop_name"] = data["stop_name"].fillna("").astype(str)
    if "parent_station" in data:
        data["parent_station"] = data["parent_station"].fillna("").astype(str)
    else:
        data["parent_station"] = ""
    if "location_type" in data:
        data["location_type"] = data["location_type"].fillna("0").astype(str)
    else:
        data["location_type"] = "0"
    data["stop_lat"] = pd.to_numeric(data["stop_lat"], errors="coerce")
    data["stop_lon"] = pd.to_numeric(data["stop_lon"], errors="coerce")
    # Nicht verwendbare Datensätze werden mit Begründung im Bericht
    # gesammelt, bevor die Gruppierung auf den gültigen Punkten arbeitet.
    bad: list[dict[str, str]] = []
    for _, row in data.iterrows():
        causes: list[str] = []
        if not row.stop_id:
            causes.append("fehlende stop_id")
        if not row.stop_name:
            causes.append("fehlender stop_name")
        if pd.isna(row.stop_lat) or pd.isna(row.stop_lon):
            causes.append("fehlende oder ungültige Koordinate")
        if causes:
            bad.append({"stop_id": row.stop_id, "reason": ", ".join(causes)})
    valid = data.loc[
        data.stop_id.ne("") & data.stop_name.ne("") & data.stop_lat.notna() & data.stop_lon.notna()
    ].copy()
    if valid.empty:
        raise DataValidationError(
            "Keine GTFS-Haltestelle mit stop_id, Name und Koordinaten vorhanden."
        )
    points = gpd.GeoDataFrame(
        valid, geometry=gpd.points_from_xy(valid.stop_lon, valid.stop_lat), crs="EPSG:4326"
    ).to_crs(analysis_crs)
    by_id = points.set_index("stop_id", drop=False)
    parent_ids = set(by_id.index)
    assigned: dict[str, str] = {}

    # Eine vorhandene Parent-Station hat immer Vorrang. Die Station selbst wird Teil ihrer Gruppe.
    parent_groups = points.loc[points.parent_station.ne("")].groupby("parent_station", sort=True)
    for parent_id, platforms in parent_groups:
        members = platforms.stop_id.tolist()
        if parent_id in parent_ids:
            members.append(parent_id)
        for member in members:
            assigned[member] = f"parent:{parent_id}"
        if parent_id not in parent_ids:
            bad.append(
                {
                    "stop_id": parent_id,
                    "reason": "parent_station verweist auf nicht vorhandenen GTFS-Stopp",
                }
            )

    # Nicht zugewiesene Stationen/Plattformen mit gleichem Namen werden als Komponenten gruppiert.
    remaining = points.loc[~points.stop_id.isin(assigned)].copy()
    for name, named in remaining.groupby(
        remaining.stop_name.map(normalize_stop_name), dropna=False
    ):
        ids = named.stop_id.tolist()
        geometries = named.geometry.tolist()
        # Die Nachbarschaft bildet Zusammenhangsgruppen: A kann über B mit
        # C verbunden sein, auch wenn A und C weiter auseinanderliegen.
        union = _UnionFind(ids)
        for position, first_id in enumerate(ids):
            first_geometry = geometries[position]
            for second_id, second_geometry in zip(
                ids[position + 1 :], geometries[position + 1 :], strict=True
            ):
                if first_geometry.distance(second_geometry) <= fallback_distance_meters:
                    union.union(first_id, second_id)
        components: dict[str, list[str]] = {}
        for stop_id in ids:
            components.setdefault(union.find(stop_id), []).append(stop_id)
        for member_ids in components.values():
            # Die kleinste Mitgliedskennung liefert einen reproduzierbaren
            # Gruppennamen unabhängig vom internen Union-Find-Vertreter.
            group_id = f"fallback:{min(member_ids)}"
            for stop_id in member_ids:
                assigned[stop_id] = group_id

    grouped_records: list[dict[str, Any]] = []
    for group_id, group in points.groupby(points.stop_id.map(assigned), sort=True):
        member_ids = group.stop_id.tolist()
        parent_id = group_id.removeprefix("parent:") if group_id.startswith("parent:") else None
        # Eine vorhandene Parent-Koordinate wird übernommen. Andernfalls
        # repräsentiert der Schwerpunkt der vereinigten Punkte die Gruppe.
        if parent_id and parent_id in by_id.index:
            coordinate = by_id.loc[parent_id].geometry
        else:
            coordinate = group.geometry.union_all().centroid
        names = group.stop_name.tolist()
        primary_name = (
            by_id.loc[parent_id, "stop_name"]
            if parent_id and parent_id in by_id.index
            else names[0]
        )
        grouped_records.append(
            {
                "stop_group_id": group_id,
                "stop_name": primary_name,
                "parent_station": parent_id,
                "source_stop_ids": ";".join(sorted(member_ids)),
                "platform_count": len(member_ids),
                "geometry": coordinate,
            }
        )
    groups = gpd.GeoDataFrame(grouped_records, geometry="geometry", crs=analysis_crs)
    if not groups.geometry.is_valid.all():
        raise DataValidationError("Ungültige Geometrie bei gruppierten Haltestellen.")
    # Erst die fertige Gruppenkoordinate entscheidet über die räumliche
    # Aufnahme; einzelne Plattformen werden nicht vorzeitig abgeschnitten.
    selected = groups.loc[groups.geometry.intersects(boundary)].copy().reset_index(drop=True)
    report = StopGroupingReport(
        original_gtfs_stops=original_count,
        stop_groups=len(groups),
        stops_inside_boundary=len(selected),
        implausible_stops=tuple(bad),
    )
    LOGGER.info(
        "%d GTFS-Stopps zu %d Gruppen, %d innerhalb der Grenze gruppiert.",
        original_count,
        len(groups),
        len(selected),
    )
    return selected, report
