"""Konfigurationsgesteuerte POI-Extraktion aus dem fertigen regionalen OSM-PBF."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

import geopandas as gpd
from shapely import wkb
from shapely.geometry import Point

from .config import POICategory
from .exceptions import DataValidationError, DependencyError

LOGGER = logging.getLogger(__name__)


def matches_filters(tags: Mapping[str, str], filters: Iterable[object]) -> bool:
    """Gibt True zurück, wenn wenigstens eine alternative Filterregel passt."""

    def matches(rule: object) -> bool:
        """Prüft eine einzelne Tagregel einschließlich ihrer UND- oder ODER-Verknüpfungen."""
        if rule.all_of:
            return all(matches(child) for child in rule.all_of)
        if rule.any_of:
            return any(matches(child) for child in rule.any_of)
        # contains prüft einzelne Werte eines durch Semikolon getrennten
        # OSM-Tags; sonst muss der gesamte Tagwert mit einer Vorgabe übereinstimmen.
        value = tags.get(str(rule.key), "")
        return (
            bool({v.strip() for v in value.split(";")} & set(rule.values))
            if rule.contains
            else value in rule.values
        )

    return any(matches(rule) for rule in filters)


def filter_keys(filters: Iterable[object]) -> set[str]:
    """Sammelt die benötigten OSM-Tag-Schlüssel aus allen verschachtelten Regeln."""
    keys: set[str] = set()
    for rule in filters:
        if rule.key:
            keys.add(rule.key)
        keys.update(filter_keys(rule.all_of))
        keys.update(filter_keys(rule.any_of))
    return keys


def matching_categories(
    tags: Mapping[str, str], categories: Iterable[POICategory]
) -> list[POICategory]:
    """Bestimmt sämtliche Kategorien, deren Filter zu den OSM-Tags passen."""
    return [category for category in categories if matches_filters(tags, category.osm_filters)]


def _deduplicate_pois(frame: gpd.GeoDataFrame, distance_meters: float = 2.0) -> gpd.GeoDataFrame:
    """Entfernt nahe POI-Dubletten mit gleicher Kategorie und gleichem normalisiertem Namen."""
    if frame.empty:
        return frame
    keep: list[int] = []
    for _, group in frame.groupby(["category_id", "normalized_name"], dropna=False):
        accepted: list[int] = []
        for index, row in group.iterrows():
            # Ohne Namen erfolgt keine Zusammenlegung. Bei gleichem Namen
            # und gleicher Kategorie entscheidet die kurze Distanzschwelle.
            if not row["normalized_name"]:
                accepted.append(index)
                continue
            if any(
                row.geometry.distance(frame.loc[other].geometry) <= distance_meters
                for other in accepted
            ):
                continue
            accepted.append(index)
        keep.extend(accepted)
    return frame.loc[keep].copy()


def extract_pois(
    osm_pbf: str | Path,
    categories: tuple[POICategory, ...],
    area_of_interest: object,
    analysis_crs: str,
) -> gpd.GeoDataFrame:
    """Extrahiert OSM-Nodes und -Flächen, die einer konfigurierten Kategorie entsprechen.

    ``area_of_interest`` muss eine Geometrie im ``analysis_crs`` sein und enthält
    bereits die verlangte Pufferzone. Tags bleiben als JSON im Ergebnis erhalten.
    """
    path = Path(osm_pbf)
    if not path.is_file():
        raise DataValidationError(f"Regionaler OSM-PBF fehlt: {path}")
    try:
        import osmium
    except ImportError as exc:
        raise DependencyError(
            "Python-Paket 'osmium' fehlt. Installieren Sie die Projektabhängigkeiten."
        ) from exc

    records: list[dict[str, object]] = []
    factory = osmium.geom.WKBFactory()
    # Der native TagFilter lässt nur konfigurationsrelevante Objekte bis zu den
    # Python-Callbacks durch. Die weitere Auswahl erfolgt auf dieser Teilmenge.
    keys = sorted(set().union(*(filter_keys(category.osm_filters) for category in categories)))
    tag_filter = osmium.filter.KeyFilter(*keys)

    def add_record(
        osm_id: str, tags: dict[str, str], geometry: Point, matched_categories: list[POICategory]
    ) -> None:
        """Erfasst Geometrie und Tags eines POIs für jede passende Kategorie."""
        # Ein Objekt kann mehreren Kategorien entsprechen. Die Kategorie
        # ist deshalb Teil der POI-Kennung, die OSM-Kennung bleibt zusätzlich erhalten.
        for category in matched_categories:
            records.append(
                {
                    "poi_id": f"{category.id}:{osm_id}",
                    "osm_id": osm_id,
                    "name": tags.get("name") or None,
                    "normalized_name": (tags.get("name") or "").casefold().strip(),
                    "category_id": category.id,
                    "category_name": category.name,
                    "tags": json.dumps(tags, ensure_ascii=False, sort_keys=True),
                    "geometry": geometry,
                }
            )

    class NodePOIHandler(osmium.SimpleHandler):
        def node(self, node: object) -> None:
            """Verarbeitet einen OSM-Punkt mit gültiger Koordinate als möglichen POI."""
            matched_categories = matching_categories(node.tags, categories)
            if not matched_categories or not node.location.valid():
                return
            tags = dict(node.tags)
            add_record(
                f"node/{node.id}",
                tags,
                Point(node.location.lon, node.location.lat),
                matched_categories,
            )

    class AreaPOIHandler(osmium.SimpleHandler):
        def area(self, area: object) -> None:
            """Verarbeitet eine OSM-Fläche über ihren repräsentativen Punkt als möglichen POI."""
            matched_categories = matching_categories(area.tags, categories)
            if not matched_categories:
                return
            try:
                # Flächen werden für das Punkt-zu-Punkt-Routing durch einen
                # garantiert auf ihrer Fläche liegenden Punkt repräsentiert.
                geometry = wkb.loads(
                    factory.create_multipolygon(area), hex=True
                ).representative_point()
            except (RuntimeError, ValueError, TypeError) as exc:
                LOGGER.warning("Ungültige OSM-Fläche %s wird übersprungen: %s", area.id, exc)
                return
            origin = "relation" if area.is_relation() else "way"
            add_record(f"{origin}/{area.orig_id()}", dict(area.tags), geometry, matched_categories)

    # Punkte können direkt gelesen werden. Der separate Flächendurchlauf
    # benötigt zusätzlich die Knotenpositionen zur Polygonbildung.
    try:
        NodePOIHandler().apply_file(
            str(path),
            filters=[tag_filter.enable_for(osmium.osm.NODE)],
        )
        AreaPOIHandler().apply_file(
            str(path),
            locations=True,
            filters=[osmium.filter.KeyFilter(*keys).enable_for(osmium.osm.AREA)],
        )
    except RuntimeError as exc:
        raise DataValidationError(f"OSM-PBF konnte nicht gelesen werden: {exc}") from exc
    columns = [
        "poi_id",
        "osm_id",
        "name",
        "normalized_name",
        "category_id",
        "category_name",
        "tags",
        "geometry",
    ]
    pois = gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs="EPSG:4326")
    if pois.empty:
        return pois.drop(columns="normalized_name")
    # Räumliche Auswahl und Dublettenabstand werden nach der Projektion
    # geprüft, damit Gebiet und Punkte im selben metrischen System liegen.
    pois = pois.to_crs(analysis_crs)
    pois = pois.loc[pois.geometry.intersects(area_of_interest)].copy()
    pois = _deduplicate_pois(pois)
    return pois.drop(columns="normalized_name").reset_index(drop=True)
