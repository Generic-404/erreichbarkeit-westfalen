from __future__ import annotations

from datetime import date, datetime

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from accessibility.config import OSMFilter, POICategory
from accessibility.osm import matching_categories
from accessibility.routing import _best_by_category, _nearest_candidate_ids
from accessibility.stops import group_stops
from accessibility.timepoints import ConcreteTimepoint


def test_poi_filter_is_configured_not_hard_coded() -> None:
    """Prüft die Auswahl von POIs anhand konfigurierter OSM-Filter."""
    school = POICategory("schools", "Schulen", 1.0, (OSMFilter("amenity", ("school",)),))
    pharmacy = POICategory(
        "pharmacies",
        "Apotheken",
        1.0,
        (OSMFilter("amenity", ("pharmacy",)), OSMFilter("healthcare", ("pharmacy",))),
    )
    assert [
        item.id for item in matching_categories({"healthcare": "pharmacy"}, (school, pharmacy))
    ] == ["pharmacies"]
    assert matching_categories({"amenity": "library"}, (school, pharmacy)) == []


def test_groups_parent_and_nearby_equal_names() -> None:
    """Prüft Parent-Stationen und die räumliche Gruppierung gleichnamiger Haltestellen."""
    stops = pd.DataFrame(
        [
            {
                "stop_id": "P",
                "stop_name": "Hauptbahnhof",
                "stop_lat": 51.9600,
                "stop_lon": 7.6200,
                "location_type": "1",
                "parent_station": "",
            },
            {
                "stop_id": "A",
                "stop_name": "Hauptbahnhof",
                "stop_lat": 51.9601,
                "stop_lon": 7.6201,
                "location_type": "0",
                "parent_station": "P",
            },
            {
                "stop_id": "B",
                "stop_name": "Hauptbahnhof",
                "stop_lat": 51.9602,
                "stop_lon": 7.6202,
                "location_type": "0",
                "parent_station": "P",
            },
            {
                "stop_id": "C",
                "stop_name": "Markt",
                "stop_lat": 51.9610,
                "stop_lon": 7.6210,
                "location_type": "0",
                "parent_station": "",
            },
            {
                "stop_id": "D",
                "stop_name": " markt ",
                "stop_lat": 51.9611,
                "stop_lon": 7.6211,
                "location_type": "0",
                "parent_station": "",
            },
        ]
    )
    points = gpd.GeoSeries([Point(7.62, 51.96)], crs="EPSG:4326").to_crs("EPSG:25832")
    groups, report = group_stops(stops, "EPSG:25832", points.iloc[0].buffer(10_000), 100)
    assert len(groups) == 2
    assert set(
        groups.loc[groups.stop_group_id.eq("parent:P"), "source_stop_ids"].iloc[0].split(";")
    ) == {"A", "B", "P"}
    assert report.original_gtfs_stops == 5
    assert report.stops_inside_boundary == 2


def test_groups_stops_when_optional_gtfs_columns_are_absent() -> None:
    """Prüft die Haltestellengruppierung ohne optionale GTFS-Spalten."""
    stops = pd.DataFrame(
        [{"stop_id": "A", "stop_name": "Platz", "stop_lat": 51.96, "stop_lon": 7.62}]
    )
    point = gpd.GeoSeries([Point(7.62, 51.96)], crs="EPSG:4326").to_crs("EPSG:25832").iloc[0]
    groups, _ = group_stops(stops, "EPSG:25832", point.buffer(100), 100)
    assert len(groups) == 1


def test_selects_configured_airline_candidates_per_category() -> None:
    """Prüft Anzahl und Entfernung der ausgewählten POI-Kandidaten je Kategorie."""
    stops = gpd.GeoDataFrame(
        {"stop_group_id": ["left", "right"]},
        geometry=[Point(0, 0), Point(100, 0)],
        crs="EPSG:25832",
    )
    pois = gpd.GeoDataFrame(
        {
            "poi_id": ["s1", "s2", "s3", "h1"],
            "category_id": ["schools", "schools", "schools", "hospitals"],
        },
        geometry=[Point(5, 0), Point(90, 0), Point(200, 0), Point(50, 0)],
        crs="EPSG:25832",
    )

    candidates, report = _nearest_candidate_ids(stops, pois, {"schools": 2, "hospitals": 5})

    assert candidates["left"] == {"s1", "s2", "h1"}
    assert candidates["right"] == {"s1", "s2", "h1"}
    assert report["schools"] == {"requested": 2, "available": 3, "effective": 2}
    assert report["hospitals"] == {"requested": 5, "available": 1, "effective": 1}


def test_reduces_all_candidate_pairs_to_best_poi_per_stop_and_category() -> None:
    """Prüft das schnellste Ziel je Haltestelle und Kategorie samt unerreichbaren Fällen."""
    stops = gpd.GeoDataFrame(
        {"stop_group_id": ["s1", "s2"]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:25832"
    )
    pois = gpd.GeoDataFrame(
        {
            "poi_id": ["p1", "p2", "p3"],
            "name": ["Langsam", "Schnell", "Klinik"],
            "category_id": ["schools", "schools", "hospitals"],
        },
        geometry=[Point(2, 2), Point(3, 3), Point(4, 4)],
        crs="EPSG:25832",
    )
    pairs = pd.DataFrame(
        [
            {
                "from_id": "s1",
                "to_id": "p1",
                "travel_time_minutes": 20.0,
                "route_mode": "transit_or_walk",
            },
            {
                "from_id": "s1",
                "to_id": "p2",
                "travel_time_minutes": 5.0,
                "route_mode": "transit_or_walk",
            },
            {
                "from_id": "s2",
                "to_id": "p3",
                "travel_time_minutes": 15.0,
                "route_mode": "transit_or_walk",
            },
        ]
    )
    point = ConcreteTimepoint("t", "Mittwoch", date(2026, 9, 9), datetime(2026, 9, 9, 12), 1.0)

    result = _best_by_category(pairs, pois, stops, point, ["schools", "hospitals"])

    assert len(result) == 4
    assert (
        result.loc[
            (result.stop_group_id == "s1") & (result.category_id == "schools"), "poi_id"
        ].iloc[0]
        == "p2"
    )
    assert (
        result.loc[
            (result.stop_group_id == "s1") & (result.category_id == "hospitals"), "route_mode"
        ].iloc[0]
        == "unreachable"
    )
