from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from accessibility.config import AnalysisTime, load_config
from accessibility.exceptions import ConfigurationError, DataValidationError
from accessibility.timepoints import derive_timepoint, validate_gtfs_timepoints


def _config(tmp_path: Path) -> dict:
    """Erzeugt eine minimale gültige Konfiguration mit vorhandenen Testdateien."""
    for name in ("boundary.gpkg", "map.pbf", "feed.zip"):
        (tmp_path / name).touch()
    return {
        "paths": {
            "boundary": str(tmp_path / "boundary.gpkg"),
            "osm_pbf": str(tmp_path / "map.pbf"),
            "gtfs": str(tmp_path / "feed.zip"),
            "output_directory": str(tmp_path / "out"),
        },
        "area": {"analysis_crs": "EPSG:25832", "buffer_km": 1},
        "analysis": {
            "reference_week": "2026-09-07",
            "timezone": "Europe/Berlin",
            "departure_time_window_minutes": 1,
            "times": [{"weekday": "Mittwoch", "time": "12:00", "weight": 1}],
        },
        "routing": {
            "maximum_travel_time_minutes": 90,
            "maximum_route_walking_time_minutes": 15,
            "maximum_grid_access_walking_time_minutes": 15,
            "walking_speed_kmh": 4.5,
            "snap_to_network": True,
        },
        "stops": {"fallback_grouping_distance_meters": 100},
        "surface": {
            "enabled": True,
            "grid_size_meters": 200,
            "airline_detour_factor": 1.25,
            "include_direct_walk_to_poi": True,
        },
        "poi_categories": [
            {
                "id": "school",
                "name": "School",
                "weight": 1,
                "osm_filters": [{"key": "amenity", "values": ["school"]}],
            }
        ],
    }


def test_loads_valid_configuration(tmp_path: Path) -> None:
    """Prüft das Einlesen der Konfiguration einschließlich Standardwerten."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(tmp_path)), encoding="utf-8")
    config = load_config(path)
    assert config.area.analysis_crs == "EPSG:25832"
    assert config.analysis.times[0].weekday == "Mittwoch"
    assert config.routing.origin_batch_size == 25


def test_rejects_unknown_weekday(tmp_path: Path) -> None:
    """Prüft die Ablehnung eines unbekannten Wochentags."""
    raw = _config(tmp_path)
    raw["analysis"]["times"][0]["weekday"] = "Fronntag"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unbekannter Wochentag"):
        load_config(path)


def test_rejects_missing_input_file(tmp_path: Path) -> None:
    """Prüft die Fehlermeldung für eine fehlende Eingabedatei."""
    raw = _config(tmp_path)
    raw["paths"]["gtfs"] = str(tmp_path / "not-there.zip")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Eingabedatei"):
        load_config(path)


def test_rejects_string_instead_of_boolean(tmp_path: Path) -> None:
    """Prüft, dass Textwerte keine booleschen Einstellungen ersetzen."""
    raw = _config(tmp_path)
    raw["routing"]["snap_to_network"] = "false"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="true oder false"):
        load_config(path)


def test_loads_per_category_candidate_configuration(tmp_path: Path) -> None:
    """Prüft die Übernahme der Kandidatenzahl je POI-Kategorie."""
    raw = _config(tmp_path)
    raw["routing"]["candidate_selection"] = {
        "enabled": True,
        "candidates_per_category": {"school": 7},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(path)

    assert config.routing.candidate_selection_enabled is True
    assert config.routing.candidate_counts == {"school": 7}


def test_rejects_incomplete_candidate_configuration(tmp_path: Path) -> None:
    """Prüft die Ablehnung unvollständiger Kandidateneinstellungen."""
    raw = _config(tmp_path)
    raw["routing"]["candidate_selection"] = {"enabled": True, "candidates_per_category": {}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="fehlt mindestens eine POI-Kategorie"):
        load_config(path)


def test_derives_weekday_to_date() -> None:
    """Prüft Datum und Kennung des aus der Referenzwoche abgeleiteten Zeitpunkts."""
    point = derive_timepoint(
        date(2026, 9, 7), AnalysisTime("Mittwoch", __import__("datetime").time(12), 1.0)
    )
    assert point.date.isoformat() == "2026-09-09"
    assert point.id == "2026-09-09T1200"


def test_rejects_timepoint_without_gtfs_service(tmp_path: Path) -> None:
    # Ein leeres, gültiges GTFS-Archiv enthält kein Angebot für den angefragten Tag.
    """Prüft die Ablehnung eines Analysezeitpunkts ohne GTFS-Angebot."""
    import zipfile

    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w"):
        pass
    point = derive_timepoint(
        date(2026, 9, 7), AnalysisTime("Mittwoch", __import__("datetime").time(12), 1.0)
    )
    with pytest.raises(DataValidationError, match="außerhalb"):
        validate_gtfs_timepoints(archive, [point])


def test_config_paths_are_relative_to_yaml_from_another_working_directory(
    tiny_project, monkeypatch
):
    """Prüft die Auflösung der Datenpfade unabhängig vom aktuellen Arbeitsverzeichnis."""
    source = tiny_project.source_path
    raw = yaml.safe_load(source.read_text())
    for name, value in raw["paths"].items():
        raw["paths"][name] = (
            [Path(path).name for path in value] if isinstance(value, list) else Path(value).name
        )
    source.write_text(yaml.safe_dump(raw))
    elsewhere = source.parent / "unrelated"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = load_config(source)
    assert loaded.paths == tiny_project.paths
