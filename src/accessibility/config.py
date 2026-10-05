"""Einlesen und Validieren der YAML-Konfiguration."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pyproj import CRS

from .exceptions import ConfigurationError

GERMAN_WEEKDAYS: dict[str, int] = {
    "Montag": 0,
    "Dienstag": 1,
    "Mittwoch": 2,
    "Donnerstag": 3,
    "Freitag": 4,
    "Samstag": 5,
    "Sonntag": 6,
}


@dataclass(frozen=True)
class PathsConfig:
    boundary: Path
    osm_pbf: Path
    gtfs: Path
    output_directory: Path
    census_csvs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class AreaConfig:
    analysis_crs: str
    buffer_km: float
    crop_osm: bool = False


@dataclass(frozen=True)
class AnalysisTime:
    weekday: str
    time: time
    weight: float


@dataclass(frozen=True)
class AnalysisConfig:
    reference_week: date
    timezone: str
    departure_time_window_minutes: int
    times: tuple[AnalysisTime, ...]


@dataclass(frozen=True)
class RoutingConfig:
    maximum_travel_time_minutes: int
    maximum_route_walking_time_minutes: int
    maximum_grid_access_walking_time_minutes: int
    walking_speed_kmh: float
    snap_to_network: bool
    origin_batch_size: int = 25
    candidate_selection_enabled: bool = False
    candidate_pois_per_category: tuple[tuple[str, int], ...] = ()
    compute_walk_only_matrix: bool = False

    @property
    def candidate_counts(self) -> dict[str, int]:
        """Gibt die konfigurierte Anzahl der POI-Kandidaten je Kategorie zurück."""
        return dict(self.candidate_pois_per_category)


@dataclass(frozen=True)
class StopsConfig:
    fallback_grouping_distance_meters: float


@dataclass(frozen=True)
class SurfaceConfig:
    enabled: bool
    grid_size_meters: float
    airline_detour_factor: float
    include_direct_walk_to_poi: bool


@dataclass(frozen=True)
class OSMFilter:
    key: str = ""
    values: tuple[str, ...] = ()
    contains: bool = False
    all_of: tuple[OSMFilter, ...] = ()
    any_of: tuple[OSMFilter, ...] = ()


@dataclass(frozen=True)
class POICategory:
    id: str
    name: str
    weight: float
    osm_filters: tuple[OSMFilter, ...]
    data_status: str = "auto"


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    area: AreaConfig
    analysis: AnalysisConfig
    routing: RoutingConfig
    stops: StopsConfig
    surface: SurfaceConfig
    poi_categories: tuple[POICategory, ...]
    source_path: Path


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Prüft, ob ein Konfigurationsabschnitt ein YAML-Objekt ist."""
    if not isinstance(value, dict):
        raise ConfigurationError(f"'{name}' muss ein YAML-Objekt sein.")
    return value


def _list(value: Any, name: str) -> list[Any]:
    """Prüft, ob ein Konfigurationswert eine nicht leere Liste ist."""
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"'{name}' muss eine nicht-leere Liste sein.")
    return value


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    """Liest ein Pflichtfeld oder meldet dessen Fehlen mit dem Abschnittsnamen."""
    if key not in mapping:
        raise ConfigurationError(f"Pflichtfeld '{section}.{key}' fehlt.")
    return mapping[key]


def _positive(value: Any, field: str, *, integer: bool = False) -> float | int:
    """Prüft eine positive, endliche Zahl und bei Bedarf ihre Ganzzahligkeit."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"'{field}' muss eine Zahl größer als 0 sein.")
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"'{field}' muss größer als 0 sein.")
    if integer and int(value) != value:
        raise ConfigurationError(f"'{field}' muss eine ganze Zahl sein.")
    return int(value) if integer else float(value)


def _boolean(value: Any, field: str) -> bool:
    """Akzeptiert ausschließlich echte boolesche Konfigurationswerte."""
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{field}' muss true oder false sein.")
    return value


def _path(value: Any, field: str) -> Path:
    """Prüft einen Pfadtext und löst eine gegebenenfalls enthaltene Tilde auf."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{field}' muss ein nicht-leerer Dateipfad sein.")
    return Path(value).expanduser()


def _parse_time(value: Any, field: str) -> time:
    """Liest eine Uhrzeit aus der Konfiguration und meldet ungültige Angaben."""
    if not isinstance(value, str):
        raise ConfigurationError(f"'{field}' muss die Uhrzeit im Format HH:MM enthalten.")
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise ConfigurationError(f"'{field}' ist keine gültige Uhrzeit (erwartet HH:MM).") from exc


def _parse_date(value: Any, field: str) -> date:
    """Liest ein ISO-Datum aus der Konfiguration und meldet ungültige Angaben."""
    if not isinstance(value, str):
        raise ConfigurationError(f"'{field}' muss ein Datum im ISO-Format YYYY-MM-DD sein.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigurationError(f"'{field}' ist kein gültiges ISO-Datum.") from exc


def validate_paths(paths: PathsConfig) -> None:
    """Prüft, ob alle konfigurierten fachlichen Eingabedateien vorhanden sind."""
    for label, path in [
        ("paths.boundary", paths.boundary),
        ("paths.osm_pbf", paths.osm_pbf),
        ("paths.gtfs", paths.gtfs),
        *[("paths.census_csvs", p) for p in paths.census_csvs],
    ]:
        if not path.is_file():
            raise ConfigurationError(
                f"Eingabedatei '{label}' fehlt: {path}. GTFS-Dateien müssen lokal vorliegen."
            )


def _parse_filter(value: Any) -> OSMFilter:
    """Übersetzt einen einfachen oder verknüpften YAML-Filter in eine OSM-Filterregel."""
    rule = _mapping(value, "osm_filter")
    for operator in ("all", "any"):
        if operator in rule:
            if len(rule) != 1:
                raise ConfigurationError("OSM-Verknüpfungen dürfen nur 'all' oder 'any' enthalten.")
            children = tuple(_parse_filter(child) for child in _list(rule[operator], operator))
            return OSMFilter(**{f"{operator}_of": children})
    key = _required(rule, "key", "osm_filter")
    values = _list(_required(rule, "values", "osm_filter"), "osm_filter.values")
    if not isinstance(key, str) or not key or not all(isinstance(v, str) and v for v in values):
        raise ConfigurationError("Ungültiger OSM-Filter.")
    return OSMFilter(
        key, tuple(values), _boolean(rule.get("contains", False), "osm_filter.contains")
    )


def load_config(path: str | Path, *, validate_input_files: bool = True) -> AppConfig:
    """Lädt Konfiguration; relative Datenpfade beziehen sich auf die YAML-Datei."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Konfigurationsdatei nicht gefunden: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Ungültiges YAML in {source}: {exc}") from exc
    root = _mapping(raw, "Konfiguration")

    paths_raw = _mapping(_required(root, "paths", ""), "paths")
    if not isinstance(paths_raw.get("census_csvs", []), list):
        raise ConfigurationError("'paths.census_csvs' muss eine Liste von CSV-Dateipfaden sein.")
    paths = PathsConfig(
        boundary=_path(_required(paths_raw, "boundary", "paths"), "paths.boundary"),
        osm_pbf=_path(_required(paths_raw, "osm_pbf", "paths"), "paths.osm_pbf"),
        gtfs=_path(_required(paths_raw, "gtfs", "paths"), "paths.gtfs"),
        output_directory=_path(
            _required(paths_raw, "output_directory", "paths"), "paths.output_directory"
        ),
        census_csvs=tuple(_path(p, "paths.census_csvs") for p in paths_raw.get("census_csvs", [])),
    )

    def resolve_path(value: Path) -> Path:
        """Löst relative Datenpfade ausgehend vom Ordner der YAML-Datei auf."""
        return (source.parent / value).resolve() if not value.is_absolute() else value

    # Alle Daten- und Ausgabepfade werden einmal relativ zur YAML-Datei
    # aufgelöst. Die Phasen sind dadurch unabhängig vom Aufrufverzeichnis.
    paths = replace(
        paths,
        boundary=resolve_path(paths.boundary),
        osm_pbf=resolve_path(paths.osm_pbf),
        gtfs=resolve_path(paths.gtfs),
        output_directory=resolve_path(paths.output_directory),
        census_csvs=tuple(resolve_path(value) for value in paths.census_csvs),
    )
    area_raw = _mapping(_required(root, "area", ""), "area")
    crs = _required(area_raw, "analysis_crs", "area")
    if not isinstance(crs, str):
        raise ConfigurationError("'area.analysis_crs' muss ein CRS-String sein, z. B. EPSG:25832.")
    try:
        parsed_crs = CRS.from_user_input(crs)
        # Puffer, Gruppierungsabstände und Kandidatensuche rechnen in Metern.
        # Dafür wird ein projiziertes CRS mit metrischen Einheiten verwendet.
        if not parsed_crs.is_projected or any(
            abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in parsed_crs.axis_info
        ):
            raise ConfigurationError(
                "'area.analysis_crs' muss ein projiziertes CRS in Metern sein."
            )
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"'area.analysis_crs' ist ungültig: {crs}") from exc
    area = AreaConfig(
        crs,
        float(_positive(_required(area_raw, "buffer_km", "area"), "area.buffer_km")),
        _boolean(area_raw.get("crop_osm", False), "area.crop_osm"),
    )

    analysis_raw = _mapping(_required(root, "analysis", ""), "analysis")
    reference_week = _parse_date(
        _required(analysis_raw, "reference_week", "analysis"), "analysis.reference_week"
    )
    if reference_week.weekday() != 0:
        raise ConfigurationError(
            "'analysis.reference_week' muss der Montag der Referenzwoche sein."
        )
    timezone = _required(analysis_raw, "timezone", "analysis")
    if not isinstance(timezone, str):
        raise ConfigurationError("'analysis.timezone' muss ein IANA-Zeitzonenname sein.")
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ConfigurationError(f"Unbekannte Zeitzone '{timezone}'.") from exc
    analysis_times: list[AnalysisTime] = []
    for index, item in enumerate(
        _list(_required(analysis_raw, "times", "analysis"), "analysis.times")
    ):
        entry = _mapping(item, f"analysis.times[{index}]")
        weekday = _required(entry, "weekday", f"analysis.times[{index}]")
        if weekday not in GERMAN_WEEKDAYS:
            valid = ", ".join(GERMAN_WEEKDAYS)
            raise ConfigurationError(f"Unbekannter Wochentag '{weekday}'. Zulässig: {valid}.")
        analysis_times.append(
            AnalysisTime(
                weekday=weekday,
                time=_parse_time(
                    _required(entry, "time", f"analysis.times[{index}]"),
                    f"analysis.times[{index}].time",
                ),
                weight=float(
                    _positive(
                        _required(entry, "weight", f"analysis.times[{index}]"),
                        f"analysis.times[{index}].weight",
                    )
                ),
            )
        )
    analysis = AnalysisConfig(
        reference_week=reference_week,
        timezone=timezone,
        departure_time_window_minutes=int(
            _positive(
                _required(analysis_raw, "departure_time_window_minutes", "analysis"),
                "analysis.departure_time_window_minutes",
                integer=True,
            )
        ),
        times=tuple(analysis_times),
    )

    routing_raw = _mapping(_required(root, "routing", ""), "routing")
    candidate_selection_raw = routing_raw.get("candidate_selection")
    if candidate_selection_raw is None:
        candidate_selection_enabled = False
        candidate_counts_raw = None
    else:
        candidate_selection = _mapping(candidate_selection_raw, "routing.candidate_selection")
        candidate_selection_enabled = _boolean(
            _required(candidate_selection, "enabled", "routing.candidate_selection"),
            "routing.candidate_selection.enabled",
        )
        candidate_counts_raw = candidate_selection.get("candidates_per_category")

    routing = RoutingConfig(
        maximum_travel_time_minutes=int(
            _positive(
                _required(routing_raw, "maximum_travel_time_minutes", "routing"),
                "routing.maximum_travel_time_minutes",
                integer=True,
            )
        ),
        maximum_route_walking_time_minutes=int(
            _positive(
                _required(routing_raw, "maximum_route_walking_time_minutes", "routing"),
                "routing.maximum_route_walking_time_minutes",
                integer=True,
            )
        ),
        maximum_grid_access_walking_time_minutes=int(
            _positive(
                _required(routing_raw, "maximum_grid_access_walking_time_minutes", "routing"),
                "routing.maximum_grid_access_walking_time_minutes",
                integer=True,
            )
        ),
        walking_speed_kmh=float(
            _positive(
                _required(routing_raw, "walking_speed_kmh", "routing"), "routing.walking_speed_kmh"
            )
        ),
        snap_to_network=_boolean(
            _required(routing_raw, "snap_to_network", "routing"), "routing.snap_to_network"
        ),
        origin_batch_size=int(
            _positive(
                routing_raw.get("origin_batch_size", 25), "routing.origin_batch_size", integer=True
            )
        ),
        candidate_selection_enabled=candidate_selection_enabled,
        compute_walk_only_matrix=_boolean(
            routing_raw.get("compute_walk_only_matrix", False), "routing.compute_walk_only_matrix"
        ),
    )
    if routing.maximum_route_walking_time_minutes > routing.maximum_travel_time_minutes:
        raise ConfigurationError(
            "'routing.maximum_route_walking_time_minutes' darf die maximale Reisezeit nicht übersteigen."
        )

    stops_raw = _mapping(_required(root, "stops", ""), "stops")
    stops = StopsConfig(
        float(
            _positive(
                _required(stops_raw, "fallback_grouping_distance_meters", "stops"),
                "stops.fallback_grouping_distance_meters",
            )
        )
    )
    surface_raw = _mapping(_required(root, "surface", ""), "surface")
    surface = SurfaceConfig(
        enabled=_boolean(_required(surface_raw, "enabled", "surface"), "surface.enabled"),
        grid_size_meters=float(
            _positive(
                _required(surface_raw, "grid_size_meters", "surface"), "surface.grid_size_meters"
            )
        ),
        airline_detour_factor=float(
            _positive(
                _required(surface_raw, "airline_detour_factor", "surface"),
                "surface.airline_detour_factor",
            )
        ),
        include_direct_walk_to_poi=_boolean(
            _required(surface_raw, "include_direct_walk_to_poi", "surface"),
            "surface.include_direct_walk_to_poi",
        ),
    )

    category_ids: set[str] = set()
    categories: list[POICategory] = []
    for index, item in enumerate(_list(_required(root, "poi_categories", ""), "poi_categories")):
        entry = _mapping(item, f"poi_categories[{index}]")
        category_id = _required(entry, "id", f"poi_categories[{index}]")
        if not isinstance(category_id, str) or not category_id.strip():
            raise ConfigurationError(
                f"'poi_categories[{index}].id' muss ein nicht-leerer String sein."
            )
        if category_id in category_ids:
            raise ConfigurationError(f"POI-Kategorie-ID '{category_id}' ist mehrfach vorhanden.")
        category_ids.add(category_id)
        filters: list[OSMFilter] = []
        for filter_index, filter_value in enumerate(
            _list(
                _required(entry, "osm_filters", f"poi_categories[{index}]"),
                f"poi_categories[{index}].osm_filters",
            )
        ):
            filters.append(_parse_filter(filter_value))
        name = _required(entry, "name", f"poi_categories[{index}]")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(
                f"'poi_categories[{index}].name' muss ein nicht-leerer String sein."
            )
        data_status = entry.get("data_status", "auto")
        if data_status not in ("auto", "available", "unknown"):
            raise ConfigurationError("data_status muss auto, available oder unknown sein.")
        categories.append(
            POICategory(
                category_id,
                name,
                float(
                    _positive(
                        _required(entry, "weight", f"poi_categories[{index}]"),
                        f"poi_categories[{index}].weight",
                    )
                ),
                tuple(filters),
                data_status,
            )
        )

    candidate_counts: tuple[tuple[str, int], ...] = ()
    if candidate_counts_raw is not None:
        candidate_mapping = _mapping(
            candidate_counts_raw, "routing.candidate_selection.candidates_per_category"
        )
        unknown_categories = set(candidate_mapping) - category_ids
        missing_categories = category_ids - set(candidate_mapping)
        if unknown_categories:
            raise ConfigurationError(
                "Unbekannte POI-Kategorie in 'routing.candidate_selection.candidates_per_category': "
                f"{', '.join(sorted(unknown_categories))}."
            )
        if candidate_selection_enabled and missing_categories:
            raise ConfigurationError(
                "Für die Kandidatenwahl fehlt mindestens eine POI-Kategorie: "
                f"{', '.join(sorted(missing_categories))}."
            )
        candidate_counts = tuple(
            (
                category.id,
                int(
                    _positive(
                        candidate_mapping[category.id],
                        f"routing.candidate_selection.candidates_per_category.{category.id}",
                        integer=True,
                    )
                ),
            )
            for category in categories
            if category.id in candidate_mapping
        )
    elif candidate_selection_enabled:
        raise ConfigurationError(
            "Bei aktivierter Kandidatenwahl muss 'routing.candidate_selection.candidates_per_category' "
            "für jede POI-Kategorie gesetzt sein."
        )
    routing = replace(routing, candidate_pois_per_category=candidate_counts)

    config = AppConfig(paths, area, analysis, routing, stops, surface, tuple(categories), source)
    if validate_input_files:
        validate_paths(config.paths)
    return config
