"""R5py-Adapter für Haltestellen-zu-POI-Reisezeiten."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import RoutingConfig
from .exceptions import DataValidationError, DependencyError
from .progress import timed
from .r5_adapter import ADAPTER_VERSION, minimum_transit_matrix
from .timepoints import ConcreteTimepoint

LOGGER = logging.getLogger(__name__)
ORIGIN_BATCH_SIZE = 25


def _r5_input_copy(source: str | Path, directory: Path, suffix: str) -> Path:
    """R5py verknüpft nur Basenames: unveränderliche, eindeutige Namen verwenden."""
    source = Path(source)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    # Der Inhalts-Hash im Namen trennt gleichnamige Dateien verschiedener
    # Datenstände und erlaubt die Wiederverwendung einer geprüften Kopie.
    target = directory / f"accessibility-input-{digest}{suffix}"
    if target.is_file():
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == digest:
                return target
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".input-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise DataValidationError("R5-Eingabedatei wurde während des Kopierens verändert.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _file_identity(path: str | Path) -> dict[str, object]:
    """Kennzeichnet eine Routing-Eingabedatei anhand von Pfad, Größe und Änderungszeit."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _geometry_identity(frame: gpd.GeoDataFrame) -> list[tuple[str, str]]:
    """Erfasst Kennungen und Geometrien zur Zuordnung zwischengespeicherter Reisezeiten."""
    return [
        (str(row.id), row.geometry.wkb_hex)
        for row in frame[["id", "geometry"]].itertuples(index=False)
    ]


def _checkpoint_signature(
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    osm_pbf: str | Path,
    gtfs_zip: str | Path,
    timepoint: ConcreteTimepoint,
    routing: RoutingConfig,
    departure_time_window_minutes: int,
    candidate_destination_ids: dict[str, set[str]] | None,
) -> str:
    """Bildet eine Prüfsignatur aus den Eingaben eines Routingblocks."""
    payload = {
        "format_version": 2,
        "adapter": ADAPTER_VERSION,
        "adapter_code": hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("r5_adapter.py").read_bytes()
        ).hexdigest(),
        "r5py": version("r5py"),
        "origins": _geometry_identity(origins),
        "destinations": _geometry_identity(destinations),
        "osm_pbf": _file_identity(osm_pbf),
        "gtfs_zip": _file_identity(gtfs_zip),
        "timepoint": timepoint.id,
        "routing": {
            k: v
            for k, v in asdict(routing).items()
            if k not in ("maximum_grid_access_walking_time_minutes", "compute_walk_only_matrix")
        },
        "departure_time_window_minutes": departure_time_window_minutes,
        "candidate_destinations": _candidate_identity(candidate_destination_ids),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()[:16]


def _candidate_identity(candidate_destination_ids: dict[str, set[str]] | None) -> str | None:
    """Verdichtet die POI-Kandidaten je Ursprung zu einer Prüfsumme."""
    if candidate_destination_ids is None:
        return None
    digest = hashlib.sha256()
    for origin_id in sorted(candidate_destination_ids):
        digest.update(origin_id.encode("utf-8"))
        digest.update(b"\0")
        for destination_id in sorted(candidate_destination_ids[origin_id]):
            digest.update(destination_id.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _nearest_candidate_ids(
    stops: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    candidate_counts: dict[str, int],
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Wählt je Haltestelle die luftliniennächsten POI-Kandidaten mit einem KD-Baum.

    R5 bestimmt anschließend die Reisezeiten innerhalb dieser Kandidatenmenge.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise DependencyError(
            "Paket 'scipy' fehlt für die räumliche POI-Kandidatenauswahl."
        ) from exc
    stop_ids = stops.stop_group_id.astype(str).tolist()
    candidates = {stop_id: set() for stop_id in stop_ids}
    stop_coordinates = np.column_stack(
        (stops.geometry.x.to_numpy(dtype=float), stops.geometry.y.to_numpy(dtype=float))
    )
    report: dict[str, dict[str, int]] = {}
    for category_id, requested_count in candidate_counts.items():
        category_pois = pois.loc[pois.category_id.eq(category_id)]
        available_count = len(category_pois)
        # Die konfigurierte Zahl gilt je Kategorie; bei weniger vorhandenen
        # POIs werden alle verfügbaren Einrichtungen dieser Kategorie gewählt.
        effective_count = min(requested_count, available_count)
        report[category_id] = {
            "requested": requested_count,
            "available": available_count,
            "effective": effective_count,
        }
        if effective_count == 0:
            continue
        coordinates = np.column_stack(
            (
                category_pois.geometry.x.to_numpy(dtype=float),
                category_pois.geometry.y.to_numpy(dtype=float),
            )
        )
        # Die Suche nutzt die projizierten Punktkoordinaten. Sie wählt nach
        # Luftlinie vor; die Fahrplanreisezeit wird erst anschließend bestimmt.
        _, indexes = cKDTree(coordinates).query(stop_coordinates, k=effective_count)
        indexes = np.asarray(indexes).reshape(len(stops), effective_count)
        poi_ids = category_pois.poi_id.astype(str).to_numpy()
        for position, stop_id in enumerate(stop_ids):
            candidates[stop_id].update(poi_ids[indexes[position]])
    return candidates, report


def _checkpoint_path(
    directory: Path,
    signature: str,
    timepoint: ConcreteTimepoint,
    modes: list[str],
    batch_number: int,
) -> Path:
    """Leitet den Dateinamen eines Routingblocks aus Zeitpunkt, Modus und Blocknummer ab."""
    mode_name = "-".join(mode.lower() for mode in modes)
    return directory / signature / timepoint.id / f"{mode_name}-{batch_number:03d}.csv"


def _load_checkpoint(
    path: Path, expected_origin_ids: set[str], expected_destination_ids: set[str] | None = None
) -> pd.DataFrame | None:
    """Lädt einen vollständigen, passenden Routingblock oder verwirft den Cacheeintrag."""
    if not path.is_file():
        return None
    try:
        cached = pd.read_csv(path, dtype={"from_id": str, "to_id": str})
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        LOGGER.warning("Ignoriere nicht lesbaren Routing-Checkpoint %s: %s", path, exc)
        return None
    expected_columns = ["from_id", "to_id", "travel_time_minutes"]
    if list(cached.columns) != expected_columns or set(cached.from_id) != expected_origin_ids:
        LOGGER.warning(
            "Ignoriere unvollständigen oder nicht passenden Routing-Checkpoint %s.", path
        )
        return None
    if cached.duplicated(["from_id", "to_id"]).any():
        LOGGER.warning("Ignoriere Routing-Checkpoint mit doppelten OD-Paaren %s.", path)
        return None
    # Ein Checkpoint muss auch unerreichbare Ursprung-Ziel-Paare enthalten.
    # Nur eine vollständige Rechteckmatrix gilt als abgeschlossener Block.
    if expected_destination_ids is not None and (
        set(cached.to_id) != expected_destination_ids
        or len(cached) != len(expected_origin_ids) * len(expected_destination_ids)
    ):
        return None
    numeric = pd.to_numeric(cached.travel_time_minutes, errors="coerce")
    if (
        (cached.travel_time_minutes.notna() & numeric.isna()).any()
        or numeric.lt(0).any()
        or np.isinf(numeric).any()
    ):
        return None
    cached["travel_time_minutes"] = numeric
    return cached


def _write_checkpoint(path: Path, matrix: pd.DataFrame) -> None:
    """Speichert einen Routingblock atomar, damit Teildateien nicht als Ergebnis gelten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            matrix.to_csv(handle, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _as_minutes(series: pd.Series) -> pd.Series:
    """Wandelt Zeitdauern oder numerische Werte in Minuten um."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds() / 60
    return pd.to_numeric(series, errors="coerce")


def _matrix(
    r5py: Any,
    network: Any,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    point: ConcreteTimepoint,
    routing: RoutingConfig,
    modes: list[str],
    departure_time_window_minutes: int,
) -> pd.DataFrame:
    # R5py erwartet die lokale Fahrplandatum/-uhrzeit als naives datetime.
    """Berechnet eine R5-Reisezeitmatrix und vereinheitlicht ihre Ergebnisspalten."""
    matrix = minimum_transit_matrix(
        r5py,
        network() if callable(network) else network,
        origins=origins,
        destinations=destinations,
        snap_to_network=routing.snap_to_network,
        departure=point.time,
        departure_time_window=timedelta(minutes=departure_time_window_minutes),
        percentiles=[50],
        transport_modes=["TRANSIT"],
        access_modes=["WALK"],
        egress_modes=["WALK"],
        max_time=timedelta(minutes=routing.maximum_travel_time_minutes),
        max_time_walking=timedelta(minutes=routing.maximum_route_walking_time_minutes),
        speed_walking=routing.walking_speed_kmh,
    )
    result = pd.DataFrame(matrix).copy()
    expected = {"from_id", "to_id", "travel_time"}
    if missing := expected - set(result.columns):
        raise DependencyError(
            f"R5py lieferte keine erwartete Matrixstruktur. Fehlende Spalten: {', '.join(sorted(missing))}"
        )
    result["from_id"] = result.from_id.astype(str)
    result["to_id"] = result.to_id.astype(str)
    result["travel_time_minutes"] = _as_minutes(result.travel_time)
    if result.travel_time_minutes.lt(0).any() or np.isinf(result.travel_time_minutes).any():
        raise DataValidationError("R5 lieferte ungültige Reisezeiten.")
    # Alle angefragten Paare bleiben in der Tabelle. Von R5 nicht gelieferte
    # Verbindungen erhalten durch den linken Join einen unbekannten Zeitwert.
    expected_pairs = pd.MultiIndex.from_product(
        [origins.id.astype(str), destinations.id.astype(str)], names=["from_id", "to_id"]
    ).to_frame(index=False)
    return expected_pairs.merge(
        result[["from_id", "to_id", "travel_time_minutes"]],
        on=["from_id", "to_id"],
        how="left",
        validate="one_to_one",
    )


def _batched_matrix(
    r5py: Any,
    network: Any,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    point: ConcreteTimepoint,
    routing: RoutingConfig,
    modes: list[str],
    departure_time_window_minutes: int,
    batch_size: int = ORIGIN_BATCH_SIZE,
    checkpoint_directory: str | Path | None = None,
    checkpoint_signature: str | None = None,
    candidate_destination_ids: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Berechnet eine große OD-Matrix in kleinen, wiederaufnehmbaren Origin-Blöcken.

    R5 hält während einer Matrixberechnung Java-seitige Suchdaten vor. Blöcke
    begrenzen deren Lebensdauer ohne die fachlichen Ergebnisse zu verändern. Falls
    ein Checkpoint-Verzeichnis übergeben wird, ist jeder abgeschlossene Block
    atomar gespeichert und bei identischem Routing-Input wiederverwendbar.
    """
    chunks: list[pd.DataFrame] = []
    total = len(origins)
    checkpoint_root = Path(checkpoint_directory) if checkpoint_directory is not None else None
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        origin_batch = origins.iloc[start:stop]
        destination_batch = destinations
        if candidate_destination_ids is not None:
            # R5 berechnet eine Rechteckmatrix für den gesamten Block.
            # Ihre Zielmenge ist die Vereinigung der individuellen Kandidaten.
            candidate_ids = set().union(
                *(
                    candidate_destination_ids.get(origin_id, set())
                    for origin_id in origin_batch.id.astype(str)
                )
            )
            destination_batch = destinations.loc[destinations.id.isin(candidate_ids)]
        checkpoint = None
        if checkpoint_root is not None and checkpoint_signature is not None:
            checkpoint = _checkpoint_path(
                checkpoint_root, checkpoint_signature, point, modes, start // batch_size + 1
            )
            cached = _load_checkpoint(
                checkpoint, set(origin_batch.id.astype(str)), set(destination_batch.id.astype(str))
            )
            # Ein gültiger Block wird ohne erneuten Netzaufbau übernommen.
            if cached is not None:
                LOGGER.info(
                    "R5-Matrix %s: Haltestellen %d–%d von %d aus Checkpoint wiederhergestellt.",
                    "/".join(modes),
                    start + 1,
                    stop,
                    total,
                )
                chunks.append(_filter_candidates(cached, candidate_destination_ids))
                continue
        LOGGER.info(
            "R5-Matrix %s: Haltestellen %d–%d von %d, %d Ziel-POIs.",
            "/".join(modes),
            start + 1,
            stop,
            total,
            len(destination_batch),
        )
        matrix = (
            _matrix(
                r5py,
                network,
                origin_batch,
                destination_batch,
                point,
                routing,
                modes,
                departure_time_window_minutes,
            )
            if not destination_batch.empty
            else pd.DataFrame(columns=["from_id", "to_id", "travel_time_minutes"])
        )
        if checkpoint is not None:
            _write_checkpoint(checkpoint, matrix)
        # Aus der gemeinsamen Zielmenge bleiben je Haltestelle nur deren
        # eigene Kandidaten erhalten; fremde Blockkandidaten zählen nicht mit.
        chunks.append(_filter_candidates(matrix, candidate_destination_ids))
    return (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=["from_id", "to_id", "travel_time_minutes"])
    )


def _filter_candidates(
    matrix: pd.DataFrame, candidates: dict[str, set[str]] | None
) -> pd.DataFrame:
    """Beschränkt eine Reisezeitmatrix auf die je Ursprung ausgewählten POI-Kandidaten."""
    if candidates is None or matrix.empty:
        return matrix
    selected = [
        (origin, target)
        for origin in matrix.from_id.unique()
        for target in candidates.get(origin, set())
    ]
    return matrix.merge(
        pd.DataFrame(selected, columns=["from_id", "to_id"]), on=["from_id", "to_id"], how="inner"
    )


def hilbert_blocks(stops: gpd.GeoDataFrame, batch_size: int) -> gpd.GeoDataFrame:
    """Sortiert Haltestellen räumlich entlang einer Hilbert-Kurve und nummeriert ihre Blöcke."""
    # Die Vorsortierung nach Kennung legt die Reihenfolge bei gleichen
    # Hilbert-Werten fest. Räumlich nahe Ursprünge landen so in kleinen Blöcken.
    ordered = stops.sort_values("stop_group_id", kind="stable").copy()
    if len(ordered) > 1 and np.any(
        np.ptp(np.column_stack((ordered.geometry.x, ordered.geometry.y)), axis=0) > 0
    ):
        ordered["hilbert_order"] = ordered.hilbert_distance()
    else:
        ordered["hilbert_order"] = 0
    ordered = ordered.sort_values("hilbert_order", kind="stable").reset_index(drop=True)
    ordered["block_id"] = np.arange(len(ordered)) // batch_size + 1
    return ordered


def route_stop_to_pois(
    stops: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    osm_pbf: str | Path,
    gtfs_zip: str | Path,
    timepoint: ConcreteTimepoint,
    routing: RoutingConfig,
    category_ids: list[str],
    departure_time_window_minutes: int,
    checkpoint_directory: str | Path | None = None,
    plan_directory: str | Path | None = None,
    timezone: str | None = None,
) -> pd.DataFrame:
    """ÖPNV-OD-Werte; Kategorie-Minima entstehen erst im separaten sources-Schritt."""
    if stops.empty or pois.empty:
        return pd.DataFrame(columns=["from_id", "to_id", "travel_time_minutes", "route_mode"])
    try:
        import r5py
    except ImportError as exc:
        raise DependencyError(
            "Paket 'r5py' fehlt. Installieren Sie die Projektabhängigkeiten und OpenJDK 21+."
        ) from exc
    candidate_destination_ids: dict[str, set[str]] | None = None
    if routing.candidate_selection_enabled:
        candidate_destination_ids, candidate_report = _nearest_candidate_ids(
            stops, pois, routing.candidate_counts
        )
        LOGGER.info(
            "Luftlinien-Kandidatenauswahl aktiv: %s.",
            "; ".join(
                f"{category}: {values['effective']} von {values['available']} POIs (konfiguriert: {values['requested']})"
                for category, values in candidate_report.items()
            ),
        )
    ordered_stops = hilbert_blocks(stops, routing.origin_batch_size)
    # Die räumliche Vorauswahl erfolgt im Analysekoordinatensystem.
    # R5 erhält anschließend Längen- und Breitengrade in EPSG:4326.
    origins = (
        ordered_stops[["stop_group_id", "geometry"]]
        .rename(columns={"stop_group_id": "id"})
        .to_crs("EPSG:4326")
    )
    destinations = pois[["poi_id", "geometry"]].rename(columns={"poi_id": "id"}).to_crs("EPSG:4326")
    origins["id"] = origins.id.astype(str)
    destinations["id"] = destinations.id.astype(str)
    checkpoint_signature = _checkpoint_signature(
        origins,
        destinations,
        osm_pbf,
        gtfs_zip,
        timepoint,
        routing,
        departure_time_window_minutes,
        candidate_destination_ids,
    )
    # Blockzuordnung und Kandidatenlisten dokumentieren, welche Paare
    # angefragt werden und wie groß die tatsächlich berechneten Matrizen sind.
    if plan_directory is not None:
        plan = Path(plan_directory)
        ordered_stops.drop(columns="geometry").to_csv(plan / "blocks.csv", index=False)
        candidate_rows = [
            (str(stop), str(poi))
            for stop in origins.id
            for poi in sorted(
                candidate_destination_ids[str(stop)]
                if candidate_destination_ids is not None
                else destinations.id
            )
        ]
        pd.DataFrame(candidate_rows, columns=["from_id", "to_id"]).to_csv(
            plan / "candidates.csv", index=False
        )
        sizes = []
        for block_id, block in ordered_stops.groupby("block_id"):
            targets = (
                set().union(*(candidate_destination_ids[str(s)] for s in block.stop_group_id))
                if candidate_destination_ids is not None
                else set(destinations.id)
            )
            sizes.append(
                {
                    "block_id": int(block_id),
                    "origins": len(block),
                    "destinations": len(targets),
                    "matrix_pairs": len(block) * len(targets),
                }
            )
        pd.DataFrame(sizes).to_csv(plan / "matrix_sizes.csv", index=False)
    # Der Netzaufbau wird aufgeschoben, bis ein Block neu berechnet werden
    # muss. Vollständig vorhandene Checkpoints benötigen ihn nicht.
    loaded_network = None

    def network():
        """Erstellt das R5-Verkehrsnetz beim ersten Bedarf und verwendet es danach erneut."""
        nonlocal loaded_network
        if loaded_network is None:
            # R5py hält persistente Links: keine Ziele in temporären Run-/Testordnern.
            input_cache = (
                Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
                / "accessibility"
                / "r5-inputs"
            )
            with timed("R5-Eingaben für sicheren Cache vorbereiten"):
                osm_input = _r5_input_copy(osm_pbf, input_cache, ".osm.pbf")
                gtfs_input = _r5_input_copy(gtfs_zip, input_cache, ".gtfs.zip")
            with timed("R5-Transportnetz laden oder aufbauen"):
                loaded_network = r5py.TransportNetwork(str(osm_input), [str(gtfs_input)])
            if timezone is not None and str(loaded_network.timezone) != timezone:
                raise DependencyError(
                    f"R5-Netzzeitzone {loaded_network.timezone} stimmt nicht mit {timezone} überein."
                )
        return loaded_network

    combined = _batched_matrix(
        r5py,
        network,
        origins,
        destinations,
        timepoint,
        routing,
        ["TRANSIT", "WALK"],
        departure_time_window_minutes,
        batch_size=routing.origin_batch_size,
        checkpoint_directory=checkpoint_directory,
        checkpoint_signature=checkpoint_signature,
        candidate_destination_ids=candidate_destination_ids,
    )
    combined["route_mode"] = np.where(
        combined.travel_time_minutes.notna(), "transit", "no_route_within_limits"
    )
    return combined


def _best_by_category(
    pair_times: pd.DataFrame,
    pois: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    timepoint: ConcreteTimepoint,
    category_ids: list[str],
) -> pd.DataFrame:
    """Wählt je Haltestelle und Kategorie das Ziel mit der kleinsten bekannten Reisezeit.

    Bei gleichen Reisezeiten bewahrt die stabile Sortierung die Reihenfolge der Eingabe.
    """
    poi_info = pd.DataFrame(pois.drop(columns="geometry"))[["poi_id", "name", "category_id"]]
    candidates = pair_times.merge(poi_info, left_on="to_id", right_on="poi_id", how="inner")
    valid = candidates.dropna(subset=["travel_time_minutes"])
    best = (
        valid.sort_values("travel_time_minutes", kind="stable")
        .drop_duplicates(["from_id", "category_id"], keep="first")[
            ["from_id", "category_id", "travel_time_minutes", "poi_id", "name", "route_mode"]
        ]
        .rename(columns={"from_id": "stop_group_id", "name": "poi_name"})
    )
    # Das vollständige Haltestellen-Kategorien-Raster erhält auch Fälle
    # ohne bekannte Verbindung; sie verschwinden nicht durch die Minimumbildung.
    base = pd.MultiIndex.from_product(
        [stops.stop_group_id.astype(str), category_ids],
        names=["stop_group_id", "category_id"],
    ).to_frame(index=False)
    result = base.merge(best, on=["stop_group_id", "category_id"], how="left", sort=False)
    result.insert(1, "timepoint_id", timepoint.id)
    result["route_mode"] = result.route_mode.fillna("unreachable")
    return result[
        [
            "stop_group_id",
            "timepoint_id",
            "category_id",
            "travel_time_minutes",
            "poi_id",
            "poi_name",
            "route_mode",
        ]
    ]
