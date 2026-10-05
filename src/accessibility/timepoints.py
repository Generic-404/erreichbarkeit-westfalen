"""Zeitpunkte und GTFS-Kalenderprüfung ohne Routing-Abhängigkeit."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from .config import GERMAN_WEEKDAYS, AnalysisConfig, AnalysisTime
from .exceptions import DataValidationError


@dataclass(frozen=True)
class ConcreteTimepoint:
    id: str
    weekday: str
    date: date
    time: datetime
    weight: float


def derive_timepoint(reference_week: date, item: AnalysisTime) -> ConcreteTimepoint:
    """Leitet den lokalen, naiven GTFS-Abfahrtszeitpunkt aus Montag + Wochentag ab."""
    service_date = reference_week + timedelta(days=GERMAN_WEEKDAYS[item.weekday])
    # R5 erwartet die lokale Fahrplanuhrzeit ohne angehängte Zeitzone.
    # Die Übereinstimmung der Netzzeitzone wird beim Netzaufbau geprüft.
    dt = datetime.combine(service_date, item.time)
    return ConcreteTimepoint(
        id=f"{service_date.isoformat()}T{item.time.strftime('%H%M')}",
        weekday=item.weekday,
        date=service_date,
        time=dt,
        weight=item.weight,
    )


def derive_timepoints(config: AnalysisConfig) -> tuple[ConcreteTimepoint, ...]:
    """Leitet alle konkreten Abfahrtszeitpunkte aus der Analysekonfiguration ab."""
    return tuple(derive_timepoint(config.reference_week, item) for item in config.times)


def _find_member(archive: zipfile.ZipFile, basename: str) -> str | None:
    """Findet eine GTFS-Datei auch innerhalb eines Unterordners im ZIP-Archiv."""
    matches = [
        name
        for name in archive.namelist()
        if name.lower().endswith(f"/{basename.lower()}") or name.lower() == basename.lower()
    ]
    return matches[0] if matches else None


def read_gtfs_csv(gtfs_zip: str | Path, basename: str) -> list[dict[str, str]]:
    """Liest eine GTFS-Textdatei aus einer ZIP, auch bei einem Ordnerpräfix im Archiv."""
    path = Path(gtfs_zip)
    try:
        with zipfile.ZipFile(path) as archive:
            member = _find_member(archive, basename)
            if member is None:
                return []
            with archive.open(member) as stream:
                text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
                return list(csv.DictReader(text))
    except zipfile.BadZipFile as exc:
        raise DataValidationError(f"GTFS-Datei ist keine lesbare ZIP-Datei: {path}") from exc


def _parse_gtfs_date(value: str) -> date:
    """Wandelt ein GTFS-Datum im Format YYYYMMDD in ein Datum um."""
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise DataValidationError(f"Ungültiges GTFS-Datum '{value}'.") from exc


def active_service_ids(gtfs_zip: str | Path, service_date: date) -> set[str]:
    """Ermittelt nach calendar.txt und calendar_dates.txt die aktiven Service-IDs."""
    active: set[str] = set()
    weekday_columns = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    # Der reguläre Kalender verlangt sowohl einen passenden Datumsbereich
    # als auch einen aktiven Wochentag.
    for row in read_gtfs_csv(gtfs_zip, "calendar.txt"):
        if not row.get("service_id"):
            continue
        try:
            in_period = (
                _parse_gtfs_date(row["start_date"])
                <= service_date
                <= _parse_gtfs_date(row["end_date"])
            )
        except KeyError as exc:
            raise DataValidationError("calendar.txt enthält nicht alle Pflichtspalten.") from exc
        if in_period and row.get(weekday_columns[service_date.weekday()], "0") == "1":
            active.add(row["service_id"])
    # Datumsbezogene Ausnahmen ergänzen oder entfernen Dienste anschließend.
    # Damit werden auch Feeds mit ausschließlich Einzeltagen berücksichtigt.
    for row in read_gtfs_csv(gtfs_zip, "calendar_dates.txt"):
        if row.get("date") != service_date.strftime("%Y%m%d"):
            continue
        service_id = row.get("service_id")
        if not service_id:
            continue
        if row.get("exception_type") == "1":
            active.add(service_id)
        elif row.get("exception_type") == "2":
            active.discard(service_id)
    return active


def validate_gtfs_timepoints(gtfs_zip: str | Path, timepoints: Iterable[ConcreteTimepoint]) -> None:
    """Prüft, ob für jeden Analysezeitpunkt mindestens ein GTFS-Verkehrsangebot besteht."""
    for point in timepoints:
        if not active_service_ids(gtfs_zip, point.date):
            raise DataValidationError(
                f"Der Analysezeitpunkt {point.date.isoformat()} ({point.weekday}) liegt außerhalb "
                "der wirksamen GTFS-Fahrplandaten oder enthält keinen aktiven Dienst."
            )
