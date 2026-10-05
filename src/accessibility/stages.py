"""Explizite Prozessgrenzen mit atomaren Ergebnissen und gezielter Invalidierung."""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from .config import AppConfig
from .exceptions import DataValidationError

STEPS = ("prepare", "route", "sources", "transfer", "score", "export")
# Die Abhängigkeiten verlangen fertige Vorgänger. Ein Einzelaufruf startet
# diese nicht automatisch, sondern verwendet ihre gespeicherten Ergebnisse.
PARENTS = {
    "prepare": (),
    "route": ("prepare",),
    "sources": ("prepare", "route"),
    "transfer": ("prepare", "sources"),
    "score": ("transfer",),
    "export": ("prepare", "route", "sources", "transfer", "score"),
}
MODULES = {
    "prepare": ("census", "osm", "stops"),
    "route": ("routing", "r5_adapter", "timepoints"),
    "sources": ("surface", "census", "routing"),
    "transfer": ("surface",),
    "score": ("scoring",),
    "export": ("outputs",),
}


def write_json(path: Path, data: object) -> None:
    """Schreibt strukturierte Daten als lesbare UTF-8-JSON-Datei."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> dict:
    """Liest eine UTF-8-JSON-Datei mit Phasen- oder Laufmetadaten."""
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict:
    """Erfasst Pfad, Größe und Änderungszeit einer Datei oder kennzeichnet ihr Fehlen."""
    try:
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except FileNotFoundError:
        return {"path": str(path.resolve()), "missing": True}


class StageStore:
    def __init__(self, config: AppConfig):
        """Leitet Ergebnisordner und internen Arbeitsordner aus der Konfiguration ab."""
        self.config = config
        self.output = config.paths.output_directory.resolve()
        # Zwischenprodukte liegen neben dem Export, damit sie bei dessen
        # Ersetzung erhalten bleiben und separat weiterverwendbar sind.
        self.root = self.output.parent / f".{self.output.name}.work"

    def directory(self, step: str) -> Path:
        """Gibt den Arbeitsordner einer Phase beziehungsweise den fertigen Exportordner zurück."""
        return self.output if step == "export" else self.root / step

    def settings(self, step: str) -> dict:
        """Sammelt die für den Statusvergleich einer Phase verwendeten Einstellungen."""
        c = self.config
        if step == "prepare":
            boundary_files = [c.paths.boundary]
            if c.paths.boundary.suffix.lower() == ".shp":
                boundary_files = [
                    c.paths.boundary.with_suffix(s) for s in (".shp", ".shx", ".dbf", ".prj")
                ]
            return {
                "inputs": [
                    identity(p)
                    for p in [*boundary_files, c.paths.osm_pbf, c.paths.gtfs, *c.paths.census_csvs]
                ],
                "area": asdict(c.area),
                "stops": asdict(c.stops),
                "grid": "CRS3035RES100m",
                "categories": [
                    {"id": x.id, "name": x.name, "filters": [asdict(f) for f in x.osm_filters]}
                    for x in c.poi_categories
                ],
            }
        if step == "route":
            return {
                "analysis": asdict(c.analysis),
                "routing": {
                    k: v
                    for k, v in asdict(c.routing).items()
                    if k
                    not in ("maximum_grid_access_walking_time_minutes", "compute_walk_only_matrix")
                },
                "r5py": version("r5py"),
            }
        if step == "sources":
            return {"data_status": {x.id: x.data_status for x in c.poi_categories}}
        if step == "transfer":
            return {
                "walking_limit": c.routing.maximum_grid_access_walking_time_minutes,
                "walking_speed": c.routing.walking_speed_kmh,
                "detour_factor": c.surface.airline_detour_factor,
            }
        if step == "score":
            return {
                "weights_percent": {x.id: x.weight for x in c.poi_categories},
                "function": "100 / 146.11*exp(-0.039*t) / 0; boundaries 10,60",
            }
        return {"category_names": {x.id: x.name for x in c.poi_categories}, "map_max_cells": 20_000}

    def signature(self, step: str, parents: dict[str, dict]) -> str:
        """Verknüpft Einstellungen, Modulprüfsummen und Vorgängerläufe zu einer Prüfsignatur."""
        code_root = Path(__file__).parent
        code = {
            name: hashlib.sha256((code_root / f"{name}.py").read_bytes()).hexdigest()
            for name in (*MODULES[step], "pipeline", "stages")
        }
        payload = {
            "format": 1,
            "settings": self.settings(step),
            "code": code,
            "parents": {name: parents[name]["run_id"] for name in PARENTS[step]},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def inspect(self) -> list[dict]:
        """Ermittelt für jede Phase, ob ihre gespeicherten Ergebnisse aktuell und vollständig sind."""
        results = []
        complete = {}
        # Die Reihenfolge folgt den Abhängigkeiten. Nur aktuelle Vorgänger
        # werden in complete für die Prüfung nachfolgender Phasen gesammelt.
        for step in STEPS:
            folder = self.directory(step)
            record = {"step": step, "status": "missing", "reason": "Noch nicht ausgeführt."}
            try:
                manifest = read_json(folder / "manifest.json")
                if any(parent not in complete for parent in PARENTS[step]):
                    record.update(status="stale", reason="Vorgänger fehlt oder ist veraltet.")
                elif manifest.get("signature") != self.signature(step, complete):
                    record.update(
                        status="stale", reason="Eingaben, Parameter, Code oder Vorgänger geändert."
                    )
                elif not manifest.get("artifacts") or any(
                    not (folder / file).is_file()
                    or {k: identity(folder / file).get(k) for k in ("size", "mtime_ns")} != expected
                    for file, expected in manifest["artifacts"].items()
                ):
                    record.update(
                        status="incomplete", reason="Ergebnisdatei fehlt oder wurde verändert."
                    )
                else:
                    complete[step] = manifest
                    record.update(status="complete", reason="Aktuell und vollständig.")
            except FileNotFoundError:
                pass
            except (ValueError, KeyError, TypeError):
                record.update(status="incomplete", reason="Ungültiges Abschlussmanifest.")
            # Das Versuchprotokoll beschreibt einen Abbruch auch dann,
            # wenn noch kein vollständiges Abschlussmanifest existiert.
            attempt = self.root / f"{step}.attempt.json"
            if attempt.exists() and record["status"] != "complete":
                try:
                    record["last_attempt"] = read_json(attempt)
                    if record["status"] == "missing":
                        record.update(
                            status="incomplete",
                            reason="Letzter Versuch fehlgeschlagen oder nicht abgeschlossen; kein gültiges Ergebnis.",
                        )
                except ValueError:
                    pass
            results.append(record)
        return results

    def require(self, step: str) -> dict[str, dict]:
        """Prüft die erforderlichen Vorgänger und liest deren Abschlussmanifeste."""
        state = {row["step"]: row for row in self.inspect()}
        for parent in PARENTS[step]:
            if state[parent]["status"] != "complete":
                raise DataValidationError(
                    f"'{step}' benötigt einen aktuellen Schritt '{parent}'. Zuerst '{parent} --config {self.config.source_path}' ausführen. {state[parent]['reason']}"
                )
        return {name: read_json(self.directory(name) / "manifest.json") for name in PARENTS[step]}

    @contextmanager
    def transaction(self, step: str):
        """Sperrt den Lauf und ersetzt die Phasenausgabe erst nach vollständigem Erfolg."""
        self.root.mkdir(parents=True, exist_ok=True)
        # Ein exklusiver Dateilock verhindert gleichzeitige Schreibvorgänge
        # verschiedener Prozesse für denselben Ergebnisordner.
        with (self.root / "process.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DataValidationError(
                    "Für diesen Ergebnisordner läuft bereits ein Verarbeitungsschritt."
                ) from exc
            parents = self.require(step)
            signature = self.signature(step, parents)
            destination = self.directory(step)
            temporary = Path(tempfile.mkdtemp(prefix=f".{step}-", dir=destination.parent))
            start = time.monotonic()
            attempt = self.root / f"{step}.attempt.json"
            write_json(
                attempt, {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
            )
            try:
                # Während yield schreibt die aufrufende Phase ihre Dateien.
                # Erst nach ihrer Rückkehr werden Vollständigkeit und Eingaben geprüft.
                yield temporary
                # Ergebnisse eines während der Ausführung veränderten Inputs nicht freigeben.
                current_parents = self.require(step)
                if self.signature(step, current_parents) != signature:
                    raise DataValidationError(
                        "Eingaben wurden während der Verarbeitung geändert. Schritt erneut ausführen."
                    )
                # Dateigröße und Änderungszeit werden im Manifest festgehalten,
                # damit spätere Phasen fehlende oder veränderte Ausgaben erkennen.
                artifacts = {
                    str(p.relative_to(temporary)): {
                        "size": p.stat().st_size,
                        "mtime_ns": p.stat().st_mtime_ns,
                    }
                    for p in temporary.rglob("*")
                    if p.is_file()
                }
                manifest = {
                    "step": step,
                    "run_id": uuid4().hex,
                    "signature": signature,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": round(time.monotonic() - start, 3),
                    "settings": self.settings(step),
                    "parents": {name: data["run_id"] for name, data in parents.items()},
                    "artifacts": artifacts,
                }
                write_json(temporary / "manifest.json", manifest)
                # Die bisherige Ausgabe bleibt als Sicherung erhalten. Scheitert
                # die Übernahme des neuen Ordners, wird die Sicherung zurückgesetzt.
                backup = None
                if destination.exists():
                    backup = destination.parent / f".{destination.name}.previous-{uuid4().hex[:8]}"
                    destination.replace(backup)
                try:
                    temporary.replace(destination)
                except BaseException:
                    if backup is not None:
                        backup.replace(destination)
                    raise
                write_json(attempt, {"status": "complete", "run_id": manifest["run_id"]})
            except BaseException as exc:
                write_json(
                    attempt,
                    {
                        "status": "failed",
                        "error": str(exc) or type(exc).__name__,
                        "duration_seconds": round(time.monotonic() - start, 3),
                    },
                )
                shutil.rmtree(temporary, ignore_errors=True)
                raise
