"""Terzilklassen und Untersuchungsprioritäten aus fertigen Exporten, ohne Routing."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd

from .exceptions import DataValidationError
from .stages import StageStore, identity, read_json, write_json

LOGGER = logging.getLogger(__name__)
GROUPS = {
    "overall": ("Gesamtbevölkerung", "population_total", "overall_score"),
    "education": ("Bildung / unter 18", "population_under18", "score_education"),
    "health": ("Gesundheit / ab 65", "population_65plus", "score_health"),
}
AREA_CATEGORIES = {"education": ("schools", "kindergartens"), "health": ("pharmacies", "hospitals")}
SOURCE = "https://pro.arcgis.com/en/pro-app/3.0/help/mapping/layer-properties/data-classification-methods.htm"
METHOD = "linear_quantile_tertiles_common_residential_score_reference_v2"
FORMAT_VERSION = 3
POPULATION_DECIMALS = 10
PRIORITY_LABELS = {
    0: "Keine Personen der Bezugsgruppe",
    1: "Keine erhöhte Priorität",
    2: "Erhöhte Untersuchungspriorität",
    3: "Hohe Untersuchungspriorität",
}


def census_number(values: pd.Series, maximum: float | None = None) -> pd.Series:
    """Zensus-Zeichenerklärung: – = genau Null oder auf Null geändert; / und . unbekannt."""
    text = values.astype("string").str.strip().replace({"–": "0"})
    text = text.str.replace(",", ".", regex=False)
    numbers = pd.to_numeric(text, errors="coerce").astype(float)
    valid = np.isfinite(numbers) & numbers.ge(0)
    if maximum is not None:
        valid &= numbers.le(maximum)
    return numbers.where(valid)


def census_column(frame: pd.DataFrame, suffix: str) -> str | None:
    """Findet ein eindeutiges Zensusmerkmal anhand seines Spaltenendes."""
    matches = [c for c in frame if c.startswith("zensus_") and c.endswith("_" + suffix)]
    if len(matches) > 1:
        raise DataValidationError(f"Mehrdeutiges Zensusmerkmal {suffix}: {matches}")
    return matches[0] if matches else None


def tertile_limits(values) -> list[float] | None:
    """33⅓-/66⅔-%-Quantile mit linearer Interpolation (NumPy/Pandas-Standard)."""
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if not len(values):
        return None
    return np.quantile(values, [1 / 3, 2 / 3], method="linear").tolist()


def classify(values, limits):
    """1: x < q1, 2: q1 <= x < q2, 3: x >= q2; gleiche Werte ungeteilt."""
    result = pd.Series(np.nan, index=values.index)
    if limits is not None:
        valid = values.notna()
        result.loc[valid] = (
            1 + values[valid].ge(limits[0]).astype(int) + values[valid].ge(limits[1]).astype(int)
        )
    return result


def _number(frame, suffix, maximum=None):
    """Liest ein numerisches Zensusmerkmal; fehlende Spalten ergeben unbekannte Werte."""
    column = census_column(frame, suffix)
    return (
        census_number(frame[column], maximum) if column else pd.Series(np.nan, index=frame.index)
    ), column


def _score(values):
    """Übernimmt ausschließlich endliche Scores im Bereich von 0 bis 100."""
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    return numbers.where(np.isfinite(numbers) & numbers.between(0, 100))


def evaluate_cells(grid: pd.DataFrame, weights: dict[str, float], references: dict | None = None):
    """Berechnet Klassen und Prioritäten aus Scores und Bevölkerung; fehlende Werte bleiben NaN."""
    if grid.grid_id.isna().any() or grid.grid_id.duplicated().any():
        raise DataValidationError("Auswertung benötigt eindeutige, nicht leere Rasterkennungen.")
    result = pd.DataFrame({"grid_id": grid.grid_id})
    population, pop_column = _number(grid, "Einwohner")
    population = population.round(POPULATION_DECIMALS)
    result["population_total"] = population
    result["overall_score"] = _score(grid.overall_score)
    definitions = {"overall": {k: float(v) for k, v in weights.items()}}
    census_columns = {"population": pop_column}
    for group, suffix, field in (
        ("education", "AnteilUnter18", "population_under18"),
        ("health", "AnteilUeber65", "population_65plus"),
    ):
        share, column = _number(grid, suffix, maximum=100)
        census_columns[group] = column
        # Gleiche Dezimalprodukte müssen an einer Klassengrenze zusammenbleiben.
        # Keine Rundung auf ganze Personen; nur binäre Gleitkommaartefakte entfernen.
        result[field] = (population * share / 100).round(POPULATION_DECIMALS)
        # Zensuskennzeichen für statistisch unsichere Altersanteile in der Karte erhalten.
        flags = column.rsplit("_", 1)[0] + "_werterlaeuternde_Zeichen" if column else ""
        result[field + "_uncertain"] = (
            grid[flags].fillna("").astype(str).str.contains("KLAMMERN", regex=False)
            if flags in grid
            else False
        )
        categories = AREA_CATEGORIES[group]
        category_weights = {k: float(weights[k]) for k in categories if k in weights}
        total = sum(category_weights.values())
        if (
            len(category_weights) != 2
            or total <= 0
            or any(not np.isfinite(v) or v < 0 for v in category_weights.values())
        ):
            raise DataValidationError(f"Für {group} fehlen die beiden gültigen Kategoriegewichte.")
        definitions[group] = {k: v / total for k, v in category_weights.items()}
        if any(f"score_{k}" not in grid for k in categories):
            raise DataValidationError(f"Für {group} fehlen Kategoriescores im Export.")
        # Der Bereichsscore verwendet das relative Gewicht seiner beiden
        # Kategorien. Beide Werte müssen vorliegen; ihr Verhältnis bleibt erhalten.
        components = pd.concat(
            [_score(grid[f"score_{k}"]) * category_weights[k] for k in categories], axis=1
        )
        result[f"score_{group}"] = (components.sum(axis=1, min_count=2) / total).round(10)

    # Feste Referenzgrenzen ermöglichen vergleichbare Kartenklassen.
    # Methode und Scoredefinitionen müssen dazu mit dem Export übereinstimmen.
    if references is not None:
        if (
            references.get("format_version") != FORMAT_VERSION
            or references.get("method") != METHOD
            or references.get("score_definitions") != definitions
        ):
            raise DataValidationError(
                "Grenzwertdatei passt nicht zur Methode oder Scoredefinition dieses Exports. "
                "Alte Grenzwertdateien im Bezugsgebiet mit evaluate neu erzeugen."
            )
        if set(references.get("thresholds", {})) != set(GROUPS):
            raise DataValidationError("Grenzwertdatei muss alle drei Auswertungen enthalten.")
        if not references.get("reference_area") or not references.get("reference_export_run_id"):
            raise DataValidationError("Grenzwertdatei benötigt Bezugsgebiet und Ursprungslauf.")

    report = {
        "format_version": FORMAT_VERSION,
        "method": METHOD,
        "reference_mode": "fixed" if references is not None else "computed",
        "score_definitions": definitions,
        "thresholds": {},
        "groups": {},
        "census_columns": census_columns,
        "selection": "Scoregrenzen je Score: alle Zellen mit positiver Gesamtbevölkerung und gültigem Score, unabhängig von Altersangaben. Bevölkerungsterzile: positive jeweilige Bezugsbevölkerung mit gültigem Score. Jede Zelle zählt gleich.",
        "numeric_normalization": "Personenzahlen und ihre Quantilgrenzen auf zehn Nachkommastellen; keine Rundung auf ganze Personen.",
        "class_rule": "1: x < q1; 2: q1 <= x < q2; 3: x >= q2. Identische Werte bleiben zusammen; Klassen können leer sein.",
        "priority_rule": "3: score_class=1 and population_class=3; 2: (1,2) or (2,3); 1: all other valid combinations; 0: known zero population",
        "source": SOURCE,
        "interpretation": "Relative Klassen im Bezugsgebiet. Die Prioritätsmatrix verknüpft Scoreklasse und Bevölkerungsklasse.",
        "zero_symbol": "Zensus – = genau Null oder auf Null geändert; leere/ungültige Werte bleiben unbekannt",
        "total_cells": len(result),
    }
    for group, (label, population_field, score_field) in GROUPS.items():
        population_values, score_values = result[population_field], result[score_field]
        # Scoreterzile beziehen sich auf bewohnte Zellen mit gültigem Score.
        # Für Bevölkerungsterzile muss zusätzlich die jeweilige Alters- oder
        # Gesamtbevölkerungsgruppe in der Zelle positiv sein.
        score_reference = population.gt(0) & score_values.notna()
        valid = score_reference & population_values.gt(0)
        limits = (
            references["thresholds"][group]
            if references is not None
            else {
                "score": tertile_limits(score_values[score_reference]),
                "population": tertile_limits(population_values[valid]),
            }
        )
        if not isinstance(limits, dict) or set(limits) != {"score", "population"}:
            raise DataValidationError(f"Ungültige Grenzwerte für {group}.")
        for key, bounds in limits.items():
            if bounds is not None and (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(
                    isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v)
                    for v in bounds
                )
                or not 0 <= bounds[0] <= bounds[1]
                or (key == "score" and bounds[1] > 100)
            ):
                raise DataValidationError(f"Ungültige {key}-Grenzwerte für {group}.")
        # Auch interpolierte oder eingelesene Personengrenzen vereinheitlichen.
        # Die übergebene Referenzdatei wird dabei nicht verändert.
        limits = {
            **limits,
            "population": (
                np.round(limits["population"], POPULATION_DECIMALS).tolist()
                if limits["population"] is not None
                else None
            ),
        }
        report["thresholds"][group] = limits
        score_classes = classify(score_values.where(score_reference), limits["score"])
        population_classes = classify(population_values.where(valid), limits["population"])
        usable = valid & score_classes.notna() & population_classes.notna()
        # Die Priorität verknüpft Score- und Bevölkerungsklasse. Ein niedriger
        # Score mit hoher Personenzahl erhält die höchste Priorität; bekannte
        # Nullbevölkerung bekommt einen eigenen Code, fehlende Angaben bleiben NaN.
        priority = pd.Series(np.nan, index=result.index)
        priority.loc[usable] = 1
        priority.loc[
            usable
            & (
                (score_classes.eq(1) & population_classes.eq(2))
                | (score_classes.eq(2) & population_classes.eq(3))
            )
        ] = 2
        priority.loc[usable & score_classes.eq(1) & population_classes.eq(3)] = 3
        priority.loc[population_values.eq(0)] = 0
        # Die Zuweisungsreihenfolge legt den angezeigten Datenstatus fest,
        # wenn mehrere Bedingungen für dieselbe Zelle zutreffen.
        status = pd.Series("ok", index=result.index)
        status.loc[valid & ~usable] = "thresholds_unavailable"
        status.loc[score_values.isna()] = "unknown_score"
        status.loc[population_values.eq(0)] = "no_population"
        status.loc[population_values.isna()] = "unknown_population"
        result[f"score_class_{group}"] = score_classes
        result[f"population_class_{group}"] = population_classes
        result[f"priority_{group}"] = priority
        result[f"evaluation_{group}_status"] = status
        # Prozentangaben beziehen sich auf die auswertbare Bevölkerung,
        # unbekannte Zellwerte gehen nicht in den Nenner ein.
        denominator = float(population_values[usable].sum())
        uncertain = (
            result.get(population_field + "_uncertain", pd.Series(False, index=result.index))
            & valid
        )
        distribution = []
        for a in (1, 2, 3):
            for b in (1, 2, 3):
                selected = usable & score_classes.eq(a) & population_classes.eq(b)
                distribution.append(
                    {
                        "score_class": a,
                        "population_class": b,
                        "cells": int(selected.sum()),
                        "population": float(population_values[selected].sum()),
                    }
                )
        summaries = {}
        for code, name in PRIORITY_LABELS.items():
            selected = priority.eq(code)
            total = float(population_values[selected].sum())
            summaries[str(code)] = {
                "label": name,
                "cells": int(selected.sum()),
                "population": total,
                "population_share_pct": 100 * total / denominator if denominator else None,
            }
        report["groups"][group] = {
            "label": label,
            "valid_populated_cells": int(usable.sum()),
            "score_reference_cells": int(score_reference.sum()),
            "unknown_population_cells": int(population_values.isna().sum()),
            "unknown_score_populated_cells": int(
                (population_values.gt(0) & score_values.isna()).sum()
            ),
            "known_population": float(population_values.sum()),
            "evaluated_population": denominator,
            "uncertain_age_cells": int(uncertain.sum()),
            "uncertain_age_population_estimate": float(population_values[uncertain].sum()),
            "score_min": float(score_values[valid].min()) if valid.any() else None,
            "score_max": float(score_values[valid].max()) if valid.any() else None,
            "collapsed_score_limits": limits["score"] is not None
            and limits["score"][0] == limits["score"][1],
            "collapsed_population_limits": limits["population"] is not None
            and limits["population"][0] == limits["population"][1],
            "matrix": distribution,
            "priority_classes": summaries,
            "status_counts": {str(k): int(v) for k, v in status.value_counts().items()},
        }
    return result, report


def _validate_export(folder):
    """Prüft das Exportmanifest und die für die Klassifikation benötigten Ergebnisdateien."""
    manifest = read_json(folder / "manifest.json")
    if manifest.get("step") != "export":
        raise DataValidationError("evaluate benötigt einen abgeschlossenen Export.")
    for name in ("accessibility.gpkg", "run_metadata.json"):
        expected = manifest.get("artifacts", {}).get(name)
        actual = identity(folder / name)
        if expected is None or any(actual.get(k) != expected.get(k) for k in ("size", "mtime_ns")):
            raise DataValidationError(f"Exportdatei fehlt oder wurde verändert: {name}")
    return manifest


def run_evaluation(
    config, reference_file: Path | None = None, reference_area: str | None = None
) -> Path:
    """Schreibt die Klassifikation atomar unter export/evaluation und erhält die Exportmanifeste."""
    from .evaluation_map import write_evaluation_map

    store = StageStore(config)
    folder = store.output
    if not (folder / "manifest.json").exists():
        raise DataValidationError(
            f"Kein abgeschlossener Export in {folder}. evaluate startet keine Vorgänger."
        )
    store.root.mkdir(parents=True, exist_ok=True)
    with (store.root / "process.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DataValidationError(
                "Für diesen Ergebnisordner läuft bereits ein Verarbeitungsschritt."
            ) from exc
        manifest = _validate_export(folder)
        metadata = read_json(folder / "run_metadata.json")
        weights = metadata["reports"]["score"]["weights_percent"]
        references = read_json(reference_file) if reference_file else None
        LOGGER.info("evaluate liest ausschließlich den fertigen Export: %s", folder)
        grid = gpd.read_file(folder / "accessibility.gpkg", layer="grid")
        detail, report = evaluate_cells(grid, weights, references)
        report.update(
            created_at=datetime.now(timezone.utc).isoformat(),
            source_export_run_id=manifest["run_id"],
            source_export=str(folder),
            input_identity=identity(folder / "accessibility.gpkg"),
            reference_origin=(str(reference_file.resolve()) if reference_file else str(folder)),
            routing=metadata["reports"].get("route"),
            original_reference=(references.get("source_export_run_id") if references else None),
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        area = reference_area or Path(metadata["config"]["paths"]["boundary"]).stem
        report["analysis_area"] = area
        report["reference_area"] = references["reference_area"] if references else area
        report["reference_export_run_id"] = (
            references["reference_export_run_id"] if references else manifest["run_id"]
        )
        for group, values in report["groups"].items():
            LOGGER.info(
                "%s: Grenzen %s; %d relevante Zellen; %d mit hoher Priorität",
                group,
                report["thresholds"][group],
                values["valid_populated_cells"],
                values["priority_classes"]["3"]["cells"],
            )
        output = folder / "evaluation"
        with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=folder) as temporary:
            destination = Path(temporary) / "result"
            destination.mkdir()
            # Alle vorhandenen Scores/Zensusattribute für die Karte erhalten.
            enriched = grid.drop(columns=[c for c in detail if c != "grid_id" and c in grid]).merge(
                detail, on="grid_id", validate="one_to_one"
            )
            detail.to_csv(destination / "evaluation_cells.csv", index=False)
            pd.DataFrame.from_dict(report["groups"], orient="index").drop(
                columns=["status_counts", "matrix", "priority_classes"]
            ).rename_axis("group").to_csv(destination / "evaluation_summary.csv")
            write_json(
                destination / "thresholds.json",
                {
                    k: report[k]
                    for k in (
                        "format_version",
                        "method",
                        "thresholds",
                        "score_definitions",
                        "reference_area",
                        "reference_export_run_id",
                        "source_export_run_id",
                        "source_export",
                        "created_at",
                    )
                },
            )
            write_json(destination / "evaluation_report.json", report)
            LOGGER.info(
                "Schreibe GeoPackage und Klassentabellen für %d Rasterzellen.", len(enriched)
            )
            enriched.to_file(destination / "evaluation.gpkg", layer="grid", driver="GPKG")
            for group in GROUPS:
                detail.loc[detail[f"priority_{group}"].ge(2)].to_csv(
                    destination / f"investigation_cells_{group}.csv", index=False
                )
            rows = [
                {"group": group, "priority": code, **values}
                for group, data in report["groups"].items()
                for code, values in data["priority_classes"].items()
            ]
            pd.DataFrame(rows).to_csv(destination / "priority_summary.csv", index=False)
            pd.DataFrame(
                [
                    {"group": group, **row}
                    for group, data in report["groups"].items()
                    for row in data["matrix"]
                ]
            ).to_csv(destination / "class_matrix.csv", index=False)
            LOGGER.info("Erzeuge Auswertungskarte mit dynamischem 100-m-Raster.")
            stops = gpd.read_file(folder / "accessibility.gpkg", layer="stops")
            pois = gpd.read_file(folder / "accessibility.gpkg", layer="pois")
            names = {c["id"]: c["name"] for c in metadata["config"]["poi_categories"]}
            map_report = write_evaluation_map(
                destination / "accessibility_map.html", enriched, stops, pois, names, report
            )
            write_json(destination / "map_report.json", map_report)
            (destination / "AUSWERTUNG.md").write_text(_markdown_report(report), encoding="utf-8")
            if _validate_export(folder)["run_id"] != manifest["run_id"]:
                raise DataValidationError(
                    "Der zugrunde liegende Export wurde während der Auswertung ersetzt."
                )
            backup = folder / f".evaluation.previous-{uuid4().hex[:8]}" if output.exists() else None
            if backup:
                output.replace(backup)
            try:
                destination.replace(output)
            except BaseException:
                if backup:
                    backup.replace(output)
                raise
        return output


def _markdown_report(report):
    """Formatiert Klassengrenzen, Prioritäten und Kennzahlen als Markdown-Bericht."""
    normalization = "Bereichsscores werden zur Vermeidung von Gleitkommaartefakten auf zehn Nachkommastellen stabilisiert."
    if report["method"] == METHOD:
        normalization += " Personenzahlen und Personengrenzen werden ebenso vereinheitlicht."
    lines = [
        "# Erreichbarkeit und Bevölkerung: Terzilklassen",
        "",
        f"Auswertungsgebiet: {report['analysis_area']}. Bezugsgebiet der Grenzen: {report['reference_area']}.",
        "Berechnet ausschließlich aus dem fertigen Export, ohne neues Routing.",
        "",
        report["selection"],
        "Jede Zelle zählt gleich. Grenzen: 33⅓-/66⅔-%-Quantile mit linearer Interpolation (NumPy/Pandas-Standard).",
        "Niedrig: x < q1; mittel: q1 <= x < q2; hoch: x >= q2. Gleiche Werte bleiben zusammen; Klassen können leer bleiben.",
        "",
        "| Auswertung | Scoregrenzen | Personengrenzen je 100 m |",
        "|---|---|---|",
    ]
    for group, values in report["groups"].items():
        limits = report["thresholds"][group]

        def shown(bounds):
            """Formatiert ein Paar Klassengrenzen oder kennzeichnet fehlende Grenzen."""
            return "nicht bestimmbar" if bounds is None else " / ".join(f"{v:.4f}" for v in bounds)

        lines.append(
            f"| {values['label']} | {shown(limits['score'])} | {shown(limits['population'])} |"
        )
    lines.extend(
        [
            "",
            "## Verknüpfung",
            "",
            "Hohe Untersuchungspriorität: niedrige Erreichbarkeit und hohe Personenzahl.",
            "Erhöhte Untersuchungspriorität: niedrige Erreichbarkeit und mittlere Personenzahl ODER mittlere Erreichbarkeit und hohe Personenzahl.",
            "Alle übrigen gültigen Kombinationen erhalten den Code 1: keine erhöhte Priorität.",
            "Bekannte Nullbevölkerung wird gesondert markiert; fehlende Daten bleiben unbekannt.",
            "",
            "## Klassifikation und Darstellung",
            "",
            "Die Klassen werden aus der Verteilung der gültigen Werte im angegebenen Bezugsgebiet gebildet.",
            "Die Quantilklassifikation erzeugt drei Klassen. Die Prioritätsmatrix verknüpft die resultierenden Score- und Bevölkerungsklassen.",
            "Gleiche Werte erhalten dieselbe Klasse. Zusammenfallende Klassengrenzen werden im Bericht gekennzeichnet und beibehalten.",
            "Grenzwerte in dieser Tabelle sind für die Darstellung gerundet; thresholds.json enthält die zur Klassifikation verwendeten Werte. "
            + normalization
            + " Gesamt- und Kategoriescores bleiben unverändert. Die Grenzen gelten für das angegebene Bezugsgebiet. Für Westfalen evaluate auf dessen vollständigem Export ohne --reference-file ausführen.",
            "Für vergleichbare Teilgebiete oder Szenarien dieselbe thresholds.json mit --reference-file verwenden. Bezugsgebiet und Ursprungslauf bleiben erhalten; Zoom und Kartenausschnitt ändern keine Grenzen.",
            "Altersgruppenzahlen werden aus Einwohnerzahl × Altersanteil / 100 rechnerisch ermittelt und nicht auf ganze Personen gerundet. KLAMMERN markiert statistisch unsichere Anteile; diese bleiben enthalten und werden separat ausgewiesen.",
            "Zensus – wird als Null eingelesen; leere oder ungültige Werte werden als unbekannt verarbeitet.",
            "Die 1-km-Übersicht zeigt arithmetische Mittel vorhandener 100-m-Werte, auch der Klassen- und Prioritätscodes. Fehlende Werte werden je Merkmal ausgelassen; bekannte Nullwerte zählen mit. Dies ist eine kartographische Zusammenfassung, keine neue Klassifikation. Die Häufigkeiten je Prioritätsklasse bleiben im Popup sichtbar.",
            "Personenzahlen werden in der Übersicht als Mittel je 100-m-Zelle mit bekannter Angabe dargestellt. 100 m und 1 km sind unabhängig vom Zoom wählbar. KLAMMERN wird in beiden Auflösungen angezeigt. Klassen- und Prioritätscodes werden für die Übersicht numerisch gemittelt.",
            "Die Auswertung verwendet die exportierten Reisezeitscores und die Zensuszahlen der jeweiligen Bevölkerungsgruppe.",
            "",
            "## Quelle",
            "",
            SOURCE,
            "",
            "Dateien: thresholds.json (wiederverwendbare Grenzen); evaluation_report.json (Provenienz, Datenabdeckung); evaluation_cells.csv; priority_summary.csv; class_matrix.csv; investigation_cells_*.csv.",
            "",
        ]
    )
    return "\n".join(lines)
