"""Festgelegte Exponentialfunktion und gewichtete Kategorieaggregation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def score_from_minutes(travel_time_minutes: float | None) -> float:
    """Wendet die stückweise exponentielle Scorefunktion auf eine Reisezeit an."""
    # Unbekannte Reisezeiten behalten NaN als Score und bleiben dadurch
    # von bekannten Reisezeiten mit einem numerischen Nullscore unterscheidbar.
    if travel_time_minutes is None or pd.isna(travel_time_minutes):
        return float("nan")
    t = float(travel_time_minutes)
    if t < 0:
        raise DataValidationError("Reisezeiten dürfen nicht negativ sein.")
    # Die Randwerte gehören zur Modellfestlegung: 10 Minuten gehen in
    # die Exponentialfunktion ein, 60 Minuten sind noch eingeschlossen.
    if t < 10:
        return 100.0
    if t <= 60:
        return 146.11 * math.exp(-0.039 * t)
    return 0.0


def score_cells(
    times: pd.DataFrame, weights: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Berechnet Kategorie- und Gesamtscores mit festen Gewichten und erhält unbekannte Werte."""
    if not math.isclose(sum(weights.values()), 100.0, abs_tol=1e-8):
        raise DataValidationError("Die Kategoriegewichte müssen zusammen 100 Prozent ergeben.")
    detail = times.copy()
    detail["category_score"] = detail.travel_time_minutes.map(score_from_minutes)
    unknown = ~detail.status.isin(["ok", "no_source_within_limits"])
    detail.loc[unknown, "category_score"] = np.nan
    if detail.duplicated(["grid_id", "category_id"]).any():
        raise DataValidationError("Doppelte Kategorie pro Rasterzelle.")
    # Eine Spalte je konfigurierter Kategorie macht auch fehlende
    # Kategorieergebnisse sichtbar und richtet die Gewichte daran aus.
    scores = detail.pivot(index="grid_id", columns="category_id", values="category_score").reindex(
        columns=weights
    )
    if set(detail.category_id) - set(weights):
        raise DataValidationError("Unbekannte Kategorie in Reisezeiten.")
    weighted = scores.mul(pd.Series(weights) / 100)
    summary = scores.rename(columns=lambda name: f"score_{name}")
    # Ein Gesamtscore verlangt alle Kategorien. Fehlende Werte führen
    # weder zu einer Teilsumme noch zur Umverteilung ihrer Gewichte.
    summary["overall_score"] = weighted.sum(axis=1, min_count=len(weights))
    summary["status"] = np.where(summary.overall_score.notna(), "complete", "incomplete")
    return detail, summary.reset_index()
