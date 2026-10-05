import copy
import json
import re
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from accessibility.evaluation import (
    GROUPS,
    classify,
    evaluate_cells,
    run_evaluation,
    tertile_limits,
)
from accessibility.evaluation_map import evaluation_overview, write_evaluation_map
from accessibility.exceptions import DataValidationError
from accessibility.stages import identity, write_json

WEIGHTS = {"schools": 10, "kindergartens": 5, "pharmacies": 12.5, "hospitals": 12.5}


def frame(scores=None, populations=None):
    """Erzeugt ein kleines Raster mit Scores und Zensusmerkmalen für die Klassifikation."""
    scores = [10, 20, 30, 40, 50, 60, 70, 80, 90] if scores is None else scores
    n = len(scores)
    return gpd.GeoDataFrame(
        {
            "grid_id": [str(i) for i in range(n)],
            "overall_score": scores,
            **{f"score_{c}": scores for c in WEIGHTS},
            "zensus_4_Einwohner": populations
            if populations is not None
            else list(range(90, 0, -10)),
            "zensus_8_AnteilUnter18": ["20,0"] * n,
            "zensus_7_AnteilUeber65": ["30,0"] * n,
            "zensus_8_werterlaeuternde_Zeichen": ["KLAMMERN"] + [""] * (n - 1),
            "center_x": [4000050 + 100 * i for i in range(n)],
            "center_y": [3200050] * n,
            "status": ["complete"] * n,
        },
        geometry=[box(4000000 + 100 * i, 3200000, 4000100 + 100 * i, 3200100) for i in range(n)],
        crs="EPSG:3035",
    )


def reference(report):
    """Übernimmt Klassengrenzen und Methodenangaben als feste Testreferenz."""
    return {**report, "reference_area": "Teilgebiet", "reference_export_run_id": "original"}


def test_empirical_tertiles_and_ties_are_not_split():
    """Prüft Terzilgrenzen und die gemeinsame Einordnung gleicher Werte."""
    assert tertile_limits([9, 3, 2, 1, 8, 7, 6, 5, 4]) == pytest.approx([11 / 3, 19 / 3])
    assert tertile_limits([1, 1, 1, 1, 2, 3]) == pytest.approx([1, 4 / 3])
    assert classify(pd.Series([1, 1, 2, 3, np.nan]), [1, 1]).tolist()[:4] == [3, 3, 3, 3]
    assert tertile_limits([]) is None


def test_classification_matrix_and_correct_age_weights():
    """Prüft Prioritäten und Gewichtung der altersbezogenen Bereichsscores."""
    grid = frame()
    cells, report = evaluate_cells(grid, WEIGHTS)
    assert report["thresholds"]["overall"]["score"] == pytest.approx([110 / 3, 190 / 3])
    assert report["thresholds"]["overall"]["population"] == pytest.approx([110 / 3, 190 / 3])
    assert cells.priority_overall.tolist() == [3, 3, 3, 1, 1, 1, 1, 1, 1]
    assert cells.population_under18.iloc[0] == 18
    assert cells.population_65plus.iloc[0] == 27
    assert cells.population_under18_uncertain.iloc[0]
    grid.loc[0, "score_kindergartens"] = 100
    cells, _ = evaluate_cells(grid, WEIGHTS)
    assert cells.score_education.iloc[0] == pytest.approx(40)
    assert sum(r["cells"] for r in report["groups"]["overall"]["matrix"]) == 9
    assert sum(
        r["population_share_pct"] for r in report["groups"]["overall"]["priority_classes"].values()
    ) == pytest.approx(100)
    assert not any("deficit" in c or "fgt" in c for c in cells)


def test_all_nine_priority_combinations_with_frozen_thresholds():
    """Prüft alle neun Kombinationen von Score- und Bevölkerungsklasse."""
    grid = frame(scores=[10] * 3 + [50] * 3 + [90] * 3, populations=[10, 50, 90] * 3)
    _, initial = evaluate_cells(frame(), WEIGHTS)
    refs = reference(initial)
    for g in GROUPS:
        refs["thresholds"][g] = {"score": [30, 60], "population": [30, 60]}
    cells, _ = evaluate_cells(grid, WEIGHTS, refs)
    assert cells.priority_overall.tolist() == [1, 2, 3, 1, 1, 2, 1, 1, 1]


def test_missing_values_zero_population_and_zero_scores():
    """Prüft die Unterscheidung unbekannter Werte und bekannter Nullwerte."""
    grid = frame(scores=[0, 30, np.nan, 90], populations=["10", None, "20", "0"])
    grid.loc[0, "zensus_8_AnteilUnter18"] = "–"
    grid.loc[2, "zensus_7_AnteilUeber65"] = "/"
    cells, report = evaluate_cells(grid, WEIGHTS)
    assert report["groups"]["overall"]["valid_populated_cells"] == 1
    assert cells.score_class_overall.iloc[0] == 3
    assert cells.evaluation_overall_status.tolist() == [
        "ok",
        "unknown_population",
        "unknown_score",
        "no_population",
    ]
    assert pd.isna(cells.priority_overall.iloc[1]) and pd.isna(cells.priority_overall.iloc[2])
    assert cells.priority_overall.iloc[3] == 0
    assert cells.priority_education.iloc[0] == 0
    assert pd.isna(cells.population_65plus.iloc[2])


def test_frozen_references_are_retained_and_incompatible_weights_rejected():
    """Prüft feste Referenzgrenzen und die Ablehnung abweichender Gewichte."""
    _, report = evaluate_cells(frame(), WEIGHTS)
    refs = reference(report)
    _, fixed = evaluate_cells(frame(scores=[95] * 9), WEIGHTS, refs)
    assert fixed["thresholds"] == report["thresholds"]
    with pytest.raises(DataValidationError):
        evaluate_cells(frame(), {**WEIGHTS, "schools": 11}, refs)
    bad = copy.deepcopy(refs)
    bad["thresholds"]["health"]["population"] = [10, 1]
    with pytest.raises(DataValidationError):
        evaluate_cells(frame(), WEIGHTS, bad)


def enriched():
    """Ergänzt das Testraster um abgeleitete Kartenklassen und Prioritäten."""
    grid = frame()
    cells, report = evaluate_cells(grid, WEIGHTS)
    grid = grid.drop(columns=["overall_score"]).merge(cells, on="grid_id")
    report.update(analysis_area="Teilgebiet", reference_area="Teilgebiet")
    return grid, report


def test_overview_means_known_values_and_preserves_quality_flags():
    """Prüft Mittelwerte bekannter Zellen und die Übernahme von Unsicherheitskennzeichen."""
    grid, _ = enriched()
    grid.loc[8, "population_total"] = np.nan
    fields = [
        "priority_overall",
        "population_total",
        "score_class_overall",
        "population_class_overall",
    ]
    displayed, size = evaluation_overview(grid, fields, max_cells=1, full_detail_limit=1)
    assert size == 1000
    assert len(displayed) == 1
    for field in fields:
        assert displayed[field].iloc[0] == pytest.approx(grid[field].mean())
    assert displayed.priority_overall.iloc[0] == pytest.approx(15 / 9)
    assert displayed.overview_overall_3.sum() == 3
    assert displayed.overview_under18_uncertain_count.iloc[0] == 1
    assert displayed.population_under18_uncertain.iloc[0]
    assert displayed.represented_cells.iloc[0] == 9
    grid["population_total"] = np.nan
    grid["priority_overall"] = [3, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    displayed, size = evaluation_overview(grid, fields)
    assert size == 1000  # Beide Rasterauflösungen stehen auch für kleine Datensätze bereit.
    assert pd.isna(displayed.population_total.iloc[0])
    assert displayed.priority_overall.iloc[0] == 1.5


def test_overview_groups_at_exact_census_kilometer_boundaries():
    """Prüft die Zuordnung zu den festen Kilometergrenzen des Zensusgitters."""
    grid, _ = enriched()
    grid.loc[8, "center_x"] = 4001050
    grid.loc[8, "geometry"] = box(4001000, 3200000, 4001100, 3200100)
    displayed, _ = evaluation_overview(grid, ["population_total", "priority_overall"])
    assert displayed.represented_cells.tolist() == [8, 1]
    assert displayed.population_total.tolist() == [55, 10]


def test_map_includes_classes_details_pois_and_valid_javascript(tmp_path):
    """Prüft Karteninhalte, Datendateien und die Syntax des erzeugten JavaScripts."""
    grid, report = enriched()
    empty = gpd.GeoDataFrame({"category_id": []}, geometry=[], crs=grid.crs)
    pois = gpd.GeoDataFrame(
        {"name": ["Schule"], "category_id": ["schools"], "category_name": ["Schulen"]},
        geometry=[Point(4000050, 3200050)],
        crs=grid.crs,
    )
    path = tmp_path / "map.html"
    result = write_evaluation_map(
        path, grid, empty, pois, {"schools": "Schulen"}, report, max_cells=1, full_detail_limit=1
    )
    assert result["detail_switch_zoom"] is None
    assert result["available_resolutions_meters"] == [1000, 100]
    assert result["resolution_selection"] == "manual"
    tiles = list((tmp_path / result["assets_directory"] / "tiles").rglob("*.json"))
    features = []
    for tile in tiles:
        payload = json.loads(tile.read_text())
        features.extend(dict(zip(payload["fields"], row[1])) for row in payload["cells"])
    assert len(features) == 9
    assert all("priority_health" in f and "score_schools" in f for f in features)
    assert "population_under18_uncertain" in features[0]
    assert result["renderer"] == "canvas"
    assert path.stat().st_size < 60_000
    html = path.read_text()
    assert "poiPane" in html and "thresholds" in html and "#081d58" in html
    assert "Hohe Untersuchungspriorität" in html
    assert "scrollbar-gutter: stable" in html and "padding-right: 24px" in html
    assert "Rasterauflösung" in html and "Zwischenfarben" in html
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    js = tmp_path / "map.js"
    js.write_text("\n".join(scripts))
    subprocess.run(["node", "--check", str(js)], check=True, capture_output=True)
    subprocess.run(
        ["node", "--check", str(tmp_path / result["assets_directory"] / "raster.js")],
        check=True,
        capture_output=True,
    )


def test_evaluate_command_uses_only_finished_export(tiny_project, monkeypatch):
    """Prüft Klassifikation und Kartenaktualisierung aus einem unveränderten fertigen Export."""
    import accessibility.cli as cli

    grid = frame()
    folder = tiny_project.paths.output_directory
    folder.mkdir()
    path = folder / "accessibility.gpkg"
    grid.to_file(path, layer="grid", driver="GPKG")
    gpd.GeoDataFrame(
        {"stop_name": ["Stop"]}, geometry=[Point(4000050, 3200050)], crs=grid.crs
    ).to_file(path, layer="stops", driver="GPKG")
    gpd.GeoDataFrame(
        {"name": ["Schule"], "category_id": ["schools"], "category_name": ["Schulen"]},
        geometry=[Point(4000050, 3200050)],
        crs=grid.crs,
    ).to_file(path, layer="pois", driver="GPKG")
    metadata = {
        "reports": {"score": {"weights_percent": WEIGHTS}},
        "config": {
            "paths": {"boundary": "Teilgebiet.gpkg"},
            "poi_categories": [{"id": c, "name": c} for c in WEIGHTS],
        },
    }
    write_json(folder / "run_metadata.json", metadata)
    artifacts = {
        name: {k: identity(folder / name)[k] for k in ("size", "mtime_ns")}
        for name in ("accessibility.gpkg", "run_metadata.json")
    }
    write_json(
        folder / "manifest.json",
        {"step": "export", "run_id": "test-export", "artifacts": artifacts},
    )
    before = {p.name: identity(p) for p in folder.iterdir()}
    monkeypatch.setattr(
        cli, "run_step", lambda *a, **kw: pytest.fail("No pipeline/routing allowed")
    )
    cli.main(["evaluate", "--config", str(tiny_project.source_path), "--area-label", "Teilgebiet"])
    assert before == {name: identity(folder / name) for name in before}
    output = folder / "evaluation"
    refs = json.loads((output / "thresholds.json").read_text())
    assert refs["reference_area"] == "Teilgebiet"
    run_evaluation(tiny_project, output / "thresholds.json", "Anderes Teilgebiet")
    report = json.loads((output / "evaluation_report.json").read_text())
    assert (
        report["analysis_area"] == "Anderes Teilgebiet" and report["reference_area"] == "Teilgebiet"
    )
    assert report["reference_export_run_id"] == "test-export"
    assert list(folder.glob(".evaluation.previous-*"))
    protected = {
        p.name: identity(p)
        for p in output.iterdir()
        if p.suffix in (".csv", ".gpkg") or p.name in ("thresholds.json", "evaluation_report.json")
    }
    monkeypatch.setattr(
        "accessibility.evaluation.evaluate_cells",
        lambda *a, **kw: pytest.fail("No classification allowed"),
    )
    cli.main(["evaluate-map", "--config", str(tiny_project.source_path)])
    assert protected == {name: identity(output / name) for name in protected}
    assert json.loads((output / "map_report.json").read_text())["renderer"] == "canvas"


def test_perfect_education_scores_stay_high_with_upper_quantile_at_maximum():
    """Prüft die höchste Scoreklasse bei einer oberen Terzilgrenze von 100."""
    grid = frame(scores=[20, 40, 60, 100, 100, 100, 100, 100, 100])
    cells, report = evaluate_cells(grid, WEIGHTS)
    assert report["thresholds"]["education"]["score"][1] == 100
    assert cells.loc[3:, "score_class_education"].eq(3).all()
    assert cells.loc[3:, "priority_education"].eq(1).all()


def test_score_reference_includes_residents_without_positive_age_population():
    """Prüft die gemeinsame Scoreverteilung aller bewohnten Zellen als Referenz."""
    grid = frame()
    _, baseline = evaluate_cells(grid, WEIGHTS)
    grid["zensus_8_AnteilUnter18"] = ["–", "/", "0", "20", "20", "20", "20", "20", "20"]
    grid["zensus_7_AnteilUeber65"] = ["30", "30", "30", "30", "30", "30", "0", "/", "–"]
    cells, report = evaluate_cells(grid, WEIGHTS)
    for group, field in [("education", "population_under18"), ("health", "population_65plus")]:
        assert report["thresholds"][group]["score"] == baseline["thresholds"][group]["score"]
        assert report["groups"][group]["score_reference_cells"] == 9
        assert report["groups"][group]["valid_populated_cells"] == 6
        assert cells["score_class_" + group].notna().all()
        target = cells[field].gt(0)
        assert report["thresholds"][group]["population"] == pytest.approx(
            tertile_limits(cells.loc[target, field])
        )
        assert cells.loc[~target, "population_class_" + group].isna().all()
        assert cells.loc[cells[field].eq(0), "priority_" + group].eq(0).all()
        assert cells.loc[cells[field].isna(), "priority_" + group].isna().all()


def test_score_reference_excludes_unknown_and_zero_residents_and_invalid_scores():
    """Prüft den Ausschluss ungeeigneter Werte aus der Score-Referenzverteilung."""
    grid = frame(scores=[0, 10, 20, 40, 60, 80, 100], populations=[None, "–", 10, 10, 10, 10, 10])
    grid.loc[6, "score_schools"] = np.nan
    grid["zensus_8_AnteilUnter18"] = "/"
    cells, report = evaluate_cells(grid, WEIGHTS)
    assert report["thresholds"]["overall"]["score"] == tertile_limits([20, 40, 60, 80, 100])
    assert report["thresholds"]["education"]["score"] == tertile_limits([20, 40, 60, 80])
    assert report["groups"]["education"]["score_reference_cells"] == 4
    assert report["thresholds"]["education"]["population"] is None
    assert cells.loc[2:5, "score_class_education"].notna().all()
    assert cells.loc[[0, 1, 6], "score_class_education"].isna().all()
    assert cells.priority_education.isna().all()


def test_decimal_age_counts_on_tertile_boundary_are_not_split():
    """Prüft die gleiche Klassifikation rechnerisch identischer dezimaler Personenzahlen."""
    grid = frame(scores=[10, 20, 30, 40, 50, 60], populations=[23, 37, 47, 8, 9, 10])
    grid["zensus_8_AnteilUnter18"] = ["17,39", "10,81", "8,51", "100", "100", "100"]
    # Alle drei Dezimalprodukte ergeben 3,9997; binäre Gleitkommawerte weichen voneinander ab.
    assert len({23 * 17.39 / 100, 37 * 10.81 / 100, 47 * 8.51 / 100}) > 1
    cells, report = evaluate_cells(grid, WEIGHTS)
    assert report["thresholds"]["education"]["population"][0] == 3.9997
    assert cells.population_under18.iloc[:3].eq(3.9997).all()
    assert cells.population_class_education.iloc[:3].eq(2).all()
    refs = reference(copy.deepcopy(report))
    refs["thresholds"]["education"]["population"][0] = 3.9997000000000003
    original = copy.deepcopy(refs)
    fixed, fixed_report = evaluate_cells(grid, WEIGHTS, refs)
    pd.testing.assert_frame_equal(fixed, cells)
    assert fixed_report["thresholds"] == report["thresholds"]
    assert refs == original


def test_old_age_specific_score_references_require_recomputation():
    """Prüft die Ablehnung von Referenzdateien mit abweichender Klassifikationsmethode."""
    _, report = evaluate_cells(frame(), WEIGHTS)
    refs = reference(report)
    refs.update(format_version=2, method="linear_quantile_tertiles_left_closed_v1")
    with pytest.raises(DataValidationError, match="neu erzeugen"):
        evaluate_cells(frame(), WEIGHTS, refs)
