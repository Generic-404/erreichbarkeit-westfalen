"""Einzeln aufrufbare Berechnungsphasen mit lokalen fachlichen Eingabedaten."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .exceptions import AccessibilityError
from .pipeline import run_step
from .stages import STEPS, StageStore


def _parser() -> argparse.ArgumentParser:
    """Definiert die einzeln aufrufbaren Phasen und ihre Kommandozeilenoptionen."""
    parser = argparse.ArgumentParser(
        prog="accessibility",
        description="ÖPNV-Erreichbarkeitsanalyse mit lokalen OSM- und GTFS-Daten",
    )
    parser.add_argument("--verbose", action="store_true", help="ausführliches Log ausgeben")
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = (
        "Daten und Zensusraster vorbereiten",
        "Haltestellen–POI-Matrix berechnen",
        "Ursprüngliche Quellen bilden",
        "Reisezeiten auf Raster übertragen",
        "Exponentialscore und Gewichte anwenden",
        "Ergebnisse exportieren",
        "Prozessstatus prüfen",
    )
    for name, description in zip((*STEPS, "status"), descriptions, strict=True):
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", required=True, type=Path, help="YAML-Konfiguration")
        command.add_argument(
            "--output-directory", type=Path, help="überschreibt paths.output_directory"
        )
        if name == "route":
            command.add_argument(
                "--max-memory", default="70%", help="R5/JVM-Heap, Standard: 70%% des gesamten RAM"
            )
    evaluation = commands.add_parser(
        "evaluate", help="Terzilklassen und Untersuchungsprioritäten aus fertigem Export"
    )
    evaluation.add_argument("--config", required=True, type=Path)
    evaluation.add_argument("--output-directory", type=Path, help="Ordner des bestehenden Exports")
    evaluation.add_argument(
        "--reference-file",
        type=Path,
        help="Gespeicherte thresholds.json für vergleichbare Klassen verwenden",
    )
    evaluation.add_argument(
        "--area-label", help="Name des ausgewerteten Gebiets; standardmäßig Name der Grenzdatei"
    )
    map_only = commands.add_parser(
        "evaluate-map", help="Nur Auswertungskarte aus vorhandenen Ergebnissen neu erstellen"
    )
    map_only.add_argument("--config", required=True, type=Path)
    map_only.add_argument("--output-directory", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Lädt die YAML-Konfiguration und startet den gewählten Befehl."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config, validate_input_files=False)
        paths = replace(
            config.paths, output_directory=args.output_directory or config.paths.output_directory
        )
        config = replace(config, paths=paths)
        # Karten- und Klassifikationsbefehle lesen fertige Ergebnisse.
        # Sie starten die vorgelagerten Berechnungsphasen nicht erneut.
        if args.command == "evaluate-map":
            from .evaluation_map import rebuild_evaluation_map

            print(f"Karte aktualisiert: {rebuild_evaluation_map(config)}")
            return
        if args.command == "evaluate":
            from .evaluation import run_evaluation

            result = run_evaluation(config, args.reference_file, args.area_label)
            print(f"Auswertung abgeschlossen: {result}")
            return
        # status liest nur den Bearbeitungsstand und nennt die erste Phase,
        # deren Ergebnis fehlt, unvollständig oder veraltet ist.
        if args.command == "status":
            states = StageStore(config).inspect()
            for state in states:
                print(f"{state['step']:10} {state['status']:12} {state['reason']}")
            next_step = next((s["step"] for s in states if s["status"] != "complete"), None)
            print(f"Nächster Schritt: {next_step}" if next_step else "Alle Schritte sind aktuell.")
            return
        previous_argv = sys.argv[:]
        try:
            with warnings.catch_warnings():
                if args.command == "route":
                    # R5py liest seinen Heap-Parameter beim ersten Import aus sys.argv.
                    sys.argv = [*sys.argv, "--max-memory", args.max_memory]
                    warnings.filterwarnings(
                        "ignore",
                        message=r"^The provided .* minutes\.",
                        category=RuntimeWarning,
                        module=r"^r5py\.r5\.regional_task$",
                    )
                result = run_step(config, args.command)
        finally:
            sys.argv = previous_argv
        print(f"Schritt {args.command} abgeschlossen: {result}")
        position = STEPS.index(args.command)
        if position + 1 < len(STEPS):
            print(
                f"Nächster Schritt separat: python -m accessibility {STEPS[position + 1]} --config '{args.config}'"
            )
    except KeyboardInterrupt:
        logging.getLogger("accessibility").info(
            "Verarbeitung abgebrochen. Abgeschlossene Caches und Routingblöcke bleiben erhalten."
        )
        raise SystemExit(130) from None
    except AccessibilityError as exc:
        logging.getLogger("accessibility").error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main(sys.argv[1:])
