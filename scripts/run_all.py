"""Führt alle CLI-Phasen mit derselben YAML-Konfiguration nacheinander aus."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from accessibility.stages import STEPS


def run_all(config: Path, max_memory: str = "70%", area_label: str | None = None) -> None:
    """Startet jede Phase separat und beendet den Ablauf beim ersten fehlgeschlagenen Aufruf."""
    config = config.expanduser().resolve()
    for step in (*STEPS, "evaluate"):
        # Jede Phase nutzt denselben Python-Interpreter und damit dieselbe
        # Umgebung. Separate Prozesse geben insbesondere den JVM-Speicher frei.
        command = [sys.executable, "-m", "accessibility", step, "--config", str(config)]
        if step == "route":
            command.extend(["--max-memory", max_memory])
        if step == "evaluate" and area_label is not None:
            command.extend(["--area-label", area_label])
        print(f"Starte {step} …", flush=True)
        # check=True bricht beim ersten Fehler ab, bevor eine Folgephase
        # mit fehlenden oder unvollständigen Eingaben gestartet werden könnte.
        subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> None:
    """Liest die Skriptoptionen und gibt Abbrüche als Prozessfehler an die Shell weiter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML-Konfiguration")
    parser.add_argument(
        "--max-memory", default="70%", help="R5/JVM-Heap, Standard: 70%% des gesamten RAM"
    )
    parser.add_argument("--area-label", help="Anzeigename des Gebiets in der erweiterten Karte")
    args = parser.parse_args(argv)
    try:
        run_all(args.config, args.max_memory, args.area_label)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode if exc.returncode > 0 else 1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
