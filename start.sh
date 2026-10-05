#!/usr/bin/env bash
# Richtet die lokale Python-Umgebung ein und startet den Westfalen-Lauf.
set -euo pipefail

max_memory="70%"
while (($#)); do
    case "$1" in
        --max-memory)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo 'Fehler: Hinter --max-memory fehlt ein Wert, z. B. "70%" oder "40G".' >&2
                exit 2
            fi
            max_memory="$2"
            shift 2
            ;;
        -h|--help)
            echo 'Aufruf: bash start.sh [--max-memory "70%"]'
            echo 'Prüft die Westfalen-Eingangsdaten, installiert Abhängigkeiten und startet alle Phasen.'
            echo 'Voraussetzungen: Linux, Python ab 3.11 mit venv, Java 21 und ausreichend RAM.'
            exit 0
            ;;
        *)
            echo "Unbekannte Option: $1. Hilfe: bash start.sh --help" >&2
            exit 2
            ;;
    esac
done

# Alle Projektpfade beziehen sich auf dieses Skript, auch bei einem Aufruf von außen.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"

if ! command -v java >/dev/null 2>&1; then
    echo 'OpenJDK 21 ist erforderlich; siehe README.' >&2
    exit 1
fi
java_version="$(java -version 2>&1)"
if [[ ! "$java_version" =~ version\ \"21[.\"] ]]; then
    echo 'Java 21 ist erforderlich. Erkannte Java-Version:' >&2
    echo "$java_version" >&2
    exit 1
fi

# Python 3.12 wird bevorzugt; andernfalls wird python3 aufgerufen.
if command -v python3.12 >/dev/null 2>&1; then
    python_command=python3.12
elif command -v python3 >/dev/null 2>&1; then
    python_command=python3
else
    echo 'Python einschließlich venv ist erforderlich; siehe README.' >&2
    exit 1
fi
if ! "$python_command" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo 'Python ab Version 3.11 ist erforderlich.' >&2
    exit 1
fi

# Die Prüfsummen kennzeichnen die für den Westfalen-Lauf vorgesehenen Eingangsdaten.
if ! command -v sha256sum >/dev/null 2>&1; then
    echo 'sha256sum aus dem Paket coreutils ist erforderlich.' >&2
    exit 1
fi
echo 'Prüfe die Eingangsdaten …'
if ! sha256sum --check data/checksums.sha256; then
    echo 'Eingangsdaten fehlen oder stimmen nicht mit den Prüfsummen überein; siehe README.' >&2
    exit 1
fi

echo "Java-Heap-Grenze: $max_memory. Python und Betriebssystem benötigen zusätzlich RAM."
echo 'Bei unzureichendem Arbeitsspeicher kann die Verarbeitung abbrechen; siehe README.'

# Eine virtuelle Umgebung hält die Projektabhängigkeiten vom System-Python getrennt.
if [[ ! -d .venv ]]; then
    if ! "$python_command" -m venv .venv; then
        echo 'Die Python-Umgebung konnte nicht erstellt werden. Das passende venv-Paket ist erforderlich.' >&2
        exit 1
    fi
fi
if [[ ! -x .venv/bin/python ]]; then
    echo 'Die vorhandene .venv enthält keinen ausführbaren Python-Interpreter.' >&2
    exit 1
fi
.venv/bin/python -m pip install -c constraints.txt -e .

# Das Gesamtskript führt die sieben Berechnungsphasen nacheinander aus.
exec .venv/bin/python scripts/run_all.py \
    --config config/westfalen.yaml \
    --area-label 'Region Westfalen' \
    --max-memory "$max_memory"
