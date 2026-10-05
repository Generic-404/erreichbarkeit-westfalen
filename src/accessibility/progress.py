"""Zeitmessung und regelmäßige Statusmeldungen für Verarbeitungsschritte."""

import logging
import threading
import time
from contextlib import contextmanager

LOGGER = logging.getLogger(__name__)


@contextmanager
def timed(label: str, timings: dict | None = None):
    """Protokolliert die Laufzeit und den Status eines Verarbeitungsschritts."""
    started = time.monotonic()
    stop = threading.Event()
    LOGGER.info("START %s", label)

    def heartbeat():
        """Meldet regelmäßig die verstrichene Laufzeit bis zum Abschluss des Schritts."""
        while not stop.wait(15):
            LOGGER.info("LÄUFT %s — %.0f s", label, time.monotonic() - started)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    successful = False
    try:
        yield
        successful = True
    finally:
        stop.set()
        thread.join(timeout=1)
        elapsed = round(time.monotonic() - started, 2)
        if timings is not None:
            timings[label] = elapsed
        LOGGER.info("%s %s — %.2f s", "FERTIG" if successful else "ABGEBROCHEN", label, elapsed)
