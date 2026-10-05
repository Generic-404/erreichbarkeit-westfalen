"""R5py 1.1.7 / R5 7.5.1: beste Intervallreisezeit mit mindestens einer ÖPNV-Fahrt.

Der Adapter verbindet R5-Komponenten für Zugang, RAPTOR-Fahrplansuche und Abgang.
Die Minima der Fahrplaniterationen werden vor dem zeitunabhängigen Abgang gebildet:
min_i min_s (T_i(s) + Gehzeit(s,p)) = min_s (min_i T_i(s) + Gehzeit(s,p)).
Je Ursprung wird ein R5-Intervalllauf ausgeführt. Direkte Wege ohne ÖPNV-Fahrt
werden aus der Ergebnismatrix ausgeschlossen.
"""

from __future__ import annotations

import copy
import hashlib
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .exceptions import DependencyError

ADAPTER_VERSION = "transit-minimum-1"


@lru_cache(maxsize=1)
def verify_engine() -> None:
    """Prüft die unterstützte R5py-Version und die SHA-256-Prüfsumme der R5-JAR."""
    if version("r5py") != "1.1.7":
        raise DependencyError("Der Routingadapter benötigt r5py==1.1.7 mit R5 7.5.1-r5py.")
    from r5py.util.classpath import R5_CLASSPATH

    with Path(R5_CLASSPATH).open("rb") as jar:
        checksum = hashlib.file_digest(jar, "sha256").hexdigest()
    if checksum != "d50be106cadd7b636cfc0e209052767d7df570629f79fdf98ecd5cf5d2d89be7":
        raise DependencyError(
            "Die R5-7.5.1-r5py-JAR muss der festgelegten SHA-256-Prüfsumme entsprechen."
        )


def minimum_transit_matrix(r5py, network, **kwargs) -> pd.DataFrame:
    """Berechnet die kürzeste Intervallreisezeit mit mindestens einer ÖPNV-Fahrt."""
    verify_engine()
    import jpype

    # JPype macht die Java-Komponenten von R5 aus Python zugänglich.
    # Zugang, Fahrplansuche und Abgang werden hier getrennt zusammengesetzt.
    J = jpype.JClass
    StreetRouter = J("com.conveyal.r5.streets.StreetRouter")
    StreetMode = J("com.conveyal.r5.profile.StreetMode")
    RoutingVariable = J("com.conveyal.r5.streets.StreetRouter$State$RoutingVariable")
    FastRaptorWorker = J("com.conveyal.r5.profile.FastRaptorWorker")
    Propagater = J("com.conveyal.r5.profile.PerTargetPropagater")
    Reducer = J("com.conveyal.r5.analyst.TravelTimeReducer")
    EnumSet = J("java.util.EnumSet")
    unreachable = np.iinfo(np.int32).max

    class MinimumTransitMatrix(r5py.TravelTimeMatrix):
        def _travel_times_per_origin(self, from_id):
            """Verbindet Zugang, RAPTOR-Minima und Abgang zu Reisezeiten eines Ursprungs."""
            # Jeder Ursprung erhält eine eigene Anfrage, damit dessen
            # Koordinate und Zeitreduktion die übrigen Ursprünge nicht ändern.
            request = copy.copy(self.request)
            request.origin = self.origins.loc[self.origins.id.eq(from_id)].geometry.item()
            task = request._regional_task
            native = self.transport_network._transport_network
            destinations = task.destinationPointSets[0]
            empty = pd.DataFrame(
                {
                    "from_id": str(from_id),
                    "to_id": self.destinations.id.to_numpy(),
                    "travel_time": np.nan,
                }
            )
            # Zuerst werden Haltestellen gesucht, die vom Ursprung über
            # das Straßennetz innerhalb der R5-Gehgrenze erreichbar sind.
            street = StreetRouter(native.streetLayer)
            street.profileRequest = task
            street.streetMode = StreetMode.WALK
            if not street.setOrigin(task.fromLat, task.fromLon):
                return empty
            street.timeLimitSeconds = min(task.maxTripDurationMinutes, task.maxWalkTime) * 60
            street.quantityToMinimize = RoutingVariable.DURATION_SECONDS
            street.route()
            access = street.getReachedStops()
            if access.isEmpty():
                return empty
            # RAPTOR durchsucht den Fahrplan im gesamten Abfahrtsfenster.
            # Die Zugangsdauern bilden den Startpunkt der ÖPNV-Suche.
            iterations = FastRaptorWorker(native.transitLayer, task, access).route()
            if not len(iterations):
                return empty
            # Je Zielhaltestelle bleibt das Minimum über alle Iterationen.
            # Da der anschließende Fußweg zeitunabhängig ist, kann diese
            # Minimumbildung vor der Übertragung auf die POIs erfolgen.
            best = np.asarray(iterations[0], dtype=np.int32).copy()
            for times in iterations[1:]:
                np.minimum(best, np.asarray(times, dtype=np.int32), out=best)
            # Eine bereits minimierte Iteration: kein Perzentil als Ersatzminimum.
            # Der folgende Ein-Minuten-Task dient nur der Ergebnisreduktion, nicht
            # dem Routing; der RAPTOR-Lauf oben hat das vollständige Zeitfenster.
            task.toTime = task.fromTime + 60
            task.monteCarloDraws = 1
            task.percentiles = jpype.JArray(jpype.JInt)([50])
            collapsed = jpype.JArray(jpype.JInt, 2)([best])
            # Direkte Wege vom Ursprung zum POI werden als unerreichbar
            # übergeben. Die Matrix enthält damit nur Wege mit ÖPNV-Anteil;
            # reine Gehangebote entstehen separat in der Quellenbildung.
            direct = jpype.JArray(jpype.JInt)(
                np.full(len(self.destinations), unreachable, dtype=np.int32)
            )
            # Der Abgang verbindet die erreichten Haltestellen über das
            # Straßennetz mit den POIs und reduziert auf eine Zeit je Ziel.
            propagater = Propagater(
                destinations,
                native.streetLayer,
                EnumSet.of(StreetMode.WALK),
                task,
                collapsed,
                direct,
            )
            propagater.travelTimeReducer = Reducer(task, native)
            result = propagater.propagate()
            values = np.asarray(result.travelTimes.getValues()[0], dtype=float)
            # R5 kennzeichnet unerreichbare Ziele mit einem Integer-Sentinel.
            # In den Ergebnistabellen wird dafür ein fehlender Wert verwendet.
            values[values == unreachable] = np.nan
            return pd.DataFrame(
                {
                    "from_id": str(from_id),
                    "to_id": self.destinations.id.to_numpy(),
                    "travel_time": values,
                }
            )

    return pd.DataFrame(MinimumTransitMatrix(network, **kwargs))
