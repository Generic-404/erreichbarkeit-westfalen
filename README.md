# Erreichbarkeit von Einrichtungen in Westfalen

Begleitprojekt zur Bachelorarbeit über die öffentliche Mobilitätsversorgung in
Westfalen. Das Programm berechnet die Erreichbarkeit von zehn Einrichtungsarten
zu Fuß und mit öffentlichen Verkehrsmitteln auf einem 100-Meter-Raster. Es
verbindet Reisezeitscores mit Zensusmerkmalen und erstellt eine interaktive Karte.
Untersucht werden die Regierungsbezirke Arnsberg, Detmold und Münster
(einschließlich Lippe).

## Voraussetzungen

- Linux mit Python ab 3.11 einschließlich `venv` und OpenJDK 21;
  geprüft mit Python 3.12.
- Internetzugang für die Installation und die Hintergrundkarte.
- Etwa 30 GiB freier Festplattenspeicher als Arbeitsreserve; wiederholte Läufe
  benötigen zusätzlichen Platz für vorherige Ergebnisse.
- Ausreichend Arbeitsspeicher und einen Browser zum Anzeigen der Karte.

**Arbeitsspeicher:** Der Java-Heap ist standardmäßig auf `70%` des gesamten
erkannten RAM begrenzt. Die Obergrenze wird beim Start berechnet und bleibt
während des Laufs fest. Python und das Betriebssystem benötigen zusätzlich
RAM. Bei unzureichendem Speicher kann die Verarbeitung durch Auslagerung
langsamer werden oder abbrechen. Für die vollständige Westfalen-Berechnung
liegt keine ermittelte RAM-Mindestgröße vor; die Heap-Grenze bezeichnet den
maximal zulässigen Java-Heap und nicht den gemessenen Speicherbedarf.

## Herunterladen und starten

1. Unter [Releases](https://github.com/Generic-404/erreichbarkeit-westfalen/releases)
   das vollständige Paket **`erreichbarkeit-westfalen-v1.0.0.zip`** herunterladen
   (etwa 1 GiB). Es enthält Code, Konfiguration und sämtliche Eingangsdaten.
   Die automatisch angebotenen Downloads „Source code“ enthalten die großen
   Eingangsdaten nicht.
2. Das Archiv entpacken und ein Terminal im entpackten Projektordner öffnen,
   in dem `start.sh` und diese README liegen.
3. Die Berechnung starten:

   ```bash
   bash start.sh
   ```

Das Skript prüft Python, Java und die Prüfsummen der Eingangsdaten, richtet die
lokale Python-Umgebung `.venv` ein, installiert die festgelegten Abhängigkeiten
und führt alle Berechnungsphasen bis zur Kartenerstellung aus. Python und Java
müssen zuvor auf dem System installiert sein. Node.js ist für diesen Ablauf
nicht erforderlich.

Das Terminal während der Berechnung geöffnet lassen. Bei einem Fehler stoppt
das Skript; ein erneuter Aufruf beginnt wieder mit der ersten Phase. Ergebnisse
werden unter `results/westfalen-v1/` gespeichert. Sie sind nicht im Download
enthalten, sondern entstehen bei der Ausführung.

Bei Bedarf lässt sich die Java-Speichergrenze beim Start überschreiben,
beispielsweise mit einer festen Obergrenze von 40 GiB:

```bash
bash start.sh --max-memory 40G
```

Die fachlichen Parameter und Datenpfade stehen in
[config/westfalen.yaml](config/westfalen.yaml). Für den mitgelieferten
Westfalen-Lauf sind dort keine Änderungen nötig.

## Ablauf der Berechnung

Das Startskript führt die folgenden sieben Phasen automatisch in dieser
Reihenfolge aus. Im Terminal wird jeweils der Name der gestarteten Phase
angezeigt. Jeder Schritt verwendet die Ergebnisse seiner Vorgänger.

| Phase | Aufgabe |
| --- | --- |
| 1. `prepare` | Liest die Eingangsdaten ein, wählt Einrichtungen aus den OSM-Daten aus, fasst Haltestellen zusammen und erstellt das 100-m-Raster mit den zugehörigen Zensusmerkmalen. |
| 2. `route` | Berechnet mit R5 und dem GTFS-Fahrplan die ÖPNV-Reisezeiten von Haltestellen zu ausgewählten Einrichtungen. |
| 3. `sources` | Bestimmt je Haltestelle und Einrichtungsart die beste geroutete Reisezeit. Daraus und aus direkt zu Fuß erreichbaren Einrichtungen entstehen die Ausgangspunkte für die Übertragung auf das Raster. |
| 4. `transfer` | Überträgt die Reisezeiten dieser Ausgangspunkte auf die Rasterzellen und berücksichtigt dabei die zusätzliche Gehzeit. |
| 5. `score` | Wandelt die Reisezeiten in Erreichbarkeitsscores je Einrichtungsart um und berechnet daraus den gewichteten Gesamtscore. |
| 6. `export` | Führt die Ergebnisse zusammen und speichert Tabellen, GeoPackages, Nachweisdaten und eine Basiskarte. |
| 7. `evaluate` | Ordnet die fertigen Scores und Bevölkerungsmerkmale in Klassen ein, leitet Untersuchungsprioritäten ab und erstellt die erweiterte interaktive Karte. Dabei werden keine neuen Reisezeiten berechnet. |

## Karte öffnen

Nach erfolgreichem Abschluss im Projektordner den lokalen Kartenserver starten:

```bash
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory results/westfalen-v1/evaluation
```

Im Browser [http://localhost:8000/accessibility_map.html](http://localhost:8000/accessibility_map.html)
öffnen. Die HTML-Datei benötigt den Server, um ihre Kartendaten nachzuladen.
Der Server bleibt im Terminal aktiv; mit `Strg+C` wird er beendet.

In der Karte lassen sich Scores und Zensusmerkmale auswählen, zwischen der
1-km-Übersicht und den 100-m-Zellen wechseln sowie Haltestellen und Einrichtungen
einblenden. Ein Klick auf eine Zelle zeigt deren Werte. Ergebnistabellen und
GeoPackages liegen ebenfalls im Ergebnisordner.

Für eigenes Hosting die Datei `evaluation/accessibility_map.html` und den in
`evaluation/map_report.json` unter `assets_directory` genannten vollständigen
Begleitordner zusammen übernehmen. Ihre relative Ordnerstruktur und die
Quellenangaben müssen erhalten bleiben.

## Datenquellen und Lizenz

Die mitgelieferten Daten halten die verwendeten Eingabestände fest;
`data/checksums.sha256` enthält ihre Prüfsummen. Die Gebietsgrenze und der
zusammengeführte OSM-Ausschnitt mit 15 km Außenpuffer sind bereits vorbereitet.

| Daten | Quelle und Stand |
| --- | --- |
| Gebietsgrenze | [Geobasis NRW, DVG2](https://www.opengeodata.nrw.de/produkte/geobasis/vkg/dvg/dvg1/Nutzerinformationen.pdf), Stand 12. November 2025; Arnsberg, Detmold und Münster ohne zusätzliche Vereinfachung vereinigt |
| OpenStreetMap | [© OpenStreetMap-Mitwirkende, ODbL](https://www.openstreetmap.org/copyright); deutsche und niederländische Teilbestände, Replikationszeitstempel `2026-09-22T20:22:59Z` |
| Fahrplandaten | [GTFS.de](https://www.gtfs.de/de/feeds/), Daten bereitgestellt von DELFI e.V.; festgehaltener Feedstand `latest-free` |
| Zensus | [Statistische Ämter des Bundes und der Länder, Zensus 2022](https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Zensus2022/_inhalt.html), Stichtag 15. Mai 2022; Bevölkerung und Altersmerkmale im 100-m-Gitter |

Der Projektcode steht unter der [MIT-Lizenz](LICENSE). Für die Eingangsdaten
gelten die jeweiligen Nutzungsbedingungen der Anbieter. Herkunft und
Attribution bleiben bei der Weitergabe erhalten.
