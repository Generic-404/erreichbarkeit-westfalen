import zipfile
from pathlib import Path

import geopandas as gpd
import osmium
import pytest
import yaml
from shapely.geometry import box

from accessibility.census import generate_census_grid
from accessibility.config import load_config


@pytest.fixture
def tiny_project(tmp_path: Path):
    """Erzeugt ein synthetisches Straßengitter mit zwei Haltestellen und zwei POI-Arten."""
    pbf = tmp_path / "streets.osm.pbf"
    with osmium.SimpleWriter(str(pbf)) as writer:
        for y in range(10):
            for x in range(10):
                writer.add_node(
                    osmium.osm.mutable.Node(
                        id=1 + y * 10 + x, location=(7 + x * 0.002, 52 + y * 0.001)
                    )
                )
        for y in range(10):
            writer.add_way(
                osmium.osm.mutable.Way(
                    id=100 + y,
                    nodes=[1 + y * 10 + x for x in range(10)],
                    tags={"highway": "residential", "name": f"Row {y}"},
                )
            )
        for x in range(10):
            writer.add_way(
                osmium.osm.mutable.Way(
                    id=200 + x,
                    nodes=[1 + y * 10 + x for y in range(10)],
                    tags={"highway": "residential", "name": f"Column {x}"},
                )
            )
        writer.add_node(
            osmium.osm.mutable.Node(
                id=1001, location=(7.002, 52.002), tags={"amenity": "school", "name": "Schule"}
            )
        )
        writer.add_node(
            osmium.osm.mutable.Node(
                id=1002,
                location=(7.010, 52.002),
                tags={"healthcare": "pharmacy", "name": "Apotheke"},
            )
        )
    gtfs = tmp_path / "feed.zip"
    with zipfile.ZipFile(gtfs, "w") as archive:
        for name, content in {
            "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\na,Test,https://example.org,Europe/Berlin\n",
            "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\nA,Start,52.002,7.002\nB,Ziel,52.002,7.010\n",
            "routes.txt": "route_id,agency_id,route_short_name,route_long_name,route_type\nr,a,T,Testbus,3\n",
            "trips.txt": "route_id,service_id,trip_id\nr,s,t1\nr,s,t2\n",
            "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\nt1,10:10:00,10:10:00,A,1\nt1,10:18:00,10:18:00,B,2\nt2,12:10:00,12:10:00,A,1\nt2,12:28:00,12:28:00,B,2\n",
            "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\ns,1,1,1,1,1,0,0,20260907,20260913\n",
        }.items():
            archive.writestr(name, content)
    boundary = gpd.GeoDataFrame(
        {"name": ["test"]}, geometry=[box(7.001, 52.001, 7.012, 52.004)], crs="EPSG:4326"
    )
    boundary_path = tmp_path / "boundary.gpkg"
    boundary.to_file(boundary_path, driver="GPKG")
    reference = generate_census_grid(boundary).iloc[0]
    census = tmp_path / "Zensus 100m.csv"
    census.write_text(
        f"GITTER_ID_100m;x_mp_100m;y_mp_100m;Alter\n{reference.grid_id};{int(reference.center_x)};{int(reference.center_y)};42,5\n"
    )
    raw = yaml.safe_load((Path(__file__).parents[1] / "config/westfalen.yaml").read_text())
    raw["paths"] = {
        "boundary": str(boundary_path),
        "osm_pbf": str(pbf),
        "gtfs": str(gtfs),
        "output_directory": str(tmp_path / "out"),
        "census_csvs": [str(census)],
    }
    raw["area"].update(buffer_km=1, crop_osm=False)
    raw["poi_categories"] = [
        c for c in raw["poi_categories"] if c["id"] in ("schools", "pharmacies")
    ]
    for category in raw["poi_categories"]:
        category["weight"] = 50
    raw["routing"]["candidate_selection"]["candidates_per_category"] = {
        "schools": 10,
        "pharmacies": 10,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_config(config_path)
