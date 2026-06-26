#!/usr/bin/env python
"""
Compute a per-station Vs30 lookup from the TR Vs30 polygon shapefile.

Each station coordinate (lon/lat) is matched to the Vs30 polygon that contains
it. Stations that fall outside every polygon (e.g. just offshore) fall back to
the nearest polygon's value.

Output schema (mirrors station_locations.json):
{
  "EDC": 576.18,
  "ADVT": 419.05,
  ...
}
"""

import argparse
import json
from pathlib import Path


def parse_args():
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent.parent
    parser = argparse.ArgumentParser(description="Compute per-station Vs30 from a polygon shapefile")
    parser.add_argument(
        "--station_locations",
        type=str,
        default=str(base_dir / "embeddings" / "station_locations.json"),
        help="Path to station_locations.json (lon/lat per station).",
    )
    parser.add_argument(
        "--shapefile",
        type=str,
        default=str(repo_root / "VS30" / "TRVs30_polygons.shp"),
        help="Path to the Vs30 polygon shapefile.",
    )
    parser.add_argument(
        "--vs30_field",
        type=str,
        default="Vs30",
        help="Attribute column holding the Vs30 value.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(base_dir / "embeddings" / "station_vs30.json"),
        help="Output JSON path for the station -> Vs30 lookup.",
    )
    parser.add_argument(
        "--no_nearest_fallback",
        action="store_true",
        help="Do not fall back to the nearest polygon for stations outside all polygons.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "compute_station_vs30.py requires geopandas/shapely in the active environment."
        ) from exc

    station_locations = json.load(open(args.station_locations, "r"))
    if not station_locations:
        raise RuntimeError(f"No stations found in {args.station_locations}")

    names = list(station_locations.keys())
    pts = gpd.GeoDataFrame(
        {"station": names},
        geometry=[
            Point(float(station_locations[n]["longitude"]), float(station_locations[n]["latitude"]))
            for n in names
        ],
        crs=4326,
    )

    print(f"Reading Vs30 polygons from {args.shapefile} ...")
    polygons = gpd.read_file(args.shapefile)
    if args.vs30_field not in polygons.columns:
        raise KeyError(
            f"Column '{args.vs30_field}' not in shapefile (have: {list(polygons.columns)})."
        )
    polygons = polygons[[args.vs30_field, "geometry"]]

    pts = pts.to_crs(polygons.crs)

    # Point-in-polygon match.
    joined = gpd.sjoin(pts, polygons, how="left", predicate="within").drop_duplicates("station")

    station_vs30 = {}
    unmatched = []
    for _, row in joined.iterrows():
        val = row[args.vs30_field]
        if val is None or (isinstance(val, float) and val != val):  # NaN check
            unmatched.append(row["station"])
        else:
            station_vs30[row["station"]] = float(val)

    if unmatched and not args.no_nearest_fallback:
        print(f"Nearest-polygon fallback for {len(unmatched)} station(s): {', '.join(unmatched)}")
        missing = pts[pts["station"].isin(unmatched)]
        near = gpd.sjoin_nearest(missing, polygons, how="left").drop_duplicates("station")
        for _, row in near.iterrows():
            val = row[args.vs30_field]
            if val is not None and not (isinstance(val, float) and val != val):
                station_vs30[row["station"]] = float(val)
                unmatched.remove(row["station"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(station_vs30, f, indent=2, sort_keys=True)

    print(f"\nSaved station Vs30: {output_path}")
    print(f"Resolved: {len(station_vs30)} / {len(names)}")
    if unmatched:
        print("Unresolved stations:", ", ".join(sorted(unmatched)))


if __name__ == "__main__":
    main()
