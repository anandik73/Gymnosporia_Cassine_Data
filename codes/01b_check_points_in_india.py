# -*- coding: utf-8 -*-
"""
Step 1b: Test every occurrence point against the India boundary polygon.

Step 1 excludes two records that were identified by inspection. This script runs
the check the Methods section describes: a point-in-polygon test of all records
against the national boundary, so the statement in the manuscript is backed by an
output file rather than by a code comment.

For any point that falls outside the polygon it also reports the shortest distance
to the boundary, which is where the "km offshore" figure in the Methods comes from.

Input : Results/cleaned_coordinates_master.xlsx  (the cleaned n = 628 dataset)
        ../Indiastate_shapefile/India_Country_Boundary.shp
Output: Results/point_in_polygon_check.csv

NOTE ON THE PROJECTION. India_Country_Boundary.prj is
WGS_1984_Web_Mercator_Auxiliary_Sphere (EPSG:3857), in metres, not degrees. Its
bounding box runs to roughly x = 1.08e7, y = 4.45e6. Testing raw longitude and
latitude against it returns "outside" for every record. Occurrence coordinates are
therefore projected to Web Mercator before the test, and boundary vertices are
projected back to degrees when a distance is needed.

Dependencies: pyshp, pandas, numpy, openpyxl. No geopandas needed.
"""

import os
import sys

import numpy as np
import pandas as pd
import shapefile  # pyshp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Results")
SHP = os.path.join(HERE, "..", "Indiastate_shapefile", "India_Country_Boundary.shp")
MASTER = os.path.join(OUT, "cleaned_coordinates_master.xlsx")

# The two records Step 1 removes, re-tested here so the check covers them too.
EXCLUDED = [
    ("Gymnosporia_rothiana", 25.44, 77.69),
    ("Gymnosporia_emarginata", 8.2833, 75.4217),
]

R_EARTH_KM = 6371.0088
R_MERC = 6378137.0  # Web Mercator sphere radius, metres


def to_mercator(lon, lat):
    x = R_MERC * np.radians(lon)
    y = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def from_mercator(x, y):
    lon = np.degrees(np.asarray(x) / R_MERC)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y) / R_MERC)) - np.pi / 2)
    return lon, lat


def load_rings():
    """Return every polygon ring in the boundary shapefile as an (n, 2) lon/lat array."""
    sf = shapefile.Reader(SHP)
    rings = []
    for shp in sf.shapes():
        pts = np.asarray(shp.points, dtype=float)
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            if b - a >= 4:
                rings.append(pts[a:b])
    return rings


def point_in_rings(px, py, rings):
    """Ray-casting test in projected coordinates. Odd crossing count means inside."""
    inside = False
    for ring in rings:
        x, y = ring[:, 0], ring[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        crosses = ((y > py) != (y1 > py))
        if not crosses.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x1 - x) * (py - y) / (y1 - y) + x
        if np.count_nonzero(crosses & (px < xin)) % 2 == 1:
            inside = not inside
    return inside


def km_to_boundary(lon, lat, rings):
    """Great-circle distance to the nearest boundary vertex, in km."""
    best = np.inf
    la = np.radians(lat)
    lo = np.radians(lon)
    for ring in rings:
        rlon, rlat = from_mercator(ring[:, 0], ring[:, 1])
        rl = np.radians(rlat)
        rlo = np.radians(rlon)
        d = 2 * np.arcsin(np.sqrt(
            np.sin((rl - la) / 2) ** 2 + np.cos(la) * np.cos(rl) * np.sin((rlo - lo) / 2) ** 2))
        best = min(best, float(d.min()) * R_EARTH_KM)
    return best


def main():
    if not os.path.exists(SHP):
        sys.exit(f"boundary shapefile not found: {SHP}")
    rings = load_rings()
    print(f"boundary rings: {len(rings)}")

    xl = pd.ExcelFile(MASTER)
    sheets = [s for s in xl.sheet_names if "_" in s and s not in ("Merge_log", "MP_point_note")]

    rows = []
    for sp in sheets:
        d = xl.parse(sp)
        lat_col = [c for c in d.columns if c.lower().startswith("lat")][0]
        lon_col = [c for c in d.columns if c.lower().startswith("lon")][0]
        for lat, lon in zip(d[lat_col], d[lon_col]):
            rows.append({"species": sp, "lat": float(lat), "lon": float(lon),
                         "in_dataset": True})
    for sp, lat, lon in EXCLUDED:
        rows.append({"species": sp, "lat": lat, "lon": lon, "in_dataset": False})

    out = pd.DataFrame(rows)
    mx, my = to_mercator(out["lon"].values, out["lat"].values)
    out["inside_india"] = [point_in_rings(x, y, rings) for x, y in zip(mx, my)]
    out["km_to_boundary"] = [
        round(km_to_boundary(r.lon, r.lat, rings), 1) if not r.inside_india else 0.0
        for r in out.itertuples()
    ]
    out.to_csv(os.path.join(OUT, "point_in_polygon_check.csv"), index=False)

    kept = out[out.in_dataset]
    print(f"\nrecords in the analysed dataset: {len(kept)}")
    print(f"  inside the India polygon : {int(kept.inside_india.sum())}")
    outside = kept[~kept.inside_india]
    print(f"  outside                  : {len(outside)}")
    if len(outside):
        print(outside[["species", "lat", "lon", "km_to_boundary"]].to_string(index=False))

    print("\nrecords excluded by Step 1, re-tested here:")
    print(out[~out.in_dataset][["species", "lat", "lon", "inside_india",
                                "km_to_boundary"]].to_string(index=False))
    print(f"\nwritten: {os.path.join(OUT, 'point_in_polygon_check.csv')}")


if __name__ == "__main__":
    main()
