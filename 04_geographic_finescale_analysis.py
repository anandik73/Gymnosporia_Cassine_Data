"""
Step 4: Geographic range size (convex-hull area), pairwise hull overlap
(geographic and climatic), and fine-scale nearest-neighbour co-occurrence
between congener pairs.

Input : Results/occurrence_climate_CHELSA_withPCs.csv (from Step 3)
Output: Results/range_vs_niche_breadth.csv
        Results/pairwise_hull_overlap.csv
        Results/finescale_nearest_neighbor.csv
        Results/within_species_nn_reference.csv
"""
import os
import numpy as np
import pandas as pd
import pyproj
from scipy import stats
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "Results")

df = pd.read_csv(f"{OUT}/occurrence_climate_CHELSA_withPCs.csv")
niche = pd.read_csv(f"{OUT}/niche_breadth_per_species.csv").set_index("species")
species_list = sorted(df["species"].unique().tolist())
geod = pyproj.Geod(ellps="WGS84")

GYMNO = ["Gymnosporia_rothiana", "Gymnosporia_emarginata", "Gymnosporia_senegalensis"]
CASSINE = ["Cassine_glauca", "Cassine_paniculata"]
CONGENER_PAIRS = [(GYMNO[0], GYMNO[1]), (GYMNO[0], GYMNO[2]), (GYMNO[1], GYMNO[2]), (CASSINE[0], CASSINE[1])]


# ---------------- geographic range (convex hull area) ----------------
def hull_area_km2(lats, lons):
    pts = np.column_stack([lons, lats])
    hull = ConvexHull(pts)
    hp = pts[hull.vertices]
    area, _ = geod.polygon_area_perimeter(hp[:, 0], hp[:, 1])
    return abs(area) / 1e6


range_rows = []
for sp in species_list:
    sub = df[df.species == sp]
    area = hull_area_km2(sub["lat"].values, sub["lon"].values)
    range_rows.append({"species": sp, "n": len(sub), "hull_area_km2": area,
                        "niche_breadth": niche.loc[sp, "niche_breadth"]})
    print(f"{sp}: hull={area:,.0f} km2, niche breadth={niche.loc[sp, 'niche_breadth']:.3f}")

range_df = pd.DataFrame(range_rows)
range_df.to_csv(f"{OUT}/range_vs_niche_breadth.csv", index=False)
corr, pval = stats.pearsonr(range_df["hull_area_km2"], range_df["niche_breadth"])
print(f"\nCorrelation (range area vs niche breadth), n=5 species: r={corr:.3f}, p={pval:.3f}")


# ---------------- pairwise hull overlap (geographic + climate PC space) ----------------
def geo_hull_overlap(sp1, sp2):
    s1, s2 = df[df.species == sp1], df[df.species == sp2]
    h1 = MultiPoint(list(zip(s1["lon"], s1["lat"]))).convex_hull
    h2 = MultiPoint(list(zip(s2["lon"], s2["lat"]))).convex_hull
    inter = h1.intersection(h2)

    def poly_area_km2(poly):
        if poly.is_empty or poly.geom_type in ("Point", "LineString"):
            return 0.0
        lons, lats = poly.exterior.coords.xy
        area, _ = geod.polygon_area_perimeter(lons, lats)
        return abs(area) / 1e6

    a1, a2 = poly_area_km2(h1), poly_area_km2(h2)
    ai = poly_area_km2(inter) if not inter.is_empty else 0.0
    pct = ai / min(a1, a2) * 100 if min(a1, a2) > 0 else 0
    c1 = np.array([s1["lon"].mean(), s1["lat"].mean()])
    c2 = np.array([s2["lon"].mean(), s2["lat"].mean()])
    _, _, dist = geod.inv(c1[0], c1[1], c2[0], c2[1])
    return pct, dist / 1000


def climate_hull_jaccard(sp1, sp2):
    p1 = df[df.species == sp1][["PC1", "PC2"]].values
    p2 = df[df.species == sp2][["PC1", "PC2"]].values
    h1, h2 = MultiPoint(p1).convex_hull, MultiPoint(p2).convex_hull
    inter, union = h1.intersection(h2).area, h1.union(h2).area
    smaller = min(h1.area, h2.area)
    jac = inter / union if union > 0 else np.nan
    pct = inter / smaller * 100 if smaller > 0 else np.nan
    return jac, pct


overlap_rows = []
print("\n--- Pairwise hull overlap ---")
for sp1, sp2 in CONGENER_PAIRS:
    geo_pct, cdist = geo_hull_overlap(sp1, sp2)
    jac, clim_pct = climate_hull_jaccard(sp1, sp2)
    overlap_rows.append({"sp1": sp1, "sp2": sp2, "geog_centroid_dist_km": cdist,
                          "geog_pct_overlap_of_smaller_hull": geo_pct,
                          "climate_jaccard_PC1PC2": jac, "climate_pct_overlap_of_smaller_hull": clim_pct})
    print(f"{sp1} - {sp2}: centroid={cdist:.0f}km, geo_overlap={geo_pct:.1f}%, "
          f"climate_jaccard={jac:.3f}, climate_overlap={clim_pct:.1f}%")
pd.DataFrame(overlap_rows).to_csv(f"{OUT}/pairwise_hull_overlap.csv", index=False)


# ---------------- fine-scale nearest-neighbour ----------------
def nn_dists(sp1, sp2):
    s1 = df[df.species == sp1][["lon", "lat"]].values
    s2 = df[df.species == sp2][["lon", "lat"]].values
    out = []
    for lon1, lat1 in s1:
        _, _, d = geod.inv(np.full(len(s2), lon1), np.full(len(s2), lat1), s2[:, 0], s2[:, 1])
        out.append(d.min() / 1000)
    return np.array(out)


nn_rows = []
print("\n--- Fine-scale nearest-neighbour ---")
for sp1, sp2 in CONGENER_PAIRS:
    d = nn_dists(sp1, sp2)
    nn_rows.append({"sp1": sp1, "sp2": sp2, "median_km": np.median(d), "mean_km": d.mean(),
                     "pct_within_25km": (d < 25).mean() * 100, "pct_within_50km": (d < 50).mean() * 100})
    print(f"{sp1} -> nearest {sp2}: median={np.median(d):.1f}km, "
          f"pct<25km={(d < 25).mean() * 100:.0f}%, pct<50km={(d < 50).mean() * 100:.0f}%")
pd.DataFrame(nn_rows).to_csv(f"{OUT}/finescale_nearest_neighbor.csv", index=False)

within_rows = []
print("\n--- Within-species NN spacing (sampling-density reference) ---")
for sp in species_list:
    s = df[df.species == sp][["lon", "lat"]].values
    out = []
    for i, (lon1, lat1) in enumerate(s):
        others = np.delete(s, i, axis=0)
        _, _, d = geod.inv(np.full(len(others), lon1), np.full(len(others), lat1), others[:, 0], others[:, 1])
        out.append(d.min() / 1000)
    within_rows.append({"species": sp, "median_within_species_nn_km": np.median(out)})
    print(f"{sp}: median within-species NN = {np.median(out):.2f} km")
pd.DataFrame(within_rows).to_csv(f"{OUT}/within_species_nn_reference.csv", index=False)

print(f"\nAll results saved to {OUT}")
