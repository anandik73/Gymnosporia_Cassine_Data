"""
Step 5: Generate the four manuscript figures.
  Fig 1: PCA biplot (climate niche space + variable loading vectors)
  Fig 2: Occurrence map on satellite basemap with India boundary
  Fig 3: Fine-scale co-occurrence bar chart
  Fig 4: Geographic range vs climatic niche breadth scatter

Input : Results/occurrence_climate_CHELSA_withPCs.csv, Results/pca_loadings.csv,
        Results/range_vs_niche_breadth.csv, Results/finescale_nearest_neighbor.csv
        ../Indiastate_shapefile/India_Country_Boundary.shp
Output: Figures/Fig1_PCA_biplot.png, Fig2_occurrence_map.png,
        Fig3_finescale_segregation.png, Fig4_range_vs_niche.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon as MplPolygon

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "Results")
FIG = os.path.join(HERE, "..", "Figures")
os.makedirs(FIG, exist_ok=True)

COLORS = {"Gymnosporia_rothiana": "#1b9e77", "Gymnosporia_emarginata": "#d95f02",
          "Gymnosporia_senegalensis": "#7570b3", "Cassine_glauca": "#e7298a", "Cassine_paniculata": "#66a61e"}
MARKERS = {"Gymnosporia_rothiana": "o", "Gymnosporia_emarginata": "s",
           "Gymnosporia_senegalensis": "^", "Cassine_glauca": "D", "Cassine_paniculata": "P"}
LABELS = {"Gymnosporia_rothiana": "$G.\\ rothiana$ (endemic)", "Gymnosporia_emarginata": "$G.\\ emarginata$ (widespread)",
          "Gymnosporia_senegalensis": "$G.\\ senegalensis$ (widespread)", "Cassine_glauca": "$C.\\ glauca$ (widespread)",
          "Cassine_paniculata": "$C.\\ paniculata$ (endemic)"}
SHORT = {"Gymnosporia_rothiana": "$G.\\ rothiana$", "Gymnosporia_emarginata": "$G.\\ emarginata$",
         "Gymnosporia_senegalensis": "$G.\\ senegalensis$", "Cassine_glauca": "$C.\\ glauca$", "Cassine_paniculata": "$C.\\ paniculata$"}

df = pd.read_csv(f"{OUT}/occurrence_climate_CHELSA_withPCs.csv")


# ---------------- Fig 1: PCA biplot ----------------
def make_biplot():
    loadings = pd.read_csv(f"{OUT}/pca_loadings.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    for sp in COLORS:
        sub = df[df.species == sp]
        ax.scatter(sub.PC1, sub.PC2, s=22, alpha=0.55, color=COLORS[sp], marker=MARKERS[sp],
                   edgecolor="k", linewidth=0.2, label=LABELS[sp], zorder=3)
        pts = sub[["PC1", "PC2"]].values
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            ax.add_patch(MplPolygon(pts[hull.vertices], closed=True, fill=True, facecolor=COLORS[sp],
                                     alpha=0.08, edgecolor=COLORS[sp], linewidth=1.2, zorder=2))
    scale = 8.0
    top_vars = loadings[["PC1", "PC2"]].abs().sum(axis=1).sort_values(ascending=False).head(10).index
    for var in top_vars:
        x, y = loadings.loc[var, "PC1"] * scale, loadings.loc[var, "PC2"] * scale
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.1), zorder=4)
        ax.text(x * 1.12, y * 1.12, var, fontsize=8.5, color="dimgray", ha="center", va="center", zorder=5)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("PC1 (thermal seasonality / diurnal-annual range)")
    ax.set_ylabel("PC2 (cold-season temperature)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/Fig1_PCA_biplot.png", dpi=220)
    print("saved Fig1_PCA_biplot.png")


# ---------------- Fig 2: occurrence map ----------------
def make_map():
    shp = os.path.join(HERE, "..", "Indiastate_shapefile", "India_Country_Boundary.shp")
    india_raw = gpd.read_file(shp)
    india = gpd.GeoDataFrame(geometry=[india_raw.union_all()], crs=india_raw.crs).to_crs(epsg=4326)

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    gdf_merc = gdf.to_crs(epsg=3857)
    india_merc = india.to_crs(epsg=3857)

    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    india_merc.boundary.plot(ax=ax, color="white", linewidth=1.0, zorder=2)
    for sp in COLORS:
        sub = gdf_merc[gdf_merc.species == sp]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=26, alpha=0.85, color=COLORS[sp], marker=MARKERS[sp],
                   edgecolor="black", linewidth=0.3, label=LABELS[sp], zorder=3)
    minx, miny, maxx, maxy = india_merc.total_bounds
    pad = 0.05 * (maxx - minx)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad * 1.2, maxy + pad * 0.5)
    try:
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=5)
    except Exception:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager, zoom=5)
    ax.set_axis_off()
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.savefig(f"{FIG}/Fig2_occurrence_map.png", dpi=220, pad_inches=0)
    print("saved Fig2_occurrence_map.png")


# ---------------- Fig 3: fine-scale segregation bar chart ----------------
def make_finescale_chart():
    nn = pd.read_csv(f"{OUT}/finescale_nearest_neighbor.csv")
    pair_labels = {("Gymnosporia_rothiana", "Gymnosporia_emarginata"): "$G.\\ rothiana$ –\n$G.\\ emarginata$",
                   ("Gymnosporia_rothiana", "Gymnosporia_senegalensis"): "$G.\\ rothiana$ –\n$G.\\ senegalensis$",
                   ("Gymnosporia_emarginata", "Gymnosporia_senegalensis"): "$G.\\ emarginata$ –\n$G.\\ senegalensis$",
                   ("Cassine_glauca", "Cassine_paniculata"): "$C.\\ glauca$ –\n$C.\\ paniculata$"}
    labels = [pair_labels[(r.sp1, r.sp2)] for r in nn.itertuples()]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, nn["pct_within_25km"], width, label="within 25 km", color="#3b6fa0")
    ax.bar(x + width / 2, nn["pct_within_50km"], width, label="within 50 km", color="#8fb8de")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of occurrence points with a\ncongener within threshold distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG}/Fig3_finescale_segregation.png", dpi=220)
    print("saved Fig3_finescale_segregation.png")


# ---------------- Fig 4: range vs niche breadth ----------------
def make_range_vs_niche():
    range_df = pd.read_csv(f"{OUT}/range_vs_niche_breadth.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for row in range_df.itertuples():
        ax.scatter(row.hull_area_km2, row.niche_breadth, s=140, color=COLORS[row.species], marker=MARKERS[row.species],
                   edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(SHORT[row.species], (row.hull_area_km2, row.niche_breadth),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Geographic range (convex-hull area, km², log scale)")
    ax.set_ylabel("Climatic niche breadth\n(mean distance to niche centroid, PC1–PC3)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/Fig4_range_vs_niche.png", dpi=220)
    print("saved Fig4_range_vs_niche.png")


if __name__ == "__main__":
    make_biplot()
    make_map()
    make_finescale_chart()
    make_range_vs_niche()
