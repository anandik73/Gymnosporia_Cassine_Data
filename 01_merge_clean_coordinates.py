"""
Step 1: Merge original + newly-added GBIF occurrence coordinates per species,
remove exact-coordinate duplicates, and produce a full point-level audit trail
of every record that was removed (duplicate or outlier) or newly added.

Input : Coordinates/ (original CSV/XLSX files) and the GBIF_*.csv pulls,
        both inside the "Gymnosporia_MS for revision" folder.
Output: Results/cleaned_coordinates_master.xlsx  (one sheet per species, final data)
        Results/removed_and_added_points.xlsx     (point-level log for the manuscript)
"""
import csv
import os
import openpyxl
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "Coordinates")
OUT = os.path.join(os.path.dirname(__file__), "Results")
os.makedirs(OUT, exist_ok=True)


def load_csv_multi(path):
    """Original single-line-per-species CSV format (Species,Lat,Lon,,Species,Lat,Lon,,...)"""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    pts = []
    for r in rows[1:]:
        r = [c.strip() for c in r]
        if len(r) >= 3 and r[0]:
            try:
                pts.append((float(r[1]), float(r[2])))
            except ValueError:
                pass
        elif len(r) == 2 and r[0]:
            try:
                pts.append((float(r[0]), float(r[1])))
            except ValueError:
                pass
    return pts


def load_xlsx(path):
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    pts = []
    for r in ws.iter_rows(values_only=True, min_row=2):
        if r[1] not in (None, "") and r[2] not in (None, ""):
            try:
                pts.append((float(r[1]), float(r[2])))
            except ValueError:
                pass
    return pts


def load_gbif_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    pts = []
    for r in rows[1:]:
        if len(r) >= 3 and r[1] and r[2]:
            try:
                pts.append((float(r[1]), float(r[2])))
            except ValueError:
                pass
    return pts


# species -> (base file type, base file, gbif file)
SPECIES_CONFIG = {
    "Gymnosporia_rothiana":     {"base": ("xlsx", "G_rothiana IN.xlsx"),      "gbif": "GBIF_Gymnosporia_rothiana.csv"},
    "Gymnosporia_emarginata":   {"base": ("xlsx", "G_emarginata IN.xlsx"),    "gbif": "GBIF_Gymnosporia_emarginata.csv"},
    "Gymnosporia_senegalensis": {"base": ("csv_multi", "G_senegalensis IN.csv"), "gbif": "GBIF_Gymnosporia_senegalensis.csv"},
    "Cassine_glauca":           {"base": ("csv_multi", "C_glauca IN.csv"),    "gbif": "GBIF_Cassine_glauca.csv"},
    "Cassine_paniculata":       {"base": ("csv_multi", "C_paniculata IN.csv"), "gbif": "GBIF_Cassine_paniculata.csv"},
}

# Records excluded before/during this script (see manuscript Methods) because
# they are implausible outliers in the *base* (pre-GBIF) data:
#  1. A G. rothiana record at ~25.44N, 77.69E (Madhya Pradesh), far outside the
#     species' known Western Ghats range - almost certainly mislabelled or
#     misidentified. Already absent from G_rothiana IN.xlsx (68 rows) relative
#     to the earlier G_rothiana IN.csv (69 rows) before this script runs.
#  2. A G. emarginata record at 8.2833N, 75.4217E, which a spatial check against
#     the India boundary shapefile places ~143 km out in the Arabian Sea (no
#     landmass at that position) - also almost certainly a coordinate error.
#     This one IS present in G_emarginata IN.xlsx and is actively excluded below.
MANUAL_EXCLUSIONS = [
    {"species": "Gymnosporia_rothiana", "lat": 25.44, "lon": 77.69,
     "reason": "Outlier far outside Western Ghats range; likely mislabelled/misidentified. Already absent from source file before this script runs."},
    {"species": "Gymnosporia_emarginata", "lat": 8.2833, "lon": 75.4217,
     "reason": "~143 km offshore in the Arabian Sea per spatial check against India boundary polygon; likely coordinate error. Actively excluded by this script."},
]
EXCLUDE_COORDS = {(e["species"], round(e["lat"], 4), round(e["lon"], 4)) for e in MANUAL_EXCLUSIONS}

log_rows = []
removed_points = []   # every individual point dropped as a duplicate
added_points = []     # every individual point newly contributed by the GBIF pull
all_final = {}

for sp, cfg in SPECIES_CONFIG.items():
    src_type, src_file = cfg["base"]
    src_path = os.path.join(BASE, src_file)
    base_pts = load_xlsx(src_path) if src_type == "xlsx" else load_csv_multi(src_path)
    gbif_pts = load_gbif_csv(os.path.join(BASE, cfg["gbif"]))

    base_rounded = [(round(a, 4), round(b, 4)) for a, b in base_pts]
    gbif_rounded = [(round(a, 4), round(b, 4)) for a, b in gbif_pts]

    # manual exclusions (implausible outliers identified by inspection; see
    # MANUAL_EXCLUSIONS above for the reasoning behind each)
    kept = []
    for pt in base_rounded:
        if (sp, pt[0], pt[1]) in EXCLUDE_COORDS:
            reason = next(e["reason"] for e in MANUAL_EXCLUSIONS
                          if e["species"] == sp and round(e["lat"], 4) == pt[0] and round(e["lon"], 4) == pt[1])
            removed_points.append({"species": sp, "lat": pt[0], "lon": pt[1], "source": src_file, "reason": reason})
        else:
            kept.append(pt)
    base_rounded = kept

    # duplicates within the base file itself
    seen = set()
    base_unique = []
    for pt in base_rounded:
        if pt in seen:
            removed_points.append({"species": sp, "lat": pt[0], "lon": pt[1],
                                    "source": src_file, "reason": "duplicate within base file"})
        else:
            seen.add(pt)
            base_unique.append(pt)
    base_set = set(base_unique)

    # duplicates within the GBIF file itself, and GBIF points already present in base
    seen_g = set()
    gbif_new = []
    for pt in gbif_rounded:
        if pt in seen_g:
            removed_points.append({"species": sp, "lat": pt[0], "lon": pt[1],
                                    "source": cfg["gbif"], "reason": "duplicate within GBIF pull"})
            continue
        seen_g.add(pt)
        if pt in base_set:
            removed_points.append({"species": sp, "lat": pt[0], "lon": pt[1],
                                    "source": cfg["gbif"], "reason": "duplicate of existing base record"})
        else:
            gbif_new.append(pt)
            added_points.append({"species": sp, "lat": pt[0], "lon": pt[1], "source": cfg["gbif"]})

    final_set = base_set | set(gbif_new)
    all_final[sp] = sorted(final_set)

    log_rows.append({
        "species": sp, "base_source_file": src_file, "base_raw_rows": len(base_pts),
        "base_unique_after_dedup": len(base_set), "base_duplicates_removed": len(base_pts) - len(base_set),
        "gbif_source_file": cfg["gbif"], "gbif_raw_rows": len(gbif_pts),
        "gbif_genuinely_new_points_added": len(gbif_new), "final_unique_points": len(final_set),
    })

log_df = pd.DataFrame(log_rows)
removed_df = pd.DataFrame(removed_points)
added_df = pd.DataFrame(added_points)
manual_reasons = {e["reason"] for e in MANUAL_EXCLUSIONS}
outliers_df = removed_df[removed_df["reason"].isin(manual_reasons)].reset_index(drop=True)
duplicates_df = removed_df[~removed_df["reason"].isin(manual_reasons)].reset_index(drop=True)

print(log_df.to_string(index=False))

with pd.ExcelWriter(f"{OUT}/cleaned_coordinates_master.xlsx", engine="openpyxl") as writer:
    log_df.to_excel(writer, sheet_name="Merge_log", index=False)
    for sp, pts in all_final.items():
        d = pd.DataFrame(pts, columns=["Lat_DD", "Lon_DD"])
        d.insert(0, "Species", sp.replace("_", " "))
        d.to_excel(writer, sheet_name=sp[:31], index=False)

with pd.ExcelWriter(f"{OUT}/removed_and_added_points.xlsx", engine="openpyxl") as writer:
    outliers_df.to_excel(writer, sheet_name="Manually_excluded_outliers", index=False)
    duplicates_df.to_excel(writer, sheet_name="Removed_duplicates", index=False)
    added_df.to_excel(writer, sheet_name="Newly_added_from_GBIF", index=False)
    log_df.to_excel(writer, sheet_name="Summary_counts", index=False)

print(f"\nSaved cleaned_coordinates_master.xlsx and removed_and_added_points.xlsx to {OUT}")
for sp, pts in all_final.items():
    print(sp, len(pts))
