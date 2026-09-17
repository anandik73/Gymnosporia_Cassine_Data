"""
Step 2: Extract all 19 CHELSA v2.1 bioclimatic variables (30 arc-second, ~1 km
resolution) at every cleaned occurrence point from Step 1.

CHELSA rasters are read directly from the public remote source via GDAL's
/vsicurl/ virtual file system (no local download of full global rasters
required), cropped to an India-extent window, and the correct per-variable
scale/offset/nodata (as stored in each file's own metadata) is applied -
this matters: CHELSA temperature variables are stored as Kelvin*10 with a
-273.15 offset, while other variables use different scale/offset values, and
these must be read from each file's metadata rather than assumed uniform.

Requires: rasterio, pandas, openpyxl. Internet access to
os.zhdk.cloud.switch.ch (CHELSA's public host).

Input : Results/cleaned_coordinates_master.xlsx (from Step 1)
Output: Results/occurrence_climate_CHELSA.csv
        chelsa_rasters/CHELSA_bioN_india.tif  (cached local crops, 19 files)
"""
import json
import os
import openpyxl
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "Results")
RASTER_DIR = os.path.join(HERE, "chelsa_rasters")
os.makedirs(RASTER_DIR, exist_ok=True)

# India study extent (matches Rodgers et al. 2002 biogeographic classification
# used in the original species descriptions), with a small buffer
MINLON, MAXLON, MINLAT, MAXLAT = 67, 98, 7, 38
CHELSA_URL = "/vsicurl/https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/CHELSA_{var}_1981-2010_V.2.1.tif"


def crop_chelsa_to_india():
    """Download (once) a cropped India-extent copy of each of the 19 bioclim
    rasters. Subsequent runs reuse the local cache in chelsa_rasters/."""
    scale_offset = {}
    for i in range(1, 20):
        var = f"bio{i}"
        outpath = os.path.join(RASTER_DIR, f"CHELSA_{var}_india.tif")
        url = CHELSA_URL.format(var=var)
        with rasterio.open(url) as ds:
            scale_offset[var] = (ds.scales[0], ds.offsets[0], ds.nodata)
            if not os.path.exists(outpath):
                window = from_bounds(MINLON, MINLAT, MAXLON, MAXLAT, ds.transform)
                data = ds.read(1, window=window)
                transform = ds.window_transform(window)
                profile = ds.profile.copy()
                profile.update({"height": data.shape[0], "width": data.shape[1], "transform": transform})
                with rasterio.open(outpath, "w", **profile) as dst:
                    dst.write(data, 1)
                print(f"{var}: cropped and cached")
            else:
                print(f"{var}: using cached local crop")
    with open(os.path.join(RASTER_DIR, "scale_offset.json"), "w") as f:
        json.dump(scale_offset, f)
    return scale_offset


def load_occurrence_points():
    wb = openpyxl.load_workbook(os.path.join(OUT, "cleaned_coordinates_master.xlsx"))
    rows = []
    for sheet in wb.sheetnames:
        if sheet == "Merge_log":
            continue
        ws = wb[sheet]
        for r in ws.iter_rows(values_only=True, min_row=2):
            rows.append({"species": sheet, "lat": r[1], "lon": r[2]})
    return pd.DataFrame(rows)


def extract_climate(df, scale_offset):
    for i in range(1, 20):
        var = f"bio{i}"
        scale, offset, nodata = scale_offset[var]
        path = os.path.join(RASTER_DIR, f"CHELSA_{var}_india.tif")
        with rasterio.open(path) as ds:
            vals = []
            for lat, lon in zip(df["lat"], df["lon"]):
                row, col = ds.index(lon, lat)
                arr = ds.read(1, window=((row, row + 1), (col, col + 1)))
                raw = float(arr[0, 0])
                v = np.nan if (nodata is not None and raw == nodata) else raw * scale + offset
                vals.append(v)
            df[var] = vals
    return df


if __name__ == "__main__":
    scale_offset = crop_chelsa_to_india()
    df = load_occurrence_points()
    print("Total occurrence points:", len(df))
    df = extract_climate(df, scale_offset)

    biocols = [f"bio{i}" for i in range(1, 20)]
    print("Missing values per variable:\n", df[biocols].isna().sum())

    df.to_csv(f"{OUT}/occurrence_climate_CHELSA.csv", index=False)
    print(f"\nSaved occurrence_climate_CHELSA.csv, shape {df.shape}")
