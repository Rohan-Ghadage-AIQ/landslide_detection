"""
utils_raster.py
----------------
Low-level raster I/O and terrain-derivative helpers used throughout the
pipeline. Terrain derivatives (slope/aspect/curvature/flow accumulation)
are implemented directly in numpy so the pipeline has no hard dependency
on richdem/pysheds/GDAL command-line tools -- easier to install and run
end-to-end, at the cost of being slower on very large rasters. For a
production district-scale run, swap `flow_accumulation_d8` for a proper
library (richdem, pysheds, or whitebox_tools) which use far faster
algorithms.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


# ---------------------------------------------------------------------------
# Basic I/O
# ---------------------------------------------------------------------------
def read_raster(path, band=1):
    """Returns (array, meta_dict) for a single band."""
    with rasterio.open(path) as src:
        arr = src.read(band).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        meta = src.meta.copy()
    return arr, meta


def write_raster(path, array, meta, dtype="float32", nodata=np.nan):
    meta = meta.copy()
    meta.update(dtype=dtype, count=1, nodata=nodata)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array.astype(dtype), 1)


def harmonize_raster(src_path, dst_path, dst_crs, dst_res,
                      resampling=Resampling.bilinear):
    """Reproject + resample a single raster onto the common analysis grid."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=dst_res)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform,
                        "width": width, "height": height})
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=dst_crs,
                    resampling=resampling)
    return dst_path


# ---------------------------------------------------------------------------
# Terrain derivatives (pure numpy)
# ---------------------------------------------------------------------------
def compute_slope_aspect(dem, cell_size):
    """Horn's method. dem: 2D array (metres). Returns slope_deg, aspect_deg."""
    dzdx = np.gradient(dem, axis=1) / cell_size
    dzdy = np.gradient(dem, axis=0) / cell_size
    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 90.0 - aspect_deg, 90.0 - aspect_deg)
    aspect_deg = np.mod(aspect_deg, 360.0)
    return slope_deg, aspect_deg


def compute_curvature(dem, cell_size):
    """Simple profile curvature approximation via second derivatives."""
    dzdx = np.gradient(dem, axis=1) / cell_size
    dzdy = np.gradient(dem, axis=0) / cell_size
    d2zdx2 = np.gradient(dzdx, axis=1) / cell_size
    d2zdy2 = np.gradient(dzdy, axis=0) / cell_size
    curvature = d2zdx2 + d2zdy2
    return curvature.astype("float32")


def compute_terrain_ruggedness(dem):
    """Mean absolute elevation difference to the 8 neighbours."""
    padded = np.pad(dem, 1, mode="edge")
    tri = np.zeros_like(dem)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[1 + dy: 1 + dy + dem.shape[0],
                              1 + dx: 1 + dx + dem.shape[1]]
            tri += np.abs(dem - shifted)
    return (tri / 8.0).astype("float32")


def flow_accumulation_d8(dem):
    """
    Simple D8 flow accumulation.

    Routes each cell's unit flow to its single steepest downslope neighbour,
    then accumulates in order from highest to lowest elevation. This is a
    standard, well-documented algorithm -- adequate for small-to-medium
    demo/research grids. For a full district at 10 m resolution, use a
    dedicated library (richdem.FlowAccumulation, pysheds, whitebox_tools)
    for speed and depression-filling robustness.
    """
    rows, cols = dem.shape
    flat = dem.ravel()
    order = np.argsort(-flat)  # highest elevation first

    # 8 neighbour offsets, D8
    neighbour_offsets = [(-1, -1), (-1, 0), (-1, 1),
                          (0, -1),           (0, 1),
                          (1, -1),  (1, 0),  (1, 1)]

    accum = np.ones(rows * cols, dtype="float64")  # each cell contributes 1 unit
    receiver = np.full(rows * cols, -1, dtype="int64")

    for idx in order:
        r, c = divmod(idx, cols)
        best_drop = 0.0
        best_idx = -1
        for dr, dc in neighbour_offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                n_idx = nr * cols + nc
                drop = dem[r, c] - dem[nr, nc]
                dist = np.hypot(dr, dc)
                slope_drop = drop / dist
                if slope_drop > best_drop:
                    best_drop = slope_drop
                    best_idx = n_idx
        receiver[idx] = best_idx

    for idx in order:
        rcv = receiver[idx]
        if rcv != -1:
            accum[rcv] += accum[idx]

    return accum.reshape(rows, cols).astype("float32")


def compute_twi(slope_deg, flow_accum, cell_size):
    slope_rad = np.radians(np.clip(slope_deg, 0.1, None))  # avoid tan(0)
    specific_catchment_area = flow_accum * cell_size
    return np.log(specific_catchment_area / np.tan(slope_rad)).astype("float32")


def compute_spi(slope_deg, flow_accum, cell_size):
    slope_rad = np.radians(slope_deg)
    specific_catchment_area = flow_accum * cell_size
    return (specific_catchment_area * np.tan(slope_rad)).astype("float32")


def distance_to_features(mask, cell_size):
    """
    Euclidean distance (metres) from every cell to the nearest True cell
    in `mask` (e.g. drainage cells, road cells).
    """
    from scipy.ndimage import distance_transform_edt
    dist_px = distance_transform_edt(~mask)
    return (dist_px * cell_size).astype("float32")


def compute_ndvi(nir_band, red_band):
    return ((nir_band - red_band) / (nir_band + red_band + 1e-6)).astype("float32")
