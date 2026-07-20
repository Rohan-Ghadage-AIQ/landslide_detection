# Landslide Early-Warning Pipeline — Pune District

A complete, runnable Python pipeline for landslide susceptibility modeling
plus a ~1-week-ahead early-warning trigger layer, as described in the
accompanying R&D documents. Tested end-to-end on synthetic demo data (see
"Quick Start" below) — verified to run without errors on this environment
before being handed over.

## What's actually validated vs. what isn't

- **Validated**: every module below runs, imports correctly, and produces
  output, using `generate_synthetic_data.py`'s synthetic DEM/inventory/
  rainfall. The full `main.py` run (data → features → training table →
  spatial CV → final model → susceptibility raster → historical backtest)
  completes successfully.
- **NOT validated**: the model has never seen real Pune terrain, real
  historical landslide locations, or real rainfall records. Predictive
  performance numbers you'll see on synthetic data (e.g. PR-AUC = 1.0) are
  an artifact of how clean/small the synthetic data is — they say nothing
  about real-world performance. Treat this as a tested scaffold, not a
  validated model.

## Project structure

```
config.py                  All tunable parameters (paths, CRS, thresholds)
utils_raster.py             Raster I/O + terrain derivatives (slope, TWI, SPI, flow accumulation)
feature_engineering.py      Builds the full static feature stack + API time series
sampling.py                 Extracts training points, generates pseudo-absences
spatial_cv.py                Spatial block cross-validation (avoids autocorrelation bias)
train_model.py               RF / XGBoost training, SMOTE, threshold optimization
early_warning.py             Susceptibility + rainfall-trigger fusion -> risk level
validate_historical.py       Backtests the warning logic against a known past event
generate_synthetic_data.py   Demo/test data generator (NOT real data)
main.py                      Runs the whole pipeline end-to-end
requirements.txt
```

## Quick start (synthetic demo — runs immediately, no real data needed)

```bash
pip install -r requirements.txt
python main.py
```

This will generate a small synthetic hillslope, an inventory of synthetic
"landslide" points on its steep flank, and a synthetic rainfall record with
a monsoon-like spike — then run the full pipeline, including a backtest
against a synthetic historical event, and print the day-by-day risk trace
for the week before it.

## Switching to real data

1. Populate the paths in `config.py` (`RAW_DIR` section) with your real,
   already-harmonized (common CRS + resolution) rasters:
   - `dem.tif`, `ndvi_baseline.tif` (optional), and any other layers you
     want to add to `feature_engineering.build_static_feature_stack`.
   - `landslide_inventory.shp` — point geometries, in the same CRS as the
     rasters.
   - `daily_rainfall.csv` — two columns: date, rainfall_mm.
2. Set `USE_SYNTHETIC_DATA = False` in `main.py`.
3. Fill in `load_real_data()` in `main.py` with your real historical event
   (`event_x`, `event_y`, `event_date`) for backtesting — or write a loop
   to backtest against your entire inventory instead of just one event.
4. Re-run `python main.py`.

## Extending the feature set

`utils_raster.py` and `feature_engineering.py` currently implement:
slope, aspect, curvature, terrain ruggedness, D8 flow accumulation, TWI,
SPI, distance-to-drainage, and NDVI (if supplied). To add lithology, soil,
land-cover, or distance-to-roads/faults, harmonize those layers to the same
grid and add them to the `feature_stack` dict in `build_static_feature_stack`
plus `config.FEATURE_COLUMNS`.

## Known limitations to flag before production use

- The D8 flow-accumulation implementation in `utils_raster.py` is pure
  numpy and adequate for demo-sized grids; for a full 10 m district raster
  (~150M+ pixels) it will be slow — swap in `richdem`, `pysheds`, or
  `whitebox_tools` for a production run.
- The `early_warning.py` fusion formula (susceptibility × rainfall-trigger
  score) is a transparent starting formula, not a fitted/learned function.
  It should be recalibrated against real forecast-vs-outcome data.
- `validate_historical.py`'s backtest uses REALIZED rainfall as a stand-in
  for the forecast that would have been issued — this assumes a perfect
  forecast and will overstate real-world lead-time reliability. Re-run
  with actual archived IMD forecasts once available.
- Spatial cross-validation falls back to a plain 80/20 split when there
  aren't enough distinct spatial blocks (as with the tiny synthetic demo
  data) — on real, district-scale data with many inventory points spread
  over a wide area, the proper spatial block CV will engage automatically.

## Nationwide / multi-region pipeline (main_national.py)

`main.py` models ONE district (a single fixed AOI). If the goal is a
model that can be pointed at ANY location in the country -- not just
Pune -- that's a different, harder problem: terrain, geology, and
rainfall regime genuinely differ across regions (Western Ghats vs.
Himalayan foothills vs. Kerala laterite hills), so a model trained only
on Pune-area examples should not be assumed to transfer elsewhere.

`main_national.py` + `multi_site_pipeline.py` + `multi_site_data.py` +
`geo_utils.py` implement the nationwide version of the workflow:

- **No single AOI polygon.** Instead, a small local tile (a few km
  radius) is pulled around EACH known historical event, wherever it
  happened.
- **`geo_utils.get_utm_epsg(lon, lat)`** picks the correct local UTM zone
  per event, so terrain features come out in consistent real-world units
  (metres) regardless of which part of India the event is in.
- **Each event's local tile is its own spatial-CV group.** This enables
  *leave-one-region-out* cross-validation
  (`multi_site_pipeline.leave_one_region_out_cv`): train on all other
  regions, test on a region the model never saw. This is the honest way
  to test a "works anywhere" claim -- it directly measures geographic
  generalization, not just interpolation within one district's terrain.
- **The historical backtest (`validate_historical.py`) is reused
  unchanged** per site, since it was already written to work off any
  point + local feature stack, not tied to Pune specifically.

Run the synthetic demo (4 illustrative regions: Western Ghats/Pune,
Kerala/Wayanad, Uttarakhand Himalaya, Sikkim Himalaya):

```bash
python main_national.py
```

### Switching to real nationwide data

1. Build a real event inventory: one row per known landslide, with at
   minimum `lon, lat, date`. GSI's National Landslide Susceptibility
   Mapping inventory and NASA's COOLR global catalog are the two
   commonly-referenced sources for India — confirm current access terms
   directly, as I can't verify live download availability here.
2. For each event, obtain a local DEM covering that point (e.g. a
   Copernicus GLO-30 or Cartosat tile downloaded for that specific
   bounding box) rather than one countrywide raster — countrywide 10 m
   DEM data is enormous and unnecessary if you're only ever sampling
   small windows around known/candidate points.
3. Replace `multi_site_data.generate_national_demo_sites()` with a loader
   that reads your real event inventory + opens/crops the matching local
   DEM tile per event (a windowed `rasterio` read, reprojected to the
   zone from `geo_utils.get_utm_epsg`).
4. With only 1 example per region (as in the demo), leave-one-region-out
   CV is not statistically meaningful — you need multiple events per
   region before those numbers mean anything. Treat the demo's CV output
   as proof the *mechanism* works, not as a performance claim.
5. For actual deployment inference on a NEW, un-labeled location, run the
   trained national model's `.predict_proba()` against that location's
   own local feature stack the same way `validate_historical.py` extracts
   a feature vector at a point — there's no separate "prediction AOI"
   needed beyond wherever you want an answer for.
#   l a n d s l i d e _ d e t e c t i o n  
 