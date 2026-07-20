"""
main_national.py
-------------------
Entry point for the NATIONWIDE version of the pipeline: trains a model
across multiple regions and validates it with leave-one-region-out
cross-validation (does it generalize to a region it never saw?) plus a
per-site historical-event backtest.

This complements, rather than replaces, main.py (the single-district
Pune pipeline). Use main.py when you want a detailed susceptibility MAP
for one district you're deploying in operationally. Use this script when
the goal is a model that can be pointed at a NEW location anywhere in
the country and still say something useful about it.

Run as-is for a synthetic multi-region demo:
    python main_national.py

For real data: see "Switching to real nationwide data" in README.md.
"""

import config
from multi_site_data import generate_national_demo_sites
from multi_site_pipeline import (
    build_national_training_table, leave_one_region_out_cv,
    train_national_model, backtest_all_sites,
)
from train_model import save_model


def main():
    print("=" * 70)
    print("NATIONWIDE LANDSLIDE EARLY-WARNING PIPELINE (multi-region)")
    print("=" * 70)

    print("\n[1/5] Generating synthetic multi-region demo sites "
          "(replace with real event inventory + local DEM tiles for "
          "production use)...")
    sites = generate_national_demo_sites()
    for s in sites:
        print(f"    - {s['site_id']}  (local CRS: {s['local_crs']}, "
              f"event date: {s['event_date'].date()})")

    print("\n[2/5] Building training rows for each region "
          "(1 positive + pseudo-absences per site)...")
    table, feature_stacks = build_national_training_table(
        sites, config.FEATURE_COLUMNS,
        pseudo_absence_ratio=config.PSEUDO_ABSENCE_RATIO,
        min_dist_m=config.PSEUDO_ABSENCE_MIN_DIST_M,
        random_state=config.RANDOM_STATE)
    available_features = [c for c in config.FEATURE_COLUMNS if c in table.columns]
    print(f"    Combined training table: {len(table)} rows across "
          f"{table['site_id'].nunique()} regions")

    print("\n[3/5] Leave-one-region-out cross-validation "
          "(train on other regions, test on a region never seen)...")
    cv_results = leave_one_region_out_cv(
        table, available_features, model_type=config.MODEL_TYPE,
        random_state=config.RANDOM_STATE)
    print(cv_results.to_string(index=False))
    print("\n    NOTE: with only ~1 labeled event per synthetic region, "
          "these numbers are illustrative of the MECHANISM only -- real "
          "generalization testing needs many events per region.")

    print("\n[4/5] Training the final national model on all regions...")
    national_model = train_national_model(
        table, available_features, model_type=config.MODEL_TYPE,
        random_state=config.RANDOM_STATE)
    save_model(national_model, config.MODEL_PATH.replace(".joblib", "_national.joblib"))
    print("    Model saved.")

    print("\n[5/5] Backtesting the trained model against each region's "
          "historical event...")
    backtest_summary = backtest_all_sites(
        sites, feature_stacks, national_model, available_features)
    print(backtest_summary[["site_id", "would_have_warned",
                             "max_lead_time_days"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("NATIONWIDE PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
