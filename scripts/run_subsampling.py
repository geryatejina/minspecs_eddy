import argparse
import os
from pathlib import Path

from minspecs_simulation.main import run_subsampling_experiment
from minspecs_simulation.types import SubsampleSpec, SubsampleMode
from minspecs_simulation.writer import write_results_to_csv
from minspecs_simulation.window_processor import set_empty_log


def build_subsample_specs():
    return [
        SubsampleSpec(mode=SubsampleMode.DECIMATE, decimate_factor=2, name="decimate_2"),
        SubsampleSpec(mode=SubsampleMode.DECIMATE, decimate_factor=5, name="decimate_5"),
        SubsampleSpec(mode=SubsampleMode.DECIMATE, decimate_factor=10, name="decimate_10"),
        SubsampleSpec(
            mode=SubsampleMode.OGIVE_STOP,
            ogive_threshold=0.05,      # looser threshold => stop sooner
            ogive_trailing_sec=60.0,   # shorter trailing window to declare steady state
            ogive_min_dwell_sec=20.0,  # shorter dwell requirement
            name="ogive_stop_loose",
        ),
        SubsampleSpec(mode=SubsampleMode.BURST, burst_on_sec=5, burst_off_sec=25, name="burst_5_25"),
        SubsampleSpec(mode=SubsampleMode.BURST, burst_on_sec=10, burst_off_sec=50, name="burst_10_50"),
        SubsampleSpec(mode=SubsampleMode.BURST, burst_on_sec=10, burst_off_sec=110, name="burst_10_110"),
        SubsampleSpec(
            mode=SubsampleMode.DIURNAL,
            name="diurnal_burst_day_high_night_low",
            day_spec=SubsampleSpec(mode=SubsampleMode.BURST, burst_on_sec=10, burst_off_sec=20, name="day_burst"),
            night_spec=SubsampleSpec(mode=SubsampleMode.BURST, burst_on_sec=5, burst_off_sec=115, name="night_burst"),
        ),
        SubsampleSpec(
            mode=SubsampleMode.DIURNAL,
            name="diurnal_decimate_day_high_night_low",
            day_spec=SubsampleSpec(mode=SubsampleMode.DECIMATE, decimate_factor=2, name="day_decimate_2"),
            night_spec=SubsampleSpec(mode=SubsampleMode.DECIMATE, decimate_factor=10, name="night_decimate_10"),
        ),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run subsampling experiment.")
    parser.add_argument(
        "--empty-log",
        help="Log empty/NaN arrays: 'stderr', 'stdout', or a file path.",
    )
    args = parser.parse_args()
    if args.empty_log:
        os.environ["MINSPECS_EMPTY_LOG"] = args.empty_log
        set_empty_log(args.empty_log)

    sites = [
        ("igbp_CRO", "BE-Lon"),
        ("igbp_CRO", "DE-Geb"),
        ("igbp_CSH", "BE-Maa"),
        ("igbp_DBF", "CZ-Lnz"),
        ("igbp_DBF", "DE-HoH"),
        ("igbp_EBF", "FR-Pue"),
        ("igbp_ENF", "Be-Bra"),
        ("igbp_ENF", "CH-Dav"),
        ("igbp_GRA", "BE-Dor"),
        ("igbp_GRA", "FR-Lqu"),
        ("igbp_MF",  "BE-Vie"),
        ("igbp_MF",  "IT-Cp2"),
        ("igbp_WET", "FI-Sii"),
        ("igbp_WET", "GL-ZaF"),
    ]

    subsample_specs = build_subsample_specs()

    results_dir = Path("subsampling_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = run_subsampling_experiment(
        ecosystem_site_list=sites,
        subsample_specs=subsample_specs,
        rotation_modes=("double",),
        data_root=Path(r"D:\data\ec\raw\ICOS_npz"),
        file_pattern="*.npz",
        max_workers=8,
        max_files_per_site=None,
        skip_map=None,    
        outlier_lower_pct=7.0,
        outlier_upper_pct=93.0,
        window_log_dir=results_dir,
    )

    write_results_to_csv(results, results_dir / "subsampling_5sites_full_period.csv")
