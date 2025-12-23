from pathlib import Path

from minspecs_simulation.main import run_subsampling_experiment
from minspecs_simulation.types import SubsampleSpec, SubsampleMode
from minspecs_simulation.writer import write_results_to_csv


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
    sites = [
        ("igbp_CRO", "BE-Lon"),
        ("igbp_DBF", "CZ-Lnz"),
        ("igbp_GRA", "BE-Dor"),
        ("igbp_WET", "FI-Sii"),
        ("igbp_MF",  "BE-Vie"),
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
        outlier_lower_pct=3.0,
        outlier_upper_pct=97.0,
        window_log_dir=results_dir,
    )

    write_results_to_csv(results, results_dir / "subsampling_3sites_full_period.csv")
