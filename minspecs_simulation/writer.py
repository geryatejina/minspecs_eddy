"""
writer.py
---------

Write experiment results to a single CSV file.

Input structure:
    experiment_results = {
        (ecosystem, site): {
            theta_index: aggregated_dict,
            ...
        },
        ...
    }

Output:
    One flat CSV file with each row representing (site, theta_index).
"""

import pandas as pd
import dataclasses
from .types import Theta


def results_to_dataframe(experiment_results):
    """
    Flatten experiment_results into a single DataFrame.

    Parameters
    ----------
    experiment_results : dict
        Results produced by run_experiment().
    """
    rows = []

    for (ecosystem, site), site_dict in experiment_results.items():
        for theta_key, metrics in site_dict.items():

            # metrics is already a flat dict of all QC, flux, theta params, etc.
            row = metrics.copy()

            # Ensure identifying metadata is present
            row["ecosystem"] = ecosystem
            row["site"] = site
            if "theta_index" not in row:
                if isinstance(theta_key, tuple):
                    row["theta_index"] = theta_key[0]
                else:
                    row["theta_index"] = theta_key

            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Reorder columns: metadata -> theta params -> everything else
    theta_fields = [f.name for f in dataclasses.fields(Theta)]
    meta_cols = [c for c in ["ecosystem", "site", "theta_index", "rotation_mode", "n_windows"] if c in df.columns]
    theta_cols = [c for c in theta_fields if c in df.columns]
    remaining = [c for c in df.columns if c not in meta_cols + theta_cols]

    ordered_cols = meta_cols + theta_cols + remaining
    df = df[ordered_cols]

    # Sort output for readability
    sort_cols = [c for c in ["ecosystem", "site", "theta_index", "rotation_mode"] if c in df.columns]
    return df.sort_values(sort_cols)


def write_results_to_csv(experiment_results, out_path):
    """
    Flatten experiment_results into a single DataFrame and write to CSV.
    """
    df = results_to_dataframe(experiment_results)
    df.to_csv(out_path, index=False)
    print(f"[writer] Results saved to {out_path}")
    return df
