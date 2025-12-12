"""
writer.py
---------

Write experiment results to a single CSV file.

Input structure:
    experiment_results = {
        (ecosystem, site): {
            (theta_index, D): aggregated_dict,
            ...
        },
        ...
    }

Output:
    One flat CSV file with each row representing (site, theta_index, D).
"""

import pandas as pd


def write_results_to_csv(experiment_results, out_path):
    """
    Flatten experiment_results into a single DataFrame and write to CSV.

    Parameters
    ----------
    experiment_results : dict
        Results produced by run_experiment().
    out_path : str or Path
        Output path for the CSV file.
    """

    rows = []

    for (ecosystem, site), site_dict in experiment_results.items():
        for (theta_index, D), metrics in site_dict.items():

            # metrics is already a flat dict of all QC, flux, θ params, etc.
            row = metrics.copy()

            # Ensure identifying metadata is present
            row["ecosystem"] = ecosystem
            row["site"] = site
            row["theta_index"] = theta_index
            row["D"] = D

            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort output for readability
    df = df.sort_values(["ecosystem", "site", "theta_index", "D"])

    df.to_csv(out_path, index=False)
    print(f"[writer] Results saved to {out_path}")

    return df
