import os
from pathlib import Path
import sys
import argparse

# Ensure repo root is on sys.path when invoked as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minspecs_simulation.io_icos import cache_csv_to_npz, DEFAULT_DATA_ROOT, DEFAULT_CACHE_ROOT

SKIP_SITE = Path('igbp_EBF/FR-Pue')


def main(raw_root: Path, cache_root: Path, overwrite: bool = False, report_every: int = 500, include_fr_pue: bool = False):
    total = converted = skipped = failed = 0

    # Discover sites up front for progress context
    sites = []
    for eco_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        for site_dir in sorted([p for p in eco_dir.iterdir() if p.is_dir()]):
            sites.append((eco_dir.name, site_dir.name, site_dir))
    site_count = len(sites)

    for idx, (eco, site, site_dir) in enumerate(sites, 1):
        if (not include_fr_pue) and (eco, site) == tuple(SKIP_SITE.parts):
            print(f"[skip-site] {eco}/{site} ({idx}/{site_count})")
            continue

        print(f"[site] {eco}/{site} ({idx}/{site_count})")

        for csv_path in sorted(site_dir.rglob('*.csv')):
            total += 1
            try:
                target = cache_csv_to_npz(csv_path, raw_root=raw_root, cache_root=cache_root, overwrite=overwrite)
                if target.exists() and not overwrite and target.stat().st_size > 0:
                    skipped += 1  # existing cache file
                    continue
                converted += 1
            except Exception as e:
                failed += 1
                print(f"[warn] failed {csv_path}: {e}")

            if report_every and total % report_every == 0:
                print(f"[progress] processed {total} files (converted={converted}, skipped={skipped}, failed={failed})", flush=True)

    print(f"[done] total={total}, converted={converted}, skipped={skipped}, failed={failed}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build NPZ cache mirroring ICOS CSV tree (resumable by default).')
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_DATA_ROOT, help='Root of ICOS CSV data')
    parser.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT, help='Output cache root for NPZ files')
    parser.add_argument('--overwrite', action='store_true', help='Rewrite existing NPZ files')
    parser.add_argument('--report-every', type=int, default=500, help='Progress report interval (files)')
    parser.add_argument('--include-fr-pue', action='store_true', help='Process FR-Pue (otherwise skipped)')
    args = parser.parse_args()

    main(
        raw_root=args.raw_root,
        cache_root=args.cache_root,
        overwrite=args.overwrite,
        report_every=args.report_every,
        include_fr_pue=args.include_fr_pue,
    )
