"""
audit_label_vocab.py
=====================
Answers "which SNOMED-CT classes actually have enough real data to be
worth adding to the model?" — using ONLY real numbers from your dataset,
never a guessed or assumed code list.

*** No SNOMED-CT codes are hardcoded anywhere in this file. ***
This is deliberate: the only codes this project can currently trust are
the 5 already verified against real preprocessing-run logs (SB, AFIB, SR,
ST, SVT). Guessing codes for the other ~22 candidate classes (RBBB, LBBB,
AV blocks, STEMI, etc.) risks silently auditing the WRONG codes, which
would defeat the entire purpose of this script. Instead:

  Step 1 (`--mode discover`) needs no codes from you at all. It scans
  every record in your real dataset, counts every SNOMED-CT code that
  actually appears, resolves each to its human-readable name via your
  own ConditionNames_SNOMED-CT.csv, and prints/saves the full list
  sorted by raw mention count. You read the codes off THIS output,
  confirmed against your own reference file — not off anything I typed.

  Step 2 (`--mode simulate --codes CODE1,CODE2,...`) takes the exact
  codes you choose from Step 1's output and simulates the REAL primary-
  label assignment outcome for that exact class set, using the same
  assign_label() logic the real preprocessing pipeline uses — not raw
  mention count, which is the metric that already burned this project
  twice (Section 4.5 of the project report). This is what tells you,
  honestly, whether a candidate class will actually have enough support
  once single-label assignment runs, before you commit to retraining.

Usage
-----
    # Step 1 — no arguments needed beyond your data path, defaults from config
    python audit_label_vocab.py --mode discover

    # Step 2 — after reading codes off Step 1's output
    python audit_label_vocab.py --mode simulate --codes 426177001,164889003,426783006,427084000,426761007,59118001
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import Counter

from config_physionet import PhysioNetConfig as PNConfig
from utils.preprocessing_physionet import (
    assign_label,
    find_all_records,
    load_snomed_names,
    _normalize_dx_codes,
    get_dx_codes,
    CODE_MERGE_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def discover(data_path: str, condition_names_csv: str, output_csv: str) -> None:
    """
    Scan the ENTIRE dataset, tally every SNOMED-CT code that actually
    appears (after CODE_MERGE_MAP normalisation), and print/save all of
    them sorted by raw mention count, with resolved names.

    This is deliberately the exact raw-mention-count metric that misled
    this project before — that's the point. Comparing THIS column
    against Step 2's real-assignment column, side by side, is what makes
    the "raw count lies" problem visible and checkable, not just
    asserted.
    """
    logger.info("Discovering records under %s ...", data_path)
    record_paths = find_all_records(data_path)
    logger.info("Found %d records.", len(record_paths))

    logger.info("Loading SNOMED-CT name table from %s ...", condition_names_csv)
    snomed_names = load_snomed_names(condition_names_csv)
    logger.info("Loaded %d code -> name mappings.", len(snomed_names))

    counter: Counter = Counter()
    unresolved_codes = set()
    n_no_dx = 0

    for i, rp in enumerate(record_paths, 1):
        try:
            codes = get_dx_codes(rp)
        except Exception as exc:
            logger.warning("Could not read header for %s: %s", rp, exc)
            continue
        if not codes:
            n_no_dx += 1
            continue
        codes = _normalize_dx_codes(codes)
        counter.update(codes)
        for c in codes:
            if c not in snomed_names:
                unresolved_codes.add(c)
        if i % 2000 == 0:
            logger.info("  ...%d / %d records scanned", i, len(record_paths))

    logger.info("Scan complete. %d records had no Dx codes.", n_no_dx)
    if unresolved_codes:
        logger.warning(
            "%d distinct codes had NO entry in your SNOMED-CT name table "
            "(shown as '?' below) — these still have real counts, just no "
            "resolved name. Check your ConditionNames_SNOMED-CT.csv if you "
            "want names for them: %s",
            len(unresolved_codes), sorted(unresolved_codes),
        )

    rows = [
        {"code": code, "name": snomed_names.get(code, "?"), "raw_mention_count": count}
        for code, count in counter.most_common()
    ]

    print("\n" + "=" * 78)
    print(f"{'Code':<14}{'Raw mentions':>14}   Name")
    print("=" * 78)
    for row in rows:
        print(f"{row['code']:<14}{row['raw_mention_count']:>14}   {row['name']}")
    print("=" * 78)
    print(
        f"\n{len(rows)} distinct codes found across {len(record_paths)} records.\n"
        f"IMPORTANT: 'Raw mentions' counts a code every time it appears anywhere in a "
        f"record's Dx list, including as a secondary/comorbid diagnosis. It does NOT "
        f"predict how many records will get that code as their PRIMARY label — see "
        f"Step 2 ('--mode simulate') before committing to any of these."
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["code", "name", "raw_mention_count"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Full table saved -> %s", output_csv)


def simulate(data_path: str, condition_names_csv: str, codes: list, min_count: int, output_json: str) -> None:
    """
    Simulate the REAL primary-label assignment outcome for exactly the
    given code list — i.e. what would actually happen if you retrained
    with this class set, without needing to actually retrain.
    """
    logger.info("Discovering records under %s ...", data_path)
    record_paths = find_all_records(data_path)

    logger.info("Loading SNOMED-CT name table from %s ...", condition_names_csv)
    snomed_names = load_snomed_names(condition_names_csv)

    code_to_idx = {code: i for i, code in enumerate(codes)}
    class_names = [snomed_names.get(c, c) for c in codes]

    logger.info("Simulating real primary-label assignment for %d candidate classes...", len(codes))
    assigned_counts = Counter()
    n_unassigned = 0

    for i, rp in enumerate(record_paths, 1):
        try:
            dx_codes = get_dx_codes(rp)
        except Exception:
            continue
        label = assign_label(dx_codes, code_to_idx)
        if label is None:
            n_unassigned += 1
        else:
            assigned_counts[label] += 1
        if i % 2000 == 0:
            logger.info("  ...%d / %d records processed", i, len(record_paths))

    results = []
    for idx, code in enumerate(codes):
        count = assigned_counts.get(idx, 0)
        results.append({
            "code": code,
            "name": class_names[idx],
            "real_assigned_count": count,
            "meets_threshold": count >= min_count,
        })

    print("\n" + "=" * 88)
    print(f"{'Code':<14}{'Name':<32}{'Real assigned count':>20}   Meets threshold ({min_count})?")
    print("=" * 88)
    for r in results:
        flag = "YES" if r["meets_threshold"] else "NO  <-- too few real samples"
        print(f"{r['code']:<14}{r['name']:<32}{r['real_assigned_count']:>20}   {flag}")
    print("=" * 88)

    n_pass = sum(r["meets_threshold"] for r in results)
    n_fail = len(results) - n_pass
    total_assigned = sum(r["real_assigned_count"] for r in results)
    print(
        f"\n{n_pass} of {len(results)} candidate classes clear the {min_count}-sample "
        f"threshold. {n_fail} do not — dropping or merging these is worth considering "
        f"before retraining, exactly as SB/AFIB/SR/ST/SVT's 3 weak siblings were dropped "
        f"and AF was merged into AFIB previously.\n"
        f"Total records that would be assigned a label: {total_assigned} / {len(record_paths)} "
        f"({n_unassigned} records have no Dx code in this candidate set at all, and would "
        f"be excluded from the dataset — same as how records outside the class vocabulary "
        f"are already excluded today)."
    )

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump({
            "candidate_codes": codes,
            "min_count_threshold": min_count,
            "results": results,
            "total_records_scanned": len(record_paths),
            "total_records_assigned": total_assigned,
            "total_records_unassigned": n_unassigned,
        }, fh, indent=2)
    logger.info("Full results saved -> %s", output_json)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit which SNOMED-CT classes have real support in your dataset.")
    parser.add_argument("--mode", choices=["discover", "simulate"], required=True)
    parser.add_argument("--codes", type=str, default=None,
                         help="Comma-separated SNOMED-CT codes to simulate (required for --mode simulate). "
                              "Get these from --mode discover's output, not from anywhere else.")
    parser.add_argument("--min_count", type=int, default=300,
                         help="Minimum real assigned-sample count for a class to be considered usable. "
                              "Your 5 current classes range 602-4000; classes below ~200-300 are unlikely "
                              "to train well even with the weighted sampler.")
    parser.add_argument("--data_path", type=str, default=PNConfig.DATA_PATH)
    parser.add_argument("--condition_names_csv", type=str, default=PNConfig.CONDITION_NAMES_CSV)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "discover":
        output_csv = os.path.join(PNConfig.PLOTS_PATH, "label_audit", "all_codes_raw_counts.csv")
        discover(args.data_path, args.condition_names_csv, output_csv)
    else:
        if not args.codes:
            raise ValueError("--mode simulate requires --codes (comma-separated SNOMED-CT codes).")
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        output_json = os.path.join(PNConfig.PLOTS_PATH, "label_audit", "simulated_assignment.json")
        simulate(args.data_path, args.condition_names_csv, codes, args.min_count, output_json)


if __name__ == "__main__":
    main()