"""Aggregate CoordBench sweep CSVs into dataset-macro means and standard deviations.

The JSON input manifest lists {"path": "fixed0.csv", "seed": 0} entries. Paths
are relative to the manifest; globs can set "expected_files" for shard counts.
Penalty sweeps also require a "source" model name.

Regression scores are floored at -1 before averaging tasks within datasets.
Methods and seeds share an eligible dataset cohort at each block size; buffered
exclusion shares one cohort across all radii. Standard deviations are across fold assignments.
"""

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from scripts.paper.white_compare import family_of, floor_r2

CELLS = (0.0, 2.0, 10.0, 20.0, 40.0)
RADII = (0.0, 10.0, 25.0, 50.0, 100.0)


def saturated(rows: list[dict]) -> set[tuple[str, float]]:
    """Identify unchanged positive block-size steps from recorded distance strings."""
    seen = defaultdict(dict)
    for row in rows:
        if row.get("median_nn_train_km", "") != "":
            seen[row["benchmark"]][float(row["cell_deg"])] = row["median_nn_train_km"]
    return {
        (benchmark, current)
        for benchmark, cells in seen.items()
        for previous, current in pairwise(sorted(cells))
        if previous > 0 and cells[current] == cells[previous]
    }


def load_inputs(manifest: Path, seeds: tuple[int, ...], buffered: bool = False) -> list[dict]:
    """Load explicit files, select matched inner folds, and normalize method names."""
    specs = json.loads(manifest.read_text())
    if not isinstance(specs, list) or not specs:
        raise ValueError("the manifest must contain a nonempty list of input files")
    groups = defaultdict(list)
    for spec in specs:
        seed = int(spec["seed"])
        if seed not in seeds:
            continue
        paths = sorted(glob.glob(str(manifest.parent / spec["path"])))
        expected = spec.get("expected_files")
        if not paths or (expected is not None and len(paths) != expected):
            raise ValueError(f"incomplete input {spec['path']}: found {len(paths)} files")
        for path in paths:
            with Path(path).open() as file:
                rows = list(csv.DictReader(file))
            if not rows:
                raise ValueError(f"empty input: {path}")
            kind = "buffer" if "radius_km" in rows[0] else "block"
            if (kind == "buffer") != buffered:
                raise ValueError(f"{path} does not match --buffered={buffered}")
            if "rule" in rows[0] and not spec.get("source"):
                raise ValueError(f"penalty input {path} needs a source name")
            groups[seed, spec.get("source", ""), "rule" in rows[0]].extend(rows)
    if {key[0] for key in groups} != set(seeds):
        raise ValueError("the manifest does not cover every requested fold seed")

    output = []
    for (seed, source, _penalty), rows in groups.items():
        bad = set() if buffered else saturated(rows)
        for row in rows:
            benchmark = row["benchmark"]
            if benchmark.startswith("bt-") or row.get("score") in ("", "None", None):
                continue
            if "fold_seed" in row and int(row["fold_seed"]) != seed:
                raise ValueError("manifest seed disagrees with the CSV fold_seed")
            value = float(row["radius_km"] if buffered else row["cell_deg"])
            if (benchmark, value) in bad:
                continue
            if "rule" in row:
                if row["rule"] != ("nested" if value == 0 else "nested_spatial"):
                    continue
                method = f"{source}/{row['config']}"
            elif buffered:
                method = f"{row['encoder']}/{row['config']}/{row['mode']}"
            elif "k" in row:
                method = f"coord_idw/k{row['k']}"
            else:
                method = row["encoder"]
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite score for {benchmark}, {method}")
            output.append(
                {
                    "seed": seed,
                    "method": method,
                    "task_type": row["task_type"],
                    "value": value,
                    "benchmark": benchmark,
                    "task": row["task"],
                    "family": row.get("family") or family_of(benchmark),
                    "score": floor_r2(score) if row["task_type"] == "regression" else score,
                    "median_nn_km": float(row["median_nn_km"]) if buffered else None,
                }
            )
    if buffered:
        # At zero radius the matched random-deletion control is identical to the
        # buffered run, so the sweep writes the baseline once.
        output.extend(
            {**row, "method": row["method"].removesuffix("/buffer") + "/random_drop"}
            for row in list(output)
            if row["value"] == 0 and row["method"].endswith("/buffer")
        )
    return output


def summarize(
    rows: list[dict],
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    values: tuple[float, ...] = CELLS,
    methods: tuple[str, ...] | None = None,
    categories: bool = False,
    buffered: bool = False,
    cohort: dict[tuple[str, float], set[str]] | None = None,
) -> list[dict]:
    """Return means, sample standard deviations, seed scores, and exact cohorts."""
    methods = methods or tuple(sorted({row["method"] for row in rows}))
    missing = set(methods) - {row["method"] for row in rows}
    if missing:
        raise ValueError(f"methods absent from inputs: {sorted(missing)}")
    tasks = {}
    families = {}
    for row in rows:
        if row["method"] not in methods or row["value"] not in values or row["seed"] not in seeds:
            continue
        key = (
            row["seed"],
            row["method"],
            row["task_type"],
            row["value"],
            row["benchmark"],
            row["task"],
        )
        if key in tasks and tasks[key] != row["score"]:
            raise ValueError(f"conflicting duplicate task result: {key}")
        tasks[key] = row["score"]
        families[row["benchmark"]] = row["family"]
    buckets = defaultdict(list)
    for (*key, _task), score in tasks.items():
        buckets[tuple(key)].append(score)
    benchmarks = defaultdict(dict)
    for (*key, benchmark), scores in buckets.items():
        benchmarks[tuple(key)][benchmark] = statistics.fmean(scores)

    distances = {
        (row["seed"], row["value"], row["benchmark"]): row["median_nn_km"]
        for row in rows
        if buffered and row["method"].endswith("/buffer")
    }
    output = []
    for task_type in sorted({key[2] for key in benchmarks}):
        common = {}
        for value in values:
            common[value] = set.intersection(
                *(
                    set(benchmarks[seed, method, task_type, value])
                    for seed in seeds
                    for method in methods
                )
            )
            if cohort is not None:
                common[value] &= cohort.get((task_type, value), set())
            if not common[value]:
                raise ValueError(
                    f"no shared {task_type} datasets at {value:g}; check methods/seeds/inputs"
                )
        if buffered:
            shared = set.intersection(*common.values())
            if not shared:
                raise ValueError("no shared buffered-exclusion cohort across the requested radii")
            common = dict.fromkeys(values, shared)
        for value in values:
            groups = ["all"]
            if categories:
                groups += sorted({families[name] for name in common[value]})
            for family in groups:
                selected = sorted(
                    name for name in common[value] if family == "all" or families[name] == family
                )
                for method in methods:
                    scores = [
                        statistics.fmean(
                            benchmarks[seed, method, task_type, value][name] for name in selected
                        )
                        for seed in seeds
                    ]
                    distance = {}
                    if buffered:
                        distance["median_nn_km"] = statistics.fmean(
                            statistics.median(distances[seed, value, name] for name in selected)
                            for seed in seeds
                        )
                    output.append(
                        dict(
                            method=method,
                            task_type=task_type,
                            family=family,
                            **{"radius_km" if buffered else "cell_deg": value},
                            mean=statistics.fmean(scores),
                            std=statistics.stdev(scores) if len(scores) > 1 else "",
                            n_benchmarks=len(selected),
                            **distance,
                            **{
                                f"seed_{seed}": score
                                for seed, score in zip(seeds, scores, strict=True)
                            },
                            benchmarks=";".join(selected),
                        )
                    )
    if not output:
        raise ValueError("no results match the requested settings")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write CSV here instead of standard output")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--cells", nargs="+", type=float, default=list(CELLS))
    parser.add_argument("--methods", nargs="+", help="exact method names to compare")
    parser.add_argument("--categories", action="store_true")
    parser.add_argument("--task-type", choices=["regression", "classification"])
    parser.add_argument("--buffered", action="store_true")
    parser.add_argument("--radii", nargs="+", type=float, default=list(RADII))
    parser.add_argument(
        "--cohort", type=Path, help="restrict to datasets from a previous summary CSV"
    )
    args = parser.parse_args()
    cohort = None
    if args.cohort:
        with args.cohort.open() as file:
            cohort = {
                (row["task_type"], float(row["cell_deg"])): set(row["benchmarks"].split(";"))
                for row in csv.DictReader(file)
                if row["family"] == "all"
            }
    inputs = load_inputs(args.manifest, tuple(args.seeds), args.buffered)
    if args.task_type:
        inputs = [row for row in inputs if row["task_type"] == args.task_type]
    rows = summarize(
        inputs,
        seeds=tuple(args.seeds),
        values=tuple(args.radii if args.buffered else args.cells),
        methods=tuple(args.methods) if args.methods else None,
        categories=args.categories,
        buffered=args.buffered,
        cohort=cohort,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    file = args.out.open("w", newline="") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            file.close()


if __name__ == "__main__":
    main()
