"""Dataset-family selection shared by the CoordBench evaluation scripts."""

from mind.eval import benchmarks as B


def gather_benchmarks(
    only: set[str] | None = None, pdfm_split: str | None = None, dm_only: set[str] | None = None
) -> list[B.Benchmark]:
    """Load the final CoordBench suite, or the requested dataset families."""
    builders = {
        "pdfm": lambda: [B.load_pdfm(split=pdfm_split)],
        "air_temp": lambda: [B.load_air_temp()],
        "calhousing": lambda: [B.load_california_housing()],
        "sustainbench": B.load_sustainbench,
        "mosaiks": B.load_usavars,
        "satclip_official": B.load_satclip_official,
        "worldclim": B.load_worldclim,
        "soilgrids": B.load_soilgrids,
        "cdc_places": B.load_cdc_places,
        "deepmind": lambda: B.load_deepmind_eval(names=dm_only),
    }
    if only is not None:
        unknown = only - builders.keys()
        if unknown:
            raise ValueError(
                f"--only names not recognized: {sorted(unknown)}. Known benchmarks: "
                f"{sorted(builders)}. A typo here would otherwise silently select zero "
                "benchmarks and produce an empty results table while exiting 0."
            )
    benches = []
    for key, build in builders.items():
        if only is not None and key not in only:
            continue
        try:
            benches.extend(build())
        except Exception as exc:
            raise RuntimeError(
                f"benchmark loader '{key}' failed: {type(exc).__name__}: {exc}"
            ) from exc
    return benches
