"""Task categories and the regression-score floor used by CoordBench summaries."""

SOCIO = (
    {
        "pdfm-conus27",
        "pdfm-conus27-extrapolation",
        "california-housing",
        "mosaiks-population",
        "mosaiks-nightlights",
        "mosaiks-income",
        "mosaiks-housing",
        "mosaiks-roads",
        "satclip-population",
    }
    | {
        f"sustainbench-{k}"
        for k in ("asset", "water", "sanitation", "child_mortality", "women_edu", "women_bmi")
    }
    | {
        f"places-{k}"
        for k in (
            "asthma",
            "cancer",
            "chd",
            "checkup",
            "copd",
            "diabetes",
            "high_chol",
            "mental_health",
            "obesity",
            "phys_health",
            "sleep_lt7",
            "smoking",
        )
    }
)
ENV = {
    "worldclim-bio1",
    "worldclim-bio12",
    "soilgrids-soc",
    "soilgrids-phh2o",
    "dm-aster_ged",
    "dm-openet_ensemble",
    "mosaiks-elevation",
    "mosaiks-treecover",
    "satclip-elevation",
    "satclip-air-temp",
}
LAND = {
    f"dm-{k}"
    for k in (
        "africa_crop_mask",
        "canada_crops_coarse",
        "canada_crops_fine",
        "descals",
        "ethiopia_crops",
        "glance",
        "lcmap_lc",
        "lcmap_lcc",
        "lcmap_lu",
        "lcmap_luc",
        "lucas_lc",
        "lucas_lu",
        "us_trees",
    )
} | {"satclip-biome", "satclip-country", "satclip-ecoregion"}

# Apply the task-level regression floor before averaging tasks within datasets.
R2_FLOOR = -1.0


def floor_r2(v: float) -> float:
    return max(v, R2_FLOOR)


def in_fam(name: str, fam: set) -> bool:
    return name in fam


def family_of(name: str) -> str:
    if in_fam(name, SOCIO):
        return "socio"
    if in_fam(name, ENV):
        return "env"
    if in_fam(name, LAND):
        return "land"
    return "other"
