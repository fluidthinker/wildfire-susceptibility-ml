# AGENTS.md

## Project purpose

This repository contains a reusable geospatial machine-learning workflow for modeling wildfire susceptibility from environmental characteristics.

New Mexico is the initial case study, but reusable code should be designed so that other U.S. states can be analyzed without rewriting core functionality.

The project emphasizes:

* clear geospatial reasoning
* reproducible data processing
* traditional supervised machine learning
* spatial validation
* readable, maintainable Python
* learning and understanding over unnecessary abstraction

Do not overengineer the project.

---

## Repository architecture

### `analysis/`

Use `analysis/` for interactive investigation and project workflows.

Analysis files:

* are normal `.py` files
* use VS Code `# %%` cell markers
* should be easy to run cell-by-cell
* contain visible case-study parameters
* may contain exploratory plots and inspection code
* should call reusable functions from `src/` rather than duplicate implementation logic

Example:

```python
# %% Case-study parameters
STATE_FIPS = "35"
TARGET_CRS = "EPSG:5070"
CELL_SIZE_M = 1000
```

New Mexico-specific parameters belong here rather than inside reusable modules.

### `src/wildfire_susceptibility/`

Use this directory for reusable project logic.

Reusable modules should:

* be state-agnostic where practical
* expose small, clearly named functions
* avoid hidden global state
* avoid user-specific paths
* avoid duplicated logic
* favor simple functions over classes unless a class provides a clear benefit

Do not create placeholder modules before actual reusable functionality exists.

---

## Python style

Write clear, professional Python intended to be understandable by another analyst or developer.

Prefer:

* descriptive variable and function names
* type hints for public functions
* `pathlib.Path` for filesystem paths
* small focused functions
* vectorized NumPy, GeoPandas, or Shapely operations when appropriate
* explicit validation of important assumptions
* readable code over clever or overly compact code

Avoid:

* hardcoded user-specific filesystem paths
* unexplained magic numbers
* unnecessary classes
* deeply nested logic
* premature frameworks or abstractions
* `sys.path` hacks
* duplicated implementations

---

## Docstrings

Use **Google-style docstrings** for public reusable functions.

Public functions should normally document:

* what the function does
* important behavioral or spatial assumptions
* `Args`
* `Returns`
* `Raises`

Example:

```python
def create_analysis_grid(
    boundary: gpd.GeoDataFrame,
    cell_size: int = 1000,
) -> gpd.GeoDataFrame:
    """Create a regular square grid within a study-area boundary.

    Candidate cells are created in the projected coordinate system and
    retained when their centroids fall within the study-area polygon.
    Cells are not clipped at the boundary, preserving equal cell areas.

    Args:
        boundary: GeoDataFrame containing one valid study-area geometry
            in a projected CRS with meter units.
        cell_size: Width and height of each square cell in meters.

    Returns:
        GeoDataFrame containing retained grid cells with unique cell IDs.

    Raises:
        ValueError: If the boundary, CRS, geometry, or cell size is invalid.
    """
```

Use concise module-level docstrings that explain the module's responsibility.

---

## Comments

## Code comments

Comments should explain **why a decision is being made**, not merely repeat what the code already says.

Good:

```python
# Snap the grid extent to exact multiples of the cell size so grid
# alignment remains deterministic across study areas.
```

Good:

```python
# Keep full cells whose centroids fall inside the study area.
# Do not clip boundary cells because every ML observation should
# represent the same 1-km² analysis unit.
```

Avoid:

```python
# Get bounds
```

immediately above:

```python
bounds = boundary.total_bounds
```

Add explanatory comments particularly when code involves:

- CRS choices and reprojection
- units and resolution
- spatial predicates and geometry assumptions
- raster resampling
- zonal aggregation
- NoData handling
- categorical encoding
- temporal assumptions
- filtering criteria
- QA/QC checks and why they matter
- data leakage prevention
- spatial validation
- modeling decisions that may not be obvious from syntax alone
- lazy loading, chunking, or limiting data access

For exploratory scripts in `analysis/`:

- Use `# %%` cells.
- Add short comments before important code blocks so the workflow can be read like a narrated analysis.
- Assume the reader is learning the workflow and should be able to understand the purpose of each major step without reverse-engineering the code.
- Prefer comments that explain the geospatial, scientific, or data-engineering reasoning behind a step.
- Make important assumptions visible in comments, especially when a change could affect scientific validity or reproducibility.

Avoid:

- comments that merely restate obvious Python syntax
- excessive line-by-line commenting
- long essay-like comments when a concise explanation is sufficient
- comments that describe what the code used to do rather than what the current code does
---

## Geospatial conventions

Treat spatial decisions as part of the analysis methodology.

Current project conventions include:

* analysis CRS: `EPSG:5070`
* projected distances and grid dimensions are expressed in meters
* standard analysis cell: `1000 m × 1000 m`
* preserve complete grid cells
* include a cell when its centroid falls within the study-area boundary
* do not clip cells into partial boundary polygons
* use Census state FIPS codes for state selection
* prefer authoritative, reproducible national data sources

Do not silently:

* reproject data
* repair geometries
* resample rasters
* replace NoData
* change spatial predicates

These operations should be explicit and documented.

---

## Raster-to-grid aggregation

Different source resolutions will eventually be summarized to the common analysis grid.

Treat aggregation rules as modeling decisions.

Examples:

* continuous raster variables: use an explicitly documented statistic such as mean or median
* categorical vegetation: use a clearly defined plurality/dominant-class rule
* aspect: do not treat raw degrees as ordinary linear values; use an appropriate circular representation
* document NoData handling and coverage requirements

Do not choose an aggregation method silently.

---

## Machine-learning conventions

The project models associations between environmental characteristics and historical wildfire occurrence.

It should not be presented as:

* real-time fire prediction
* ignition forecasting
* full wildfire hazard
* insurance-style wildfire risk

The initial model intentionally emphasizes biophysical susceptibility and excludes human ignition/exposure variables unless the project scope is explicitly changed.

Use preprocessing pipelines where appropriate so transformations are learned from training data and applied consistently to validation/test data.

Spatial validation is a first-class part of the project, not an afterthought.

Avoid causal claims from model associations.

---

## Data and outputs

Do not commit large source datasets or generated GIS products unless explicitly requested.

Use:

* `data/raw/` for downloaded source data
* `data/interim/` for intermediate products
* `data/processed/` for prepared analytical datasets
* `outputs/` for generated figures, maps, and results

Code should be able to recreate important derived outputs from documented source data.

---

## Dependency management

The Conda environment is defined in:

`environment.yml`

Python package configuration is defined in:

`pyproject.toml`

Do not:

* install packages automatically
* modify the environment unless required by the task
* introduce a dependency when the existing standard library or project stack can reasonably solve the problem

If a required dependency is missing, report it and stop rather than installing it without approval.

---

## Codex task behavior

Before making substantive changes:

1. inspect relevant existing files
2. inspect repository status
3. preserve working behavior unless the task explicitly changes it

For each task:

* modify only files relevant to the requested work
* do not expand scope without permission
* do not commit
* do not push
* do not create or switch branches
* do not alter Git remotes
* do not install packages
* do not modify unrelated files

After implementation:

1. summarize files created or modified
2. provide a major-step walkthrough of the implemented workflow
3. explain important implementation decisions
4. identify any good engineering patterns used, such as batching, checkpointing,
   validation, deterministic processing, bounded-memory design, modular pipelines,
   or restartability
5. provide a short 80/20 takeaway
6. report assumptions or issues
7. show `git status`
8. stop and wait for review


---
## Learning-oriented implementation summaries

When completing or substantially refactoring an analysis script, include a concise walkthrough of the script organized by its major processing steps.

The walkthrough should:

- explain the workflow in plain English before discussing implementation details
- group the code into approximately 5–10 major steps
- name the functions or code sections responsible for each step
- explain what each major step accomplishes and why it exists
- call out important geospatial, scientific, or data-engineering decisions
- distinguish core logic from lower-priority plumbing or compatibility code
- end with a short 80/20 takeaway describing the few ideas most important to understand

Do not provide a line-by-line code explanation unless explicitly requested.

Prefer a structure like:

1. Validate inputs
2. Find/select source data
3. Define processing extent
4. Process or transform data
5. Aggregate to analysis units
6. QA/QC
7. Write final output

The exact steps should reflect the actual script rather than forcing this template.



---

## Guiding principle

Prefer code that makes the scientific and geospatial reasoning easy to understand.

The goal is not merely to make the code run.

The goal is to produce a reusable workflow whose assumptions, spatial decisions, and modeling choices can be understood and defended.


