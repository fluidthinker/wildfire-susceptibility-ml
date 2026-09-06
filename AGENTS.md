# AGENTS.md

## Project purpose

This repository contains a reusable geospatial machine-learning workflow for modeling wildfire susceptibility from environmental characteristics.

New Mexico is the initial case study, but reusable code should be designed so that other U.S. states can be analyzed without rewriting core functionality.

The project emphasizes:

- clear geospatial reasoning
- reproducible data processing
- traditional supervised machine learning
- spatial validation
- readable, maintainable Python
- learning and understanding over unnecessary abstraction

Do not overengineer the project.

---

## Repository architecture

### `analysis/`

Use `analysis/` for case-study workflows, prototypes, QA/QC, and interactive investigation.

Analysis scripts use normal `.py` files with VS Code `# %%` cell markers.

More specific instructions for analysis scripts are defined in:

`analysis/AGENTS.md`

### `src/wildfire_susceptibility/`

Use `src/wildfire_susceptibility/` for reusable project logic.

More specific instructions for reusable library code are defined in:

`src/wildfire_susceptibility/AGENTS.md`

---

## General Python style

Write professional Python that is understandable by another analyst or developer.

Prefer:

- descriptive variable and function names
- type hints for nontrivial functions
- `pathlib.Path` for filesystem paths
- small, focused functions
- clear separation of responsibilities
- vectorized NumPy, Pandas, GeoPandas, Xarray, or Shapely operations when appropriate
- explicit validation of important assumptions
- readable intermediate variables when they improve understanding
- readable code over clever or overly compact code

Avoid:

- hardcoded user-specific filesystem paths
- unexplained magic numbers
- unnecessary classes
- deeply nested logic
- premature frameworks or abstractions
- `sys.path` hacks
- duplicated implementations
- long procedural blocks when meaningful responsibilities can be expressed as functions

Use Google-style docstrings for nontrivial functions.

---

## Geospatial conventions

Treat spatial decisions as part of the analysis methodology.

Current project conventions include:

- analysis CRS: `EPSG:5070`
- projected distances and grid dimensions are expressed in meters
- standard analysis cell: `1000 m × 1000 m`
- preserve complete grid cells
- include a cell when its centroid falls within the study-area boundary
- do not clip retained cells into partial boundary polygons
- use Census state FIPS codes for state selection
- prefer authoritative, reproducible national data sources

Do not silently:

- reproject data
- repair geometries
- resample rasters
- replace NoData
- change spatial predicates
- change temporal definitions
- change aggregation rules

These operations must be explicit and documented.

---

## Raster-to-grid aggregation

Different source datasets are summarized to the common analysis grid.

Treat aggregation and resampling rules as scientific/modeling decisions.

Examples:

- continuous variables: use an explicitly documented statistic such as mean or median
- categorical vegetation: use a clearly defined plurality/dominant-class rule
- aspect: do not treat raw degrees as ordinary linear values
- continuous raster reprojection: choose and document an appropriate resampling method
- document NoData handling and coverage requirements

Do not choose aggregation or resampling methods silently.

---

## Machine-learning conventions

The project models associations between environmental characteristics and historical wildfire occurrence.

It should not be presented as:

- real-time fire prediction
- ignition forecasting
- full wildfire hazard
- insurance-style wildfire risk

The initial model emphasizes biophysical susceptibility and intentionally excludes human ignition/exposure variables unless the project scope is explicitly changed.

Use preprocessing pipelines where appropriate so transformations are learned from training data and applied consistently to validation and test data.

Spatial validation is a first-class part of the project.

Avoid causal claims from model associations.

---

## Data and outputs

Use:

- `data/raw/` for downloaded source data
- `data/interim/` for intermediate products and checkpoints
- `data/processed/` for prepared analytical datasets
- `outputs/` for generated figures, maps, and model results

Do not commit large source datasets or generated GIS products unless explicitly requested.

Important derived outputs should be reproducible from documented source data and code.

---

## Dependency management

The Conda environment is defined in:

`environment.yml`

Python package configuration is defined in:

`pyproject.toml`

Do not:

- install packages automatically
- modify the environment unless required by the task
- introduce a dependency when the existing project stack can reasonably solve the problem

If a required dependency is missing, report it and stop rather than installing it without approval.

---

## Engineering principles

Prefer designs that improve reliability, reproducibility, and clarity.

Useful patterns include:

- prototype small → validate → scale
- batch large workloads when needed
- checkpoint expensive work
- make pipelines restartable
- validate intermediate outputs before trusting them
- keep memory use bounded
- reuse completed work
- stream large data when practical
- only mark work as finished after it succeeds
- clean up partial work after failures
- prefer deterministic processing so reruns are reproducible
- materialize data only where persistence provides value
- keep processing representation separate from final analysis representation
- build independent feature pipelines that join through authoritative keys such as `cell_id`
- use the simplest architecture that safely fits the workload

When one of these patterns is relevant, make it visible in comments or the implementation summary.

---

## Codex task behavior

Before making substantive changes:

1. inspect relevant existing files
2. inspect repository status
3. read the applicable `AGENTS.md` files
4. preserve validated behavior unless the task explicitly changes it

For each task:

- modify only files relevant to the requested work
- do not expand scope without permission
- do not commit
- do not push
- do not create or switch branches
- do not alter Git remotes
- do not install packages
- do not modify unrelated files

After implementation:

1. summarize files created or modified
2. provide a major-step walkthrough of the implemented workflow
3. explain important scientific, geospatial, and engineering decisions
4. identify good engineering patterns used
5. provide a short 80/20 takeaway
6. report assumptions, uncertainties, or issues
7. show `git status --short`
8. stop and wait for human review

Do not provide a line-by-line explanation unless explicitly requested.

---

## Guiding principle

Prefer code that makes the scientific, geospatial, and engineering reasoning easy to understand.

The goal is not merely to make the code run.

The goal is to produce a workflow whose assumptions, decisions, and outputs can be understood, validated, reproduced, and defended.