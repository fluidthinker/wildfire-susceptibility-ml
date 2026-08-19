# Wildfire Susceptibility ML

Reusable geospatial machine learning workflow for modeling wildfire susceptibility from environmental characteristics, with New Mexico as the initial case study.

## Status
Early development.

## Initial case study
New Mexico.

## Development approach
- `analysis/` contains interactive `.py` files using VS Code `# %%` cells.
- These files are used for exploration and learning.
- Reusable logic belongs in `src/wildfire_susceptibility/`.
- Reusable functionality will be extracted into Python modules as the project develops.
- Raw and large geospatial datasets are intentionally excluded from Git.

## Project directory overview
- `analysis/`: exploratory, script-based analysis workflows
- `data/`: raw, interim, and processed data directories
- `outputs/`: generated figures and maps
- `src/wildfire_susceptibility/`: reusable project logic

The project is intentionally lightweight and designed to support early-stage geospatial modeling work before broader reusable abstractions are introduced.
