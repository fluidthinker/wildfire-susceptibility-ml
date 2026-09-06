# AGENTS.md — Reusable library code

These instructions apply to files under `src/wildfire_susceptibility/`.

This directory contains reusable project logic.

Code here should be clean, focused, testable, and state-agnostic where practical.

---

## Reusable design

Reusable functions should:

- avoid New Mexico-specific assumptions unless explicitly parameterized
- avoid user-specific paths
- avoid hidden global state
- expose clear inputs and outputs
- separate meaningful responsibilities
- avoid duplicated logic
- be deterministic where practical
- validate important assumptions
- favor functions over classes unless a class provides a clear design benefit

Do not create placeholder abstractions before actual reuse exists.

Move logic from `analysis/` into `src/` when:

- the behavior has been validated
- the responsibility is clearly defined
- reuse is demonstrated or strongly justified
- moving it improves maintainability rather than merely adding abstraction

---

## Function design

Prefer small, focused functions with descriptive names.

A function should ideally have one clear responsibility.

Prefer:

    def aggregate_continuous_raster_to_grid(...):
        ...

over functions that:

- download data
- transform it
- aggregate it
- validate it
- write it

all in one operation.

Separate I/O, transformation, aggregation, and validation when that makes responsibilities clearer.

Do not fragment simple logic into unnecessary one-line functions.

---

## Docstrings

Use Google-style docstrings for all public functions and for nontrivial private helpers.

Document:

- purpose
- important spatial/scientific assumptions
- `Args`
- `Returns`
- `Raises`

Example:

    def create_analysis_grid(
        boundary: gpd.GeoDataFrame,
        cell_size: int = 1000,
    ) -> gpd.GeoDataFrame:
        """Create a regular square analysis grid within a study-area boundary.

        Candidate cells are created in the projected coordinate system and
        retained when their centroids fall within the study-area polygon.
        Cells are not clipped at the boundary so every retained observation
        preserves the same nominal area.

        Args:
            boundary: GeoDataFrame containing one valid study-area geometry
                in a projected CRS with meter units.
            cell_size: Width and height of each square cell in meters.

        Returns:
            GeoDataFrame containing retained cells with unique IDs.

        Raises:
            ValueError: If the boundary, CRS, geometry, or cell size is invalid.
        """

---

## Comments

Comments in `src/` should be concise and professional.

Explain:

- why a non-obvious decision is necessary
- spatial/scientific assumptions
- unusual algorithms
- important validation
- numerical or memory considerations

Do not use the more tutorial-like comment density expected in `analysis/`.

Avoid comments that merely repeat the code.

Good:

    # Preserve complete cells so every ML observation represents the same
    # nominal analysis area.

Avoid:

    # Loop over cells.

---

## Readability

Prefer:

- descriptive names
- explicit intermediate variables when they clarify intent
- straightforward control flow
- early validation
- early returns when they simplify logic
- standard library and existing project dependencies where sufficient

Avoid:

- clever one-liners that obscure spatial logic
- deeply nested conditions
- broad functions with many unrelated responsibilities
- unnecessary mutable global state
- premature optimization
- unnecessary object-oriented design

Pythonic code means clear, idiomatic, and maintainable, not merely compact.

---

## Geospatial behavior

Reusable geospatial functions must make important behavior explicit.

Where relevant, expose or document:

- CRS requirements
- expected units
- resolution
- resampling method
- spatial predicate
- NoData behavior
- grid alignment
- aggregation statistic
- temporal assumptions

Do not silently reproject, resample, repair geometries, or replace missing data.

---

## Validation

Validate assumptions at appropriate boundaries.

Examples:

- projected CRS required for meter-based operations
- positive cell size
- unique IDs
- aligned raster dimensions
- expected feature domains
- valid geometries
- expected schema

Raise clear exceptions with actionable messages.

Avoid silently correcting scientifically meaningful problems unless explicitly requested.

---

## Performance and memory

Prefer algorithms whose resource behavior is understandable.

When working with large raster or tabular datasets:

- avoid loading unnecessary data
- use bounded reads where practical
- preserve lazy evaluation when beneficial
- distinguish lazy evaluation from bounded-memory execution
- use batching or checkpointing only when workload size justifies the complexity
- release large intermediate objects when necessary

Do not sacrifice clarity for premature micro-optimization.

---

## Reproducibility

Prefer deterministic behavior.

Where ordering affects results:

- define ordering explicitly
- use stable keys
- make precedence rules visible

Do not rely on incidental filesystem, API, or collection ordering when it could affect scientific output.

---

## Scope

Reusable modules should implement stable project capabilities.

Do not include:

- exploratory plots
- one-off prototype output
- New Mexico-only QA narration
- interactive investigation
- temporary debugging code

Those belong in `analysis/`.