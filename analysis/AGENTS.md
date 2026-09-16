# AGENTS.md — Analysis scripts

These instructions apply to files under `analysis/`.

Analysis scripts are both working project code and learning artifacts.

The reader should be able to understand the major scientific and data-processing workflow without reverse-engineering dense Python.

---

## Analysis-script structure

Use normal `.py` files with VS Code `# %%` cell markers.

Organize scripts so the file has a clear separation between:

1. imports
2. constants and paths
3. helper function definitions
4. the main workflow
5. final QA/QC and visualization

Prefer this general organization:

    # %% Imports

    # %% Parameters and paths

    # %% Helper functions

    # %% Main workflow

    # %% Run script

The exact sections should reflect the actual workflow rather than forcing this template.

---

## Function placement

Unless there is a strong reason otherwise, define all helper functions before the executable workflow begins.

Do not interleave function definitions with calls to those functions.

Prefer:

    imports
    ↓
    parameters
    ↓
    all helper functions
    ↓
    main()
    ↓
    if __name__ == "__main__":
        main()

This keeps the file easy to scan.

The helper functions explain HOW individual steps work.

The `main()` function explains WHAT happens and WHEN.

---

## Main workflow function

Prefer a `main()` function for nontrivial analysis scripts.

Define helper functions first, then place the high-level execution flow inside `main()`.

The `main()` function should read like a concise workflow summary.

It should make the major processing steps explicit with numbered step comments that explain both:

- what happens
- why the step exists

Prefer:

    def main():
        """Run the PRISM precipitation prototype workflow."""

        # STEP 1 — Acquire the authoritative source data.
        # This gives the workflow a reproducible source rather than relying on
        # an undocumented local file.
        source_path = acquire_source(...)

        # STEP 2 — Validate the source product and metadata.
        # Confirm that production is using the same source assumptions that
        # were established during the prototype.
        source = validate_source(...)

        # STEP 3 — Select the prototype analysis cells.
        # Keep the test small enough to inspect before statewide scaling.
        test_grid = select_test_grid(...)

        # STEP 4 — Reproject and create the temporary integration surface.
        # Align the source to the analysis grid using the resampling rule
        # appropriate for this data type.
        processed = transform_source(...)

        # STEP 5 — Aggregate precipitation to 1-km analysis cells.
        # Convert the source-specific raster representation into one feature
        # value per authoritative analysis cell.
        features = aggregate_to_grid(...)

        # STEP 6 — Validate the resulting feature table.
        validate_features(features)

        # STEP 7 — Report QA/QC results.
        report_qaqc(...)

        # STEP 8 — Create the spatial QA plot.
        plot_spatial_qa(...)


    if __name__ == "__main__":
        main()

Keep `main()` focused on orchestration.

Do not place most implementation detail directly inside `main()`.

The reader should be able to understand the entire workflow by reading `main()` before reading any helper functions.

Prefer:

    # STEP 2 — Group the existing 1-km cells into bounded processing batches.
    # Each batch later requests only the source raster needed for that region.

over:

    # STEP 2 — Prepare batches.

---

## Top-level workflow should read like a table of contents

The executable workflow should call clearly named functions in a logical order.

Prefer:

    source_metadata = inspect_source(...)
    test_grid = select_test_grid(...)
    processed_surface = reproject_source(...)
    feature_table = aggregate_to_grid(...)
    validate_feature_table(feature_table)
    report_qaqc(...)
    plot_spatial_qa(...)

over one long procedural block containing all implementation details.

The workflow should be understandable at a glance.

---

## Functions

Use clearly named functions for meaningful units of work.

Examples:

    download_source_if_missing(...)
    validate_source_metadata(...)
    select_prototype_grid(...)
    read_source_window(...)
    reproject_precipitation_surface(...)
    aggregate_to_analysis_grid(...)
    validate_feature_table(...)
    report_qaqc(...)
    plot_spatial_qa(...)

Extract a function when doing so:

- clarifies a meaningful responsibility
- makes a major workflow step easier to understand
- isolates nontrivial logic
- makes validation easier
- reduces duplication
- provides likely reusable behavior

Do not create tiny one-line functions merely to increase the number of functions.

Prefer functions with one clear responsibility.

### Prefer one clear responsibility per function

A function should perform one conceptually coherent job.

When a function mixes several responsibilities such as loading data,
validating contracts, transforming data, writing artifacts, and recording
provenance, consider separating those responsibilities into smaller,
clearly named helpers.

Prefer a workflow such as:

    grid = load_analysis_grid()
    validate_analysis_grid_contract(grid)

    metadata = load_source_metadata()
    validate_source_contract(metadata)

    legend = load_evt_legend()
    validate_evt_legend_contract(legend)

    fingerprint = build_run_fingerprint(...)

over a broad helper such as:

    validate_inputs(...)

when that broad helper actually performs many unrelated tasks.

Good function boundaries should make the workflow visible from the function
names alone.

A useful test is:

**Can the function's purpose be explained clearly in one short sentence?**

Do not split functions merely to reduce line count. Split them when the code
contains distinct conceptual responsibilities.

Prefer composition of small, clearly named functions over large functions that
mix loading, validation, transformation, provenance, and side effects.

Avoid vague helper names such as:

- `prepare_data()`
- `handle_raster()`
- `process_everything()`
- `validate_inputs()`

when a more specific name can describe the real responsibility.

---

## Google-style docstrings

Use Google-style docstrings for all nontrivial functions, including private helper functions.

A nontrivial function should normally document:

- what the function does
- important scientific, spatial, temporal, or engineering assumptions
- `Args`
- `Returns`, when a value is returned
- `Raises`, for important expected failure conditions

### Write the purpose in plain language first

The opening sentences of a docstring should explain the function's real job
before describing implementation details.

The reader should be able to answer:

- What does this function actually do?
- What data does it operate on?
- Why does this function exist in the workflow?

Prefer:

    """Group the existing 1-km analysis cells into manageable processing batches.

    The authoritative grid is not changed. Each cell is assigned a processing
    row, column, and batch so the pipeline can process one bounded part of the
    state at a time instead of creating one large statewide raster in memory.
    """

over:

    """Map existing squares to integer positions and bounded rectangular windows."""

Use domain language the reader already understands before implementation
jargon.

`Args`, `Returns`, and `Raises` describe the interface; they do not replace a
plain-language explanation of the function's purpose.

Example:

    def reproject_precipitation_surface(
        source: rasterio.io.DatasetReader,
        bounds: tuple[float, float, float, float],
        resolution_m: int,
    ) -> np.ndarray:
        """Reproject PRISM precipitation to an aligned integration grid.

        PRISM is treated as a continuous climate surface. Bilinear resampling
        estimates precipitation between neighboring source cells while the
        raster is transformed into EPSG:5070.

        The finer integration grid is used to approximate the spatial mean
        within each 1-km analysis cell. It does not create new climate
        information or imply that PRISM has observations at the finer
        resolution.

        Args:
            source: Open PRISM raster dataset.
            bounds: Target bounds in EPSG:5070.
            resolution_m: Pixel size of the temporary integration grid in meters.

        Returns:
            Two-dimensional precipitation array aligned to the target grid.

        Raises:
            ValueError: If the requested bounds or output grid are invalid.
        """

A one-line docstring is appropriate only for genuinely trivial helpers.

---

## Comments for learning

Analysis scripts should contain more explanatory comments than production library code.

Comments should remain professional and useful rather than narrating obvious Python syntax.

Use three levels of comments.

### 1. Major-step comments

Use these inside `main()` so the workflow can be understood quickly.

Example:

    # STEP 4 — Reproject PRISM and create the temporary integration surface

### 2. Decision comments

Explain why an important scientific, geospatial, or engineering choice was made.

Example:

    # Use bilinear resampling because precipitation is continuous.
    # Nearest-neighbor would preserve discrete source-cell values and create
    # artificial steps.

### 3. Mechanism comments

Explain non-obvious operations whose syntax hides an important concept.

Example:

    # `reproject()` performs several operations together:
    #   1. maps locations from PRISM's native CRS into EPSG:5070,
    #   2. estimates values at new pixel locations using bilinear interpolation,
    #   3. writes those estimates onto the aligned destination grid.
    #
    # The 100 m grid is an integration aid. It does not create new 100 m
    # climate information from the ~800 m PRISM source.

### Narrate conceptual blocks inside complex functions

For substantial helper functions, use short comments to expose the internal
workflow.

The reader should not have to reverse-engineer a dense function body to
discover that it contains several conceptual stages.

Prefer:

    # Load only the authoritative key and geometry needed by this pipeline.
    grid = ...

    # Check the basic grid contract: expected row count, CRS, and unique IDs.
    ...

    # Verify that every retained geometry is a complete 1-km square.
    ...

    # Compare the current source metadata with the contract established during
    # the prototype.
    ...

    # Build the valid categorical class set, excluding documented NoData.
    ...

Comments should describe what conceptual task is happening and why.

Do not comment every Python statement. Add narration where the code moves from
one meaningful responsibility to another.

Add mechanism comments especially around:

- CRS transformations
- Rasterio `reproject()`
- raster resampling
- interpolation
- raster windows
- transforms and grid alignment
- source support pixels / halos
- masking and NoData handling
- boolean masks and what `True` / `False` represent
- NumPy array shapes before and after transformations
- NumPy reshaping
- multidimensional aggregation
- parallel arrays returned by functions such as
  `np.unique(..., return_counts=True)`
- categorical class counting and plurality selection
- tie detection and deterministic fallback rules
- zonal-style summaries
- spatial joins
- index mapping
- Dask lazy evaluation
- chunking
- batching
- checkpoints
- restart logic
- file streaming
- temporary `.part` files
- compact comprehensions or expressions that hide important behavior

Do not comment obvious assignments such as:

    # Get bounds
    bounds = grid.total_bounds

---

## Prefer readable intermediate variables

Do not optimize for shortest code.

When a compact expression hides an important spatial or computational idea, introduce descriptive intermediate variables.

Avoid leaving important logic unexplained:

    cell_means = values.reshape(8, 10, 8, 10).mean(axis=(1, 3))

Prefer:

    # Reorganize the fine-resolution raster into:
    # analysis rows × fine rows × analysis columns × fine columns.
    grouped_values = values.reshape(
        analysis_rows,
        samples_per_cell,
        analysis_columns,
        samples_per_cell,
    )

    # Average only the two fine-resolution dimensions, leaving one value
    # for each 1-km analysis cell.
    cell_means = grouped_values.mean(axis=(1, 3))

For validation logic, prefer named boolean checks when a long compound
condition combines several different assumptions.

Prefer:

    wrong_row_count = len(grid) != EXPECTED_CELLS
    wrong_crs = grid.crs.to_epsg() != 5070
    missing_ids = grid.cell_id.isna().any()
    duplicate_ids = not grid.cell_id.is_unique

    if wrong_row_count or wrong_crs or missing_ids or duplicate_ids:
        raise ValueError(...)

over one dense conditional containing all four concepts.

Do not expand simple conditions unnecessarily. Use named checks when they make
the data contract easier to read.

Readable code is preferred over clever code.

Pythonic code means clear, idiomatic, maintainable, and easy to follow.

It does not mean shortest possible code.

---

## Scientific and geospatial reasoning

Make important assumptions visible near the code that implements them.

Explain choices involving:

- CRS
- units
- resolution
- temporal periods
- raster resampling
- interpolation
- zonal aggregation
- NoData
- source filtering
- spatial predicates
- categorical aggregation
- leakage prevention
- validation design
- model features
- support pixels or halos

Do not silently make these decisions.

---

## Exploratory QA and production data contracts

Separate exploratory data discovery from production contract validation.

### Exploratory QA

During prototype work, investigate the source deeply enough to understand what
it is and to make scientific and engineering decisions.

Typical exploratory questions include:

- What product and version is this?
- What is the CRS?
- What is the native resolution?
- What does NoData mean?
- What values or categories occur?
- What metadata or legend defines those values?
- How does the source align with the analysis grid?
- Are there unexpected values or spatial patterns?

The purpose of exploratory QA is to discover and understand the assumptions
that the production pipeline will later depend on.

### Define the data contract

Once the source and method are understood, make the important expectations
explicit.

A raster contract might include:

- product/version
- CRS
- resolution
- number of bands
- data type
- NoData value
- categorical or continuous interpretation
- legend/version when applicable

A tabular contract might include:

- required columns
- data types
- unique key
- expected row count
- allowed value ranges
- missing-value rules

### Production contract validation

Production code should validate that current inputs still satisfy the known
contract rather than repeatedly rediscovering the dataset from scratch.

Mental model:

    explore
    ↓
    understand
    ↓
    define contract
    ↓
    production validates contract
    ↓
    process

A useful distinction is:

**Exploration asks: "What is this data?"**

**Production validation asks: "Is this still the data I expect?"**

Validation should enforce known contracts, not repeat exploratory QA without a
specific reason.

---

## QA/QC

Prototype and analysis scripts should contain explicit QA/QC.

Where relevant, validate:

- CRS
- units
- resolution
- dimensions
- bounds
- expected row counts
- unique IDs
- duplicate keys
- missing values
- NoData behavior
- physical value domains
- grid alignment
- source metadata
- output schema

Prefer failing clearly when an important assumption is violated rather than allowing questionable data to continue downstream.

---

## Prototype-first development

For new datasets or algorithms:

1. understand the source data
2. build a small prototype
3. validate scientific behavior
4. inspect QA/QC
5. measure resource use when relevant
6. test important assumptions or sensitivity when warranted
7. scale only after the prototype is understood and validated

Do not copy a complex architecture from another dataset unless the new workload demonstrates that it is necessary.

Use the simplest architecture that safely fits the workload.

---

## Prototype-to-production refactoring

Before scaling a successful prototype statewide:

1. preserve validated scientific behavior
2. refactor substantial procedural blocks into meaningful functions
3. define helper functions before the execution workflow
4. put the high-level execution flow inside `main()`
5. make the major workflow steps visible inside `main()`
6. improve names and intermediate variables
7. split helpers that combine distinct conceptual responsibilities
8. add plain-language Google-style docstrings
9. add major-step, decision, and mechanism comments
10. narrate dense function bodies at conceptual transitions
11. preserve or strengthen QA/QC
12. move genuinely reusable logic into `src/` only when reuse is demonstrated

Do not carry messy exploratory structure directly into production simply because the prototype produced correct output.

---

## Engineering patterns

When relevant, make useful engineering patterns visible in comments or the implementation summary.

Examples:

- prototype → validate → scale
- explore → understand → define contract → validate contract → process
- bounded-memory processing
- streaming
- reuse completed work
- only mark work as finished after it succeeds
- clean up partial work after failure
- validation before reuse
- batching
- checkpointing
- restartability
- deterministic processing
- modular feature pipelines
- materialize only where persistence provides value
- keep processing representation separate from final analysis representation
- separate exploratory QA from production contract validation
- prefer composition of single-responsibility helpers

---

## Implementation summary

After completing or substantially refactoring an analysis script, provide a concise walkthrough organized into approximately 5–10 major steps.

For each major step:

- state what happens in plain English
- name the responsible function or code section
- explain what data or object is involved
- explain why the step exists
- give a concrete example when it would materially improve understanding
- call out important scientific or engineering decisions

Prefer a workflow-first explanation such as:

    CHECK INPUTS
    What data?
    - authoritative 1-km grid
    - source metadata
    - source legend

    What happens?
    - confirm that current inputs still match the approved contract

    Why?
    - prevent the pipeline from silently processing a changed or incompatible source

    Example:
    - expected 30 m source resolution but current metadata reports 90 m → stop

Also identify useful engineering patterns used.

End with an **80/20 takeaway** describing the few ideas most important to understand.

Do not provide a line-by-line explanation unless explicitly requested.
