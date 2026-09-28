"""Compare completed Random Forest Random and Spatial CV results only."""

# %% Imports
from pathlib import Path

from wildfire_susceptibility.validation_comparison import run_validation_comparison


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_POPULATION = {"rows": 252_569, "positives": 6_585, "negatives": 245_984, "fold_count": 5}


# %% Main workflow
def main() -> None:
    """Publish the saved Random Forest validation comparison."""
    # The shared workflow loads summaries, validates comparability and folds,
    # builds the table and figure, then validates and publishes all artifacts.
    run_validation_comparison(
        ROOT, model_key="random_forest", model_label="Random Forest",
        expected_model="RandomForestClassifier", expected_population=EXPECTED_POPULATION,
        require_input_hash=True,
    )


# %% Run script
if __name__ == "__main__":
    main()
