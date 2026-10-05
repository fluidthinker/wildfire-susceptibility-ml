from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Create a clean static portfolio figure showing how model performance changed
# under increasingly realistic geographic validation.
#
# Output:
#   assets/images/wildfire-validation-performance.png
# -----------------------------------------------------------------------------

# Save location relative to the project root.
output_path = Path("assets/images/wildfire-validation-performance.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Data: Random Forest ROC-AUC results from the project.
labels = [
    "Random CV",
    "Spatial CV",
    "Eastern Test Area",
]
values = [0.956, 0.901, 0.570]

# Optional short explanations shown below the x-axis labels.
descriptions = [
    "cells mixed across state",
    "held-out spatial blocks",
    "geographically separate test area",
]

# Create figure.
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(labels))
bars = ax.bar(x, values)

# Add value labels above each bar.
for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# Main title and subtitle.
fig.suptitle(
    "Validation Results",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

ax.set_title(
    "Random Forest ROC-AUC\nPerformance became weaker as geographic testing became more realistic",
    fontsize=12,
    pad=16,
)

# Axes formatting.
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.linspace(0, 1.0, 6))
ax.grid(axis="y", linewidth=0.8, alpha=0.35)
ax.set_axisbelow(True)

# Clean up spines.
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add small descriptive text under each bar label.
for xi, desc in zip(x, descriptions):
    ax.text(
        xi,
        -0.10,
        desc,
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.get_xaxis_transform(),
    )

# Footer note.
fig.text(
    0.5,
    0.02,
    "Higher is better. The eastern test area was never used during model development.",
    ha="center",
    fontsize=10,
)

plt.tight_layout(rect=[0.04, 0.08, 0.98, 0.90])

# Save figure.
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path.resolve()}")