# %% Imports
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# %% Setup
analysis_size = 1000
source_size = 450
destination_size = 250

# Deliberately shift the source raster so it does NOT align
# with our 1-km analysis grid.
source_x_offset = -175
source_y_offset = -125


# %% Plot the three grids
fig, ax = plt.subplots(figsize=(9, 9))


# --------------------------------------------------
# 1. Draw the four 1-km analysis cells
# --------------------------------------------------
for row in range(2):
    for col in range(2):
        x = col * analysis_size
        y = row * analysis_size

        rectangle = Rectangle(
            (x, y),
            analysis_size,
            analysis_size,
            fill=False,
            linewidth=3,
        )

        ax.add_patch(rectangle)

        cell_name = chr(
            ord("A") + row * 2 + col
        )

        ax.text(
            x + analysis_size / 2,
            y + analysis_size / 2,
            f"Cell {cell_name}",
            ha="center",
            va="center",
            fontsize=13,
        )


# --------------------------------------------------
# 2. Draw misaligned 450-m source pixels
# --------------------------------------------------
x = source_x_offset

while x < 2000:
    y = source_y_offset

    while y < 2000:
        rectangle = Rectangle(
            (x, y),
            source_size,
            source_size,
            fill=False,
            linestyle="--",
            linewidth=1.5,
        )

        ax.add_patch(rectangle)

        # Show the source-pixel centroid.
        ax.plot(
            x + source_size / 2,
            y + source_size / 2,
            marker="o",
            markersize=3,
        )

        y += source_size

    x += source_size


# --------------------------------------------------
# 3. Draw aligned 250-m destination grid
# --------------------------------------------------
for x in range(0, 2000, destination_size):
    ax.axvline(
        x,
        linewidth=0.4,
        alpha=0.4,
    )

for y in range(0, 2000, destination_size):
    ax.axhline(
        y,
        linewidth=0.4,
        alpha=0.4,
    )


# --------------------------------------------------
# Plot formatting
# --------------------------------------------------
ax.set_xlim(-250, 2250)
ax.set_ylim(-250, 2250)

ax.set_aspect("equal")

ax.set_title(
    "Misaligned 450-m Source Raster vs "
    "1-km Analysis Grid\n"
    "with Aligned 250-m Destination Grid"
)

ax.set_xlabel("meters")
ax.set_ylabel("meters")

plt.show()
# %%
