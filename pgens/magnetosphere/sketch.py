import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import myplotlib
import numpy as np

plt.style.use("latex")

fig = plt.figure(dpi=600)
ax = fig.add_subplot(111)

r0 = 0.15

ax.plot([0, 0], [r0, 1], c="k", lw=0.5)
ax.plot([0, 0], [-r0, -1], c="k", lw=0.5)
ax.add_patch(
    mpatches.Arc((0, 0), 2, 2, angle=0, theta1=-90, theta2=90, color="k", lw=0.5)
)
ax.add_patch(
    mpatches.Arc(
        (0, 0), 2 * r0, 2 * r0, angle=0, theta1=-90, theta2=90, color="k", lw=0.5
    )
)
ax.add_patch(
    mpatches.Arc(
        (0, 0),
        2 - 0.25,
        2 - 0.25,
        angle=0,
        theta1=-90,
        theta2=90,
        color="b",
        lw=0.5,
        ls="--",
    )
)
ax.add_patch(
    mpatches.Arc(
        (0, 0),
        2 - 0.1,
        2 - 0.1,
        angle=0,
        theta1=-90,
        theta2=90,
        color="r",
        lw=0.5,
        ls="--",
    )
)
ax.text(
    1.05 / np.sqrt(2), 1.05 / np.sqrt(2), "absorbing particle boundaries", c="r", size=5
)
ax.text(
    1.05 / np.sqrt(2),
    1.05 / np.sqrt(2) - 0.04,
    r"$\mathtt{grid.boundaries.absorb.ds}$",
    c="r",
    size=5,
)
ax.annotate(
    "",
    xy=(1.02 / np.sqrt(2), 1.02 / np.sqrt(2)),
    xytext=(0.93 / np.sqrt(2), 0.93 / np.sqrt(2)),
    arrowprops=dict(arrowstyle="<->", lw=0.5, color="r"),
    size=5,
)
ax.annotate(
    "",
    xy=(1.01 * np.cos(np.pi / 6), 1.01 * np.sin(np.pi / 6)),
    xytext=(0.87 * np.cos(np.pi / 6), 0.87 * np.sin(np.pi / 6)),
    arrowprops=dict(arrowstyle="<->", lw=0.5, color="b"),
    size=5,
)
ax.text(
    1.05 * np.cos(np.pi / 6),
    1.05 * np.sin(np.pi / 6),
    "matching field boundaries",
    c="b",
    size=5,
)
ax.text(
    1.05 * np.cos(np.pi / 6),
    1.05 * np.sin(np.pi / 6) - 0.04,
    r"$\mathtt{grid.boundaries.match.ds}$",
    c="b",
    size=5,
)
ax.add_patch(
    mpatches.Arc(
        (0, 0),
        0.4,
        0.4,
        angle=0,
        theta1=-90,
        theta2=90,
        color="g",
        lw=0.5,
        ls=":",
    )
)
ax.add_patch(
    mpatches.Arc(
        (0, 0),
        0.5,
        0.5,
        angle=0,
        theta1=-90,
        theta2=90,
        color="g",
        lw=0.5,
        ls=":",
    )
)
ax.annotate(
    "",
    xy=(0.27 / np.sqrt(2), 0.27 / np.sqrt(2)),
    xytext=(0.18 / np.sqrt(2), 0.18 / np.sqrt(2)),
    arrowprops=dict(arrowstyle="<->", lw=0.5, color="g"),
    size=5,
)
ax.text(
    0.27 / np.sqrt(2),
    0.27 / np.sqrt(2) + 0.04,
    "particle atmosphere injection",
    c="g",
    size=5,
)
ax.text(
    0.27 / np.sqrt(2),
    0.27 / np.sqrt(2),
    r"$\mathtt{grid.boundaries.atmosphere.height}$",
    c="g",
    size=5,
)
ax.annotate(
    "",
    xy=(0.13 / np.sqrt(2), -0.13 / np.sqrt(2)),
    xytext=(0.22 / np.sqrt(2), -0.22 / np.sqrt(2)),
    arrowprops=dict(arrowstyle="<->", lw=0.5, color="b"),
    size=5,
)
ax.text(
    0.23 / np.sqrt(2),
    -0.23 / np.sqrt(2),
    "buffer zone for resetting fields",
    c="b",
    size=5,
)
ax.text(
    0.23 / np.sqrt(2),
    -0.23 / np.sqrt(2)-0.04,
    r"size in cells = \# of filters",
    c="b",
    size=5,
)
ax.set(
    xlim=(-0.05, 1.05),
    ylim=(-1.05, 1.05),
    aspect=1,
    xticks=[],
    yticks=[],
    frame_on=False,
)
plt.savefig("sketch.png", bbox_inches="tight")
