import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import myplotlib

plt.style.use("latex")

fig = plt.figure(dpi=600)
ax = fig.add_subplot(111)
ax.plot()
ax.set(
    xlim=(-1, 1),
    ylim=(-0.5, 0.5),
    xlabel="$x$",
    ylabel="$y$",
    xticks=[],
    yticks=[],
    aspect=1,
)
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.axvline(0.95, color="red", lw=0.5, ls=":")
ax.axvline(-0.95, color="red", lw=0.5, ls=":")
ax.axhline(-0.45, color="blue", lw=0.5, ls=":")
ax.axhline(0.45, color="blue", lw=0.5, ls=":")
ax.axhline(-0.05, color="g", lw=0.5, ls="--")
ax.axhline(0.05, color="g", lw=0.5, ls="--")
ax.axhline(0.35, color="magenta", lw=0.5, ls="--")
ax.axhline(-0.35, color="magenta", lw=0.5, ls="--")
ax.annotate(
    "",
    xy=(-1, 0.25),
    xytext=(-0.95, 0.25),
    arrowprops=dict(arrowstyle="<->", lw=0.5, color="red"),
    size=3,
)
ax.annotate(
    "",
    xy=(-0.5, 0.45),
    xytext=(-0.5, 0.5),
    arrowprops=dict(arrowstyle="<->", lw=0.5, color="b"),
    size=3,
)
ax.text(
    -0.94,
    0.25,
    r"$\mathtt{grid.boundaries.match.ds[0]}$",
    color="red",
    size=5,
    ha="left",
    va="center",
)
ax.text(
    -0.5,
    0.44,
    r"$\mathtt{grid.boundaries.match.ds[1]}$",
    color="blue",
    size=5,
    ha="center",
    va="top",
)
ax.annotate(
    "",
    xy=(-0.5, -0.05),
    xytext=(-0.5, 0.05),
    arrowprops=dict(arrowstyle="<->", lw=0.5, color="g"),
    size=3,
)
ax.annotate(
    "",
    xy=(0.5, 0.35),
    xytext=(0.5, 0.5),
    arrowprops=dict(arrowstyle="<->", lw=0.5, color="magenta"),
    size=3,
)
ax.text(
    -0.5,
    -0.06,
    r"$\mathtt{setup.cs\_width}$",
    color="g",
    size=5,
    ha="center",
    va="top",
)
ax.text(
    0.5,
    0.33,
    r"$\mathtt{setup.inj\_ypad}$",
    color="magenta",
    size=5,
    ha="center",
    va="top",
)
ax.text(
    0.0,
    0.0,
    r"current layer plasma with peak density $\mathtt{setup.cs\_overdensity} \times n_0$",
    color="green",
    size=5,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
)
ax.text(
    0.0,
    -0.25,
    "background plasma with density $n_0$",
    color="green",
    size=5,
    ha="center",
    va="center",
)
ax.annotate(
    "",
    xy=(0.7, 0.2),
    xytext=(0.9, 0.2),
    arrowprops=dict(arrowstyle="<-", lw=0.5, color="red"),
    size=5,
)
ax.annotate(
    "",
    xy=(0.7, -0.2),
    xytext=(0.9, -0.2),
    arrowprops=dict(arrowstyle="->", lw=0.5, color="red"),
    size=5,
)
ax.text(
    0.9,
    0.22,
    r"magnetic field of strength $\mathtt{setup.bg\_B}\times B_0$",
    color="red",
    size=5,
    ha="right",
    va="bottom",
)
ax.text(
    -0.93,
    -0.22,
    "matching fields",
    color="red",
    size=5,
    ha="left",
    va="center",
    rotation=90,
)
ax.text(
    0,
    -0.43,
    "matching fields",
    color="blue",
    size=5,
    ha="center",
    va="bottom",
)
ax.add_patch(mpatches.Circle((0.75, 0.15), 0.02, color="red", fill=False, lw=0.5))
ax.add_patch(mpatches.Circle((0.75, 0.15), 0.005, color="red", fill=True, lw=0.5))

ax.add_patch(mpatches.Circle((0.75, -0.15), 0.02, color="red", fill=False, lw=0.5))
ax.add_patch(mpatches.Circle((0.75, -0.15), 0.005, color="red", fill=True, lw=0.5))
ax.text(
    0.71,
    0.15,
    r"$\mathtt{setup.bg\_Bguide} \times B_0$",
    color="r",
    size=5,
    ha="right",
    va="center",
)
plt.savefig("sketch.png", bbox_inches="tight")
