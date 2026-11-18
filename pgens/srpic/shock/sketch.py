import matplotlib.pyplot as plt
import myplotlib

plt.style.use("latex")

fig = plt.figure(figsize=(14, 6), dpi=600)
ax = fig.add_subplot(111)

ax.plot()
ax.set(
    xlim=(0, 10),
    ylim=(-2, 2),
    aspect=1,
)
ax.text(
    0.1,
    0.5 + 0.15,
    r'$\mathtt{grid.boundaries.fields[0][0]: ``CONDUCTOR"}$',
    ha="left",
    va="center",
    c="C3",
)
ax.text(
    0.1,
    0.5 - 0.1,
    r'$\mathtt{grid.boundaries.particles[0][0]: ``REFLECT"}$',
    ha="left",
    va="center",
    c="C3",
)
ax.text(
    0.1,
    1.95,
    r'$\mathtt{grid.boundaries.fields[1]: ``PERIODIC"}$',
    ha="left",
    va="top",
    c="C2",
)
ax.text(
    0.1,
    1.75,
    r'$\mathtt{grid.boundaries.particles[1]: ``PERIODIC"}$',
    ha="left",
    va="top",
    c="C2",
)
ax.text(
    10 - 0.1,
    0.15 - 0.5,
    r'$\mathtt{grid.boundaries.fields[0][1]: ``MATCH"}$',
    ha="right",
    va="center",
    c="C1",
)
ax.text(
    10 - 0.1,
    -0.1 - 0.5,
    r'$\mathtt{grid.boundaries.particles[0][1]: ``ABSORB"}$',
    ha="right",
    va="center",
    c="C1",
)
ax.spines["left"].set_color("C3")
ax.spines["top"].set_color("C2")
ax.spines["bottom"].set_color("C2")
ax.spines["right"].set_color("C1")

ax.text(
    2, -1.45, r"$\mathtt{setup.filling\_fraction}$", ha="center", va="bottom", c="C0"
)
ax.annotate("", (0, -1.5), (4, -1.5), arrowprops=dict(arrowstyle="<->", color="C0"))
ax.annotate("", (4, -1.35), (4.5, -1.35), arrowprops=dict(arrowstyle="<|-", color="C0"))
ax.text(
    4.25, -1.3, r"$\mathtt{setup.injector\_velocity}$", ha="left", va="bottom", c="C0"
)
ax.annotate("", (4.5, 0.5), (6, 1.75), arrowprops=dict(arrowstyle="<-", color="C4"))
ax.plot([4.5, 5], [0.5, 0.5], c="C4", ls="--", lw=1)
ax.text(
    5.5,
    1.2,
    r"initial (upstream) B-field of strength $\mathtt{setup.Bmag}\times B_0$",
    ha="left",
    va="center",
    c="C4",
)
ax.text(
    5,
    0.7,
    r"angle w.r.t. the shock normal: $\mathtt{setup.Btheta}$",
    ha="left",
    va="center",
    c="C4",
)
ax.text(
    5,
    0.3,
    r"out-of-plane angle: $\mathtt{setup.Bphi}$",
    ha="left",
    va="center",
    c="C4",
)

ax.axvline(4, c="C0", ls="--")

ax.annotate("", (2, -0.75), (0.75, -0.75), arrowprops=dict(arrowstyle="<|-", color="C5"))
ax.text(
    0.75,
    -0.1,
    r"upstream plasma:",
    ha="left",
    va="center",
    c="C5",
)
ax.text(
    0.75,
    -0.1-0.15,
    r"$u_x = \mathtt{setup.drift\_ux}$",
    ha="left",
    va="center",
    c="C5",
)
ax.text(
    0.75,
    -0.1-2 * 0.15,
    r"$T_-/m_0 c^2 = \mathtt{setup.temperature}$",
    ha="left",
    va="center",
    c="C5",
)
ax.text(
    0.75,
    -0.1-3 * 0.15,
    r"$T_i/m_0 c^2 = \mathtt{setup.temperature} \times \mathtt{setup.temperature\_ratio}$",
    ha="left",
    va="center",
    c="C5",
)
ax.set(
    xticks=[], yticks=[]
)
plt.savefig("sketch.png", bbox_inches="tight")
