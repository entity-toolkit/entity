import matplotlib.pyplot as plt
import myplotlib

plt.style.use("latex")

fig = plt.figure(dpi=600)
ax = fig.add_subplot(111)

ax.set(
    xlim=(0, 2),
    ylim=(0, 1),
    aspect=1,
)
ax.text(0.2, 0.5, r"i-th species:", c="C0")
ax.text(0.2, 0.5 - 0.06, r"$u_D^i = \mathtt{setup.drifts\_in\_*[i]}$", c="C0")
ax.text(0.2, 0.5 - 2 * 0.06, r"$n_i=\mathtt{setup.densities[i / 2]}$", c="C0")
ax.text(0.2, 0.5 - 3 * 0.06, r"$T_i/m_0c^2=\mathtt{setup.temperatures[i]}$", c="C0")
ax.annotate("", (0.5, 0.2), (0.1, 0.25), arrowprops=dict(arrowstyle="->", color="C0"))

ax.text(0.95, 0.8, r"i+1-th species:", c="C2")
ax.text(0.95, 0.8 - 0.06, r"$u_D^{i+1} = \mathtt{setup.drifts\_in\_*[i+1]}$", c="C2")
ax.text(0.95, 0.8 - 2 * 0.06, r"$n_{i+1}=\mathtt{setup.densities[(i + 1) / 2]}$", c="C2")
ax.text(0.95, 0.8 - 3 * 0.06, r"$T_{i+1}/m_0c^2=\mathtt{setup.temperatures[i+1]}$", c="C2")
ax.annotate("", (1.25, 0.4), (1.75, 0.5), arrowprops=dict(arrowstyle="->", color="C2"))

ax.text(1.0, 0.98, r"periodic boundaries", c="C3", ha="center", va="top")
ax.spines["top"].set_color("C3")
ax.spines["bottom"].set_color("C3")
ax.spines["left"].set_color("C3")
ax.spines["right"].set_color("C3")
ax.set(xticks=[], yticks=[])

plt.savefig("sketch.png", bbox_inches="tight")
