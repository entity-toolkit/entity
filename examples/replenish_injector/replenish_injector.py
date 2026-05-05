import nt2
import matplotlib.pyplot as plt


def plot(t, data):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.dpi"] = 250
    ax = plt.gca()
    data.fields.N_1_2.sel(t=t, method="nearest").plot(
        vmin=0, vmax=1.5, ax=ax, cmap="inferno"
    )
    plt.gcf().axes[1].set_ylabel(r"$n_-+n_+$")

    ax.annotate(
        "",
        xy=(0.08, 0.75),
        xycoords="axes fraction",
        xytext=(0.08, 0.55),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#e05c2e", lw=1.8),
    )
    ax.text(
        0.095,
        0.65,
        r"$E_y$",
        transform=ax.transAxes,
        color="#e05c2e",
        fontsize=12,
        va="center",
    )

    circle = plt.Circle(
        (0.08, 0.35),
        0.035,
        transform=ax.transAxes,
        fill=False,
        color="#2e80e0",
        lw=1.8,
        clip_on=False,
    )
    ax.add_patch(circle)
    ax.plot(0.08, 0.35, ".", transform=ax.transAxes, color="#2e80e0", markersize=6)
    ax.text(
        0.125,
        0.35,
        r"$B_z$",
        transform=ax.transAxes,
        color="#2e80e0",
        fontsize=12,
        va="center",
    )
    ax.set(
        xlabel=r"$x$",
        ylabel=r"$y$",
        title=rf"$t={{{t:.2f}}}$",
        xlim=(0, 1),
        ylim=(0, 1),
    )


data_uniform_replenish = nt2.Data("uniform_replenish")
data_uniform_replenish.makeMovie(plot, framerate=10)

data_nonuniform_replenish = nt2.Data("nonuniform_replenish")
data_nonuniform_replenish.makeMovie(plot, framerate=10)
