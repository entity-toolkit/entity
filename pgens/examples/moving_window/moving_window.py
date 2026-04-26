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
        xy=(0.28, 0.08),
        xycoords="axes fraction",
        xytext=(0.08, 0.08),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#e05c2e", lw=1.8),
    )
    ax.text(
        0.095,
        0.095,
        r"$v_0$",
        transform=ax.transAxes,
        color="#e05c2e",
        fontsize=12,
        va="center",
    )
    ax.set(
        xlabel=r"$x$",
        ylabel=r"$y$",
        title=rf"$t={{{t:.2f}}}$",
    )


moving_window = nt2.Data("moving_window")
moving_window.makeMovie(plot, framerate=10)
