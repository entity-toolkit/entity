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

    ax.set(
        xlabel=r"$x$",
        ylabel=r"$y$",
        title=rf"$t={{{t:.2f}}}$",
        xlim=(0, 1),
        ylim=(0, 1),
    )


piston = nt2.Data("piston")
piston.makeMovie(plot, framerate=10)
