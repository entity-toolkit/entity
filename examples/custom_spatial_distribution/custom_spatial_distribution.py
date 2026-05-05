import nt2
import matplotlib.pyplot as plt

data = nt2.Data("custom_spatial_distribution")

def plot(t, data):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.dpi"] = 150
    data.fields.N_1_2.sel(t=t).plot(vmin=0, vmax=1.5)
    plt.gca().set(xlabel=r"$x$", ylabel=r"$y$", title=rf"$t={{{t:.2f}}}$")
    plt.gcf().axes[1].set_ylabel(r"$n_1+n_2$")
    plt.gca().set_aspect(1)
    plt.tight_layout()

data.makeMovie(plot, framerate=30)
