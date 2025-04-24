import nt2.read as nt2r
import matplotlib.pyplot as plt

data = nt2r.Data("langmuir.h5")


def plot(ti, d):
    # for 2D
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = fig.add_subplot(211)
    d.Rho.isel(t=ti).plot(ax=ax, cmap="inferno", vmin=0, vmax=4)
    ax = fig.add_subplot(212)
    d.Ex.isel(t=ti).plot(ax=ax, cmap="RdBu_r", vmin=-1, vmax=1)
    for ax in fig.get_axes()[::2]:
        ax.set_aspect("equal")
    fig.get_axes()[0].set(xlabel="", xticks=[])
    fig.get_axes()[2].set(title=None)
