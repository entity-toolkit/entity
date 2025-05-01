import nt2.read as nt2r
import matplotlib.pyplot as plt

data = nt2r.Data("em_vacuum.h5")


def plot(ti):
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = fig.add_subplot(121)
    data.Bz.isel(t=ti).plot(ax=ax, cmap="BrBG")
    ax = fig.add_subplot(122)
    data.Ey.isel(t=ti).plot(ax=ax, cmap="RdBu_r")
    for ax in fig.axes[::2]:
        ax.set_aspect("equal")
