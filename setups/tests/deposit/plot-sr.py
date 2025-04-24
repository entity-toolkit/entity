import nt2
import matplotlib.pyplot as plt
import matplotlib as mpl

data = nt2.Data(path=f"sr-np8")


def plot(ti):
    fig = plt.figure(figsize=(9, 6), dpi=300)
    gs = mpl.gridspec.GridSpec(1, 3, figure=fig)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i, (ax, j) in enumerate(zip(axs, ["Jr", "Jth", "Jph"])):
        data.fields.isel(t=ti)[j].polar.pcolor(
            ax=ax,
            cbar_position="top",
            cbar_size="2%",
            norm=mpl.colors.SymLogNorm(linthresh=1e-8, vmin=-1e-4, vmax=1e-4),
            cmap="seismic",
        )
        ax.set_title(None)
        ax.add_artist(mpl.patches.Circle((0, 0), 1, color="k", alpha=0.2))
        ax.add_artist(mpl.patches.Circle((0, 0), 5, edgecolor="k", facecolor="none"))
        if i > 0:
            ax.set_yticklabels([])
            ax.set_ylabel(None)


nt2.export.makeFrames(plot, data.fields.s, "sr-dep", num_cpus=4)
nt2.export.makeMovie(framerate=10, input="sr-dep/", number=5, output="sr-dep.mp4")
