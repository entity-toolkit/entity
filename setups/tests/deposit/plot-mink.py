import nt2
import matplotlib.pyplot as plt
import matplotlib as mpl

datas = []
cpus = [1, 2, 3, 4, 5, 6, 8]
for i in cpus:
    datas.append(nt2.Data(path=f"mink-np{i}"))


def plot(ti):
    fig = plt.figure(figsize=(16, 7), dpi=300)
    gs = mpl.gridspec.GridSpec(3, 7, figure=fig)

    for p, quant in enumerate(["Jx", "Jy", "Jz"]):
        axs = [fig.add_subplot(gs[p, i]) for i in range(7)]
        (datas[0].fields[quant]).isel(t=ti).plot(
            ax=axs[0],
            cmap="seismic",
            add_colorbar=False,
            norm=mpl.colors.SymLogNorm(
                linthresh=1e-5,
                linscale=1,
                vmin=-1e-2,
                vmax=1e-2,
            ),
        )
        for i, (d, ax) in enumerate(zip(datas[1:], axs[1:])):
            (d.fields[quant] - datas[0].fields[quant]).isel(t=ti).plot(
                ax=ax,
                cmap="seismic",
                add_colorbar=False,
                norm=mpl.colors.SymLogNorm(
                    linthresh=1e-10,
                    linscale=1,
                    vmin=-1e-7,
                    vmax=1e-7,
                ),
            )

        for i, ax in enumerate(axs):
            ax.set_aspect(1)
            if i > 0:
                if p == 0:
                    ax.set_title(f"np{cpus[i]} - np1")
                else:
                    ax.set_title(None)
                ax.set_yticklabels([])
                ax.set_ylabel(None)
            else:
                if p == 0:
                    ax.set_title(f"np1")
                else:
                    ax.set_title(None)

            if p != 2:
                ax.set_xticklabels([])
                ax.set_xlabel(None)


nt2.export.makeFrames(plot, datas[0].fields.s[::4], "mink-diff", num_cpus=4)
nt2.export.makeMovie(framerate=10, input="mink-diff/", number=5, output="mink-diff.mp4")
