import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import nt2.read as nt2r
import nt2.plot as nt2p

data = nt2r.Data("<PATH-TO>/Magnetosphere.h5")


def frame(ti, data):
    select = lambda d: d.isel(t=ti).sel(r=slice(0, 10))

    fig = plt.figure(figsize=(13, 10), dpi=300)

    fig.suptitle(
        rf"$t = {{{data.t[ti] * data.attrs['problem/psr_omega'] / (2 * np.pi):.2f}}} P_*$",
        fontsize=15,
        y=0.95,
    )
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.0, hspace=0.2
    )
    axs = gs.subplots().flatten()

    cbar_right = dict(cbar_size="3%", cbar_ticksize=8, cbar_pad=0.35)
    cbar_left = dict(
        cbar_size="3%",
        cbar_ticksize=8,
        cbar_position="left",
        invert_x=True,
        cbar_pad=0.05,
    )
    fld_props = dict(
        norm=mpl.colors.SymLogNorm(vmin=-1, vmax=1, linthresh=1e-3, linscale=1)
    )
    select(data.N_2 - data.N_1).polar.pcolor(
        ax=axs[0],
        norm=mpl.colors.SymLogNorm(vmin=-1, vmax=1, linthresh=1e-3, linscale=1),
        cmap="twilight",
        label=r"$(n_+-n_-)/n_{\rm GJ}^*$",
        **cbar_right,
    )
    select(data.N_2 + data.N_1).polar.pcolor(
        ax=axs[0],
        norm=mpl.colors.LogNorm(vmin=1e-3, vmax=1),
        label=r"$(n_++n_-)/n_{\rm GJ}^*$",
        cmap="inferno",
        **cbar_left,
    )
    for ax, x in zip(axs[1:], ["r", "th", "ph"]):
        invert_x = False
        for f, cmap in zip(["E", "B"], ["RdBu_r", "BrBG"]):
            select(data[f"{f}{x}"]).polar.pcolor(
                ax=ax,
                **fld_props,
                cmap=cmap,
                **(cbar_left if invert_x else cbar_right),
            )
            if x == "ph":
                select(data).polar.fieldplot(
                    "Br",
                    "Bth",
                    sample={
                        "template": "dipole",
                        "nth": 30,
                        "radius": 2.0,
                        "pole": 1 / 10,
                    },
                    invert_x=invert_x,
                    c="k",
                    lw=0.25,
                    zorder=1,
                    ax=ax,
                )
            invert_x = True
    for i, ax in enumerate(axs):
        nt2p.annotatePulsar(
            ax, data, rmax=10, ti=ti, ax_props={"color": "w"} if i == 0 else {}
        )
        ax.set(
            ylabel="",
            yticks=[],
            yticklabels=[],
            title="",
        )

    for ax in fig.axes:
        if ax.get_aspect() == "auto":

            def latexify(f):
                if f.startswith("E") or f.startswith("B"):
                    f = (
                        f.replace("r", "_r")
                        .replace("th", "_\\theta")
                        .replace("ph", "_\\phi")
                    )
                    return rf"${{{f[0]}}}{{{f[1:]}}}$"
                else:
                    return f

            ax.set_title(latexify(ax.get_ylabel()))
            ax.set_ylabel("")
