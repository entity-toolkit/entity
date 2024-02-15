import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import nt2.read as nt2r

data = nt2r.Data("<PATH-TO>/DepositSR.h5")


def frame(ti, data):
    t0 = data.isel(t=ti).t.values[()]
    vmax = 1e-8
    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    data.Er.isel(t=ti).polar.pcolor(
        ax=ax,
        norm=mpl.colors.SymLogNorm(
            vmin=-vmax, vmax=vmax, linthresh=vmax / 1e2, linscale=1
        ),
        cmap="RdBu_r",
    )
    x1, x2 = data.particles[1].isel(t=ti).r * np.sin(
        data.particles[1].isel(t=ti).th
    ), data.particles[1].isel(t=ti).r * np.cos(data.particles[1].isel(t=ti).th)
    if len(x1) > 0:
        ax.scatter(x1, x2, s=5, c="b", ec="w", lw=0.5)
    x1, x2 = data.particles[2].isel(t=ti).r * np.sin(
        data.particles[2].isel(t=ti).th
    ), data.particles[2].isel(t=ti).r * np.cos(data.particles[2].isel(t=ti).th)
    if len(x1) > 0:
        ax.scatter(x1, x2, s=5, c="r", ec="w", lw=0.5)
    ax.add_artist(
        mpl.patches.Arc(
            (0, 0),
            2 * 10,
            2 * 10,
            fill=False,
            ec="k",
            ls=":",
            lw=0.5,
            theta1=-90,
            theta2=90,
        )
    )
    axins = ax.inset_axes([0.4, 0.8, 0.6 - 0.025, 0.2 - 0.01])
    (
        data.Er.sel(t=slice(data.t.min(), t0)).sel(r=10, method="nearest")
        * np.sin(data.th)
    ).sum(dim="th").plot(ax=axins)
    axins.set(
        ylim=(-3 * vmax, 3 * vmax),
        xlim=(data.t.min(), data.t.max()),
        title="",
        ylabel=rf"$\int E_r d\Omega$ @ (r = 10)",
    )
    axins.axvline(t0, color="b", lw=0.5, ls=":")
    axins.axhline(0, color="r", lw=0.5, ls="--")
