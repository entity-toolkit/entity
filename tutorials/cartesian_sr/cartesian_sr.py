import matplotlib.pyplot as plt
import nt2
import numpy as np


def get_dipole(xs, ys):
    xx, yy = np.meshgrid(xs, ys)
    rr = np.sqrt(xx**2 + yy**2)
    bx = 3 * xx * yy / rr**5
    by = (3 * yy**2 - rr**2) / rr**5
    return bx, by


def plot(t, data):
    fig = plt.figure(figsize=(6, 5), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])

    d = data.fields.sel(t=t, method="nearest")
    (d.N_1 + d.N_2).plot(ax=ax, vmin=0, vmax=20, cmap="inferno", add_colorbar=False)
    cbar_ticks = np.linspace(0, 20, 50)
    ax_cbar.pcolormesh(
        [0, 1],
        cbar_ticks,
        np.array([cbar_ticks] * 2).T,
        cmap="inferno",
        vmin=0,
        vmax=20,
        rasterized=True,
    )
    ax_cbar.yaxis.tick_right()
    ax_cbar.yaxis.set_label_position("right")
    ax_cbar.set(xticks=[], ylabel=r"$n_\pm$")
    for spine in ax_cbar.spines.values():
        spine.set_visible(False)

    ys = np.linspace(-2.9, 2.9, 50)
    xs = -0.5 * np.ones_like(ys)

    bx, by = get_dipole(data.fields.x, data.fields.y)
    ax.streamplot(
        data.fields.x.values,
        data.fields.y.values,
        d.Bx.values + bx,
        d.By.values + by,
        color="#ffffff90",
        linewidth=0.25,
        zorder=10,
        arrowstyle="->",
        arrowsize=0.5,
        density=30,
        start_points=np.array([xs, ys]).T,
    )
    ax.add_artist(
        plt.Circle((0, 0), data.attrs["setup.r_plummet"], color="C0", zorder=20)
    )

    ax.set(
        xlim=(-4, 2),
        ylim=(-3, 3),
        aspect=1,
        xlabel="$x$",
        ylabel="$y$",
        title=f"$t = {t:.2f}$",
    )


data = nt2.Data("cartesian_sr")
data.makeMovie()
