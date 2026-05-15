import nt2
import matplotlib.pyplot as plt

data_fix = nt2.Data("fix_field_boundaries")
data_match = nt2.Data("match_field_boundaries")


def plot(t, _):
    plt.rcParams["text.usetex"] = True
    omega = data_fix.attrs["setup.omega"]

    data_fix.fields.coords["xo"] = data_fix.fields.coords["x"] * omega
    data_match.fields.coords["xo"] = data_match.fields.coords["x"] * omega
    xmax = data_match.fields.coords["xo"].max().values[()]
    ds = data_match.attrs["grid.boundaries.match.ds"][0] * omega

    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax1 = fig.add_subplot(211)
    data_fix.fields.Ez.sel(t=t, method="nearest").plot(ax=ax1, label="$E_z$", x="xo")
    data_fix.fields.By.sel(t=t, method="nearest").plot(ax=ax1, label="$B_y$", x="xo")

    ax2 = fig.add_subplot(212)
    data_match.fields.Ez.sel(t=t, method="nearest").plot(ax=ax2, label="$E_z$", x="xo")
    data_match.fields.By.sel(t=t, method="nearest").plot(ax=ax2, label="$B_y$", x="xo")

    for ax in [ax1, ax2]:
        ax.set(
            xlim=(0, omega),
            ylim=(-1.1, 1.1),
            ylabel="",
            title=None,
        )

    ax1.set(xticklabels=[], xlabel="")
    ax2.set(xlabel=r"$x$ [$c / \omega$]")
    ax1.set(title=rf"$\omega t={omega * t:.1f}$")

    for ax, label in [(ax1, "FIXED"), (ax2, "MATCH")]:
        ax.text(
            0.99,
            0.94,
            r"\texttt{" + label + "}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    plt.subplots_adjust(hspace=0.35)

    mid_y = (ax1.get_position().y0 + ax2.get_position().y1) / 2
    right_x = ax1.get_position().x1

    handles, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(right_x, mid_y),
        bbox_transform=fig.transFigure,
        ncol=1,
        frameon=True,
    )
    leg.set_zorder(10)
    ax2.fill_between(
        [xmax - ds, xmax],
        -1.1,
        1.1,
        color="grey",
        alpha=0.3,
        transform=ax2.get_xaxis_transform(),
    )
    plt.tight_layout()


data_fix.makeMovie(plot, framerate=15)
