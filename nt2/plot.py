def annotatePulsar(
    ax, data, rmax, rstar=1.1, ti=None, time=None, attrs={}, ax_props={}, star_props={}
):
    import numpy as np
    import matplotlib as mpl

    if ti is None and time is None:
        raise ValueError("Must provide either ti or time")
    if time is None:
        time = data.isel(t=ti).t.values[()]
    if (
        omega := attrs.get("psr_omega", data.attrs.get("problem/psr_omega", None))
    ) is None or (
        spinup := abs(
            attrs.get(
                "psr_spinup_time", data.attrs.get("problem/psr_spinup_time", None)
            )
        )
    ) is None:
        print(
            "WARNING: No spinup time or spin period found, please specify explicitly as `attrs = {'psr_omega': ..., 'psr_spinup_time': ...}`"
        )
        demo_rotation = False
    else:
        phase = (
            omega
            * spinup
            * (0.5 * (time / spinup) ** 2 if (time < spinup) else (time / spinup - 0.5))
        )
        demo_rotation = True

    if omega is not None:
        title = rf"$t={{{time / (2 * np.pi / omega):.2f}}}~P$"
    else:
        title = rf"$t={{{time:.2f}}}$"

    ax.set_title(title, fontsize=8)
    ax.annotate(
        "",
        xy=(0.0, rmax * 0.95),
        xytext=(0.0, -rmax * 0.95),
        zorder=4,
        arrowprops=dict(arrowstyle="->", color=ax_props.get("color", "k"), lw=0.5),
    )
    for i in range(-int(rmax * 0.8) // 2 - 1, int(rmax * 0.8) // 2):
        if i != -1:
            ax.add_artist(
                mpl.lines.Line2D(
                    [0, -0.1],
                    [2 * (i + 1), 2 * (i + 1)],
                    color=ax_props.get("color", "k"),
                    lw=0.5,
                )
            )
        if i >= 0:
            ax.text(
                -0.2,
                2 * (i + 1),
                rf"${{{2 * (i + 1)}}}$",
                color=ax_props.get("color", "k"),
                fontsize=ax_props.get("fontsize", 8),
                ha="right",
                va="center",
            )

    ax.text(
        -rmax * 0.025,
        rmax * 0.925,
        r"$r/R_*$",
        color=ax_props.get("color", "k"),
        fontsize=ax_props.get("fontsize", 8),
        ha="right",
        va="center",
    )
    ax.axis("off")

    if demo_rotation:
        phi0 = phase
        dphi = 0.25
        thetas = np.linspace(0, np.pi, 100)
        xs1 = np.sin(thetas) * np.cos(phi0)
        ys1 = np.cos(thetas)
        xs2 = np.sin(thetas) * np.cos(phi0 + dphi)
        ys2 = np.cos(thetas)
        xs = np.concatenate([xs1, xs2[::-1]])
        ys = np.concatenate([ys1, ys2[::-1]])
        ax.add_artist(
            mpl.patches.Polygon(
                (rstar + 0.02) * np.array([xs, ys]).T,
                color=star_props.get("c1", "r"),
                lw=0,
                zorder=6,
                alpha=0.25 if (phi0 // np.pi) % 2 != 0 else 1.0,
            )
        )
        ax.add_artist(
            mpl.patches.Polygon(
                (rstar + 0.02) * np.array([-xs, ys]).T,
                color=star_props.get("c2", "b"),
                lw=0,
                zorder=6,
                alpha=0.25 if (phi0 // np.pi) % 2 == 0 else 1.0,
            )
        )
    ax.add_artist(
        mpl.patches.Circle(
            (0, 0),
            rstar,
            color=star_props.get("color", "royalblue"),
            fill=True,
            zorder=5,
        )
    )
