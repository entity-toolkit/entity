import matplotlib.pyplot as plt
import matplotlib as mpl
import nt2.read as nt2r

data = nt2r.Data("<PATH-TO>/Reconnection.h5")


def plot(ti, data):
    select = lambda d: d.isel(t=ti, x=slice(None, None, 4), y=slice(None, None, 4))

    sigma0 = (data.attrs["units/skindepth0"] / data.attrs["units/larmor0"]) ** 2
    EM_energy = (sigma0 / 2) * (
        (data.Bx**2 + data.By**2 + data.Bz**2)
        + (data.Ex**2 + data.Ey**2 + data.Ez**2)
    ).mean(("x", "y")).isel(t=slice(None, None, 10))
    prtl_energy = (data.E).mean(("x", "y")).isel(t=slice(None, None, 10))

    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(221)

    select(data.Rho).plot(cmap="inferno", norm=mpl.colors.LogNorm(vmin=0.25, vmax=10))
    ax.set(aspect="equal", title=r"$\rho$")
    fig.axes[1].set_ylabel("")

    ax = fig.add_subplot(222)
    EM_energy.plot(ax=ax, label=r"$(E^2+B^2)/8\pi$")
    prtl_energy.plot(ax=ax, label=r"particle energy")
    (EM_energy + prtl_energy).plot(ax=ax, label="total")
    ax.axvline(data.isel(t=ti).t, color="k", linestyle="--")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(223)
    select((data.Ez * data.By) / (data.Bx**2 + data.By**2 + data.Bz**2)).plot(
        ax=ax, vmin=-0.25, vmax=0.25, cmap="RdBu_r"
    )
    ax.set(aspect="equal", title=r"$E_z B_y / B^2$")

    ax = fig.add_subplot(224)
    select(data.E / data.Rho).plot(ax=ax, norm=mpl.colors.LogNorm(1, 20), cmap="afmhot")
    ax.set(aspect="equal", title=r"$\langle\gamma\rangle$", yticklabels=[], ylabel="")

    fig.suptitle(rf"$t = {{{data.isel(t=ti).t.values:.2f}}}$", y=0.95, fontsize=15)
