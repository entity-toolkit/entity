import matplotlib.pyplot as plt
import numpy as np
import nt2.read as nt2r

data = nt2r.Data("<PATH-TO>/Weibel.h5")

def plot(ti, data):
    select = lambda f: f.isel(t=ti)

    fig = plt.figure(figsize=(11, 10), dpi=150)
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.1)
    axs = gs.subplots()

    select(data).Bz.plot(ax=axs[0][0], vmin=-0.5, vmax=0.5, cmap="RdBu_r")
    select(data).Bx.plot(ax=axs[0][1], vmin=-2, vmax=2, cmap="RdBu_r")
    select(data).Rho.plot(ax=axs[1][0], vmin=0, vmax=100, cmap="inferno")

    for cbar, ax in zip(fig.axes[-3:], [axs[0][0], axs[0][1], axs[1][0]]):
        ax.set(title=cbar.get_ylabel())
        cbar.set(ylabel="")

    for ax in [axs[0][0], axs[0][1], axs[1][0]]:
        ax.set(aspect="equal")

    axs[0][0].set(xticklabels=[], xlabel="")
    axs[0][1].set(yticklabels=[], ylabel="")

    (data.Bx**2 + data.By**2 + data.Bz**2).mean(("x", "y")).plot(ax=axs[1][1])
    axs[1][1].set(ylabel=r"$\langle B^2\rangle$", yscale="log", ylim=(1e-8, 10))

    omega_p0 = 1 / data.attrs["units/skindepth0"]
    omega_p = omega_p0
    gamma = data.attrs["problem/drift_b"]
    beta = np.sqrt(1 - data.attrs["problem/drift_b"] ** -2)
    alpha = 1
    Bs = beta * omega_p * alpha * (alpha + 1)
    gamma = omega_p * np.sqrt(2 / gamma) * beta
    ts = data.t.values[()]
    B2s = 1e-6 * np.exp(ts * gamma)
    axs[1][1].plot(ts, B2s)
    axs[1][1].axhline(Bs**2, c="k", ls="--")

    fig.suptitle(rf"$t={{{select(data).t:.2f}}}$", y=0.95, fontsize=15)
