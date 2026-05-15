import nt2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = nt2.Data("compton_kompaneets")
stats = pd.read_csv("compton_kompaneets/compton_kompaneets_stats.csv")
stats.columns = stats.columns.str.strip()

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"

fig = plt.figure(figsize=(9, 4))
gs = fig.add_gridspec(1, 2, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])

tC = 1 / (
    data.attrs["setup.temperature"] * data.attrs["two_body.thomson_optical_depth"]
)

tvals = len(data.spectra.t.values)

nphot = data.spectra.N_3.isel(t=-1).sum().values[()]
for ti, t in enumerate([0, 0.5 * tC, tC, 3 * tC]):
    ax1.plot(
        data.spectra.E.coarsen(E=2).mean().values / data.attrs["setup.temperature"],
        data.spectra.N_3.sel(t=t, method="nearest").coarsen(E=2).mean().values,
        c=plt.get_cmap("viridis")(ti / 3),
        label=f"$y={t / tC:.1f}$",
        lw=1,
    )

es = data.spectra.E.values / data.attrs["setup.temperature"]
dndes = es**2 * np.exp(-es)
dndes /= np.sum(dndes)
dndes *= nphot
ax1.plot(
    es,
    dndes,
    c="k",
    ls=":",
    label=r"$\propto \varepsilon_{\rm ph}^2 e^{-\varepsilon_{\rm ph} / T_\pm}$",
)

ax1.set(
    yscale="log",
    ylim=(1, 1e5),
    xlim=(0, 10),
    xlabel=r"$\varepsilon / T_\pm$",
    ylabel=r"$dn_{\rm ph}/d\varepsilon$",
)
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(stats["time"] / tC, stats["T00_3"], c="C0")
ax2.set(ylabel=r"total photon energy", xlabel=r"$y\equiv t/t_C$")
ax2.yaxis.label.set_color("C0")
ax2.tick_params(axis="y", labelcolor="C0")
ax2twin = ax2.twinx()
ax2twin.plot(data.spectra.t.values / tC, data.spectra.N_3.sum("E"), c="C2")
ax2twin.set(ylabel=r"photon number")
ax2twin.yaxis.label.set_color("C2")
ax2twin.tick_params(axis="y", labelcolor="C2")

plt.savefig("compton_kompaneets_plot.png", bbox_inches="tight")
