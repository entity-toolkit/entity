import nt2
import matplotlib.pyplot as plt
import numpy as np

data = nt2.Data("compton_jones")

photons = data.particles.sel(sp=3).isel(t=-1).load()
photons = photons[np.sqrt(photons.ux**2 + photons.uy**2 + photons.uz**2) > 0.01]

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "serif"

fig = plt.figure(figsize=(9, 4))
gs = fig.add_gridspec(1, 2, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

gamma = np.sqrt(1 + data.attrs["setup.electron_4vel"] ** 2)
e0 = data.attrs["setup.photon_energy"]
Gamma = 4 * e0 * gamma
emax = gamma * Gamma / (1 + Gamma)

es = data.spectra.E[1:-1] / emax

dnde = data.spectra.N_3.isel(t=-1)[1:-1]
dnde /= np.trapezoid(dnde, es)
ax1.plot(es, dnde)

es = np.linspace(es.values.min(), es.values.max(), 250)
qs = es / (1 + Gamma * (1 - es))
dnde_th = (
    2 * qs * np.log(qs)
    + (1 + 2 * qs) * (1 - qs)
    + 0.5 * Gamma**2 * qs**2 / (1 + Gamma * qs) * (1 - qs)
)

dnde_th /= np.trapezoid(dnde_th, es)

ax1.plot(es, dnde_th, c="k", ls=":")
ax1.set(
    xlim=(0, 1),
    ylim=(0, 4),
    xlabel=r"$\varepsilon_{\rm ph} / \varepsilon_{\rm max}$",
    ylabel=r"$dn_{\rm ph}/d\varepsilon_{\rm ph}$",
)

plt.scatter(
    photons.ux / emax,
    photons.uy / emax,
    s=1,
    linewidth=0,
)
xs = np.linspace(0, 1, 100)
ys = 2 / gamma * xs
ax2.plot(xs, ys, c="k", ls="--", lw=0.5)
ax2.plot(xs, -ys, c="k", ls="--", lw=0.5)
ax2.set(
    xlabel=r"$p_{\rm ph}^x / \varepsilon_{\rm max}$",
    ylabel=r"$p_{\rm ph}^y / \varepsilon_{\rm max}$",
)

plt.savefig("compton_jones.png", bbox_inches="tight")
