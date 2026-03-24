import nt2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (6, 2)

data = nt2.Data("atmosphere")

f = data.fields.N_1_2.isel(t=-1)

f.plot()
xs = np.linspace(0, 2, 100)
plt.plot(
    xs,
    data.attrs["grid.boundaries.atmosphere.density"]
    * np.exp(-xs / data.attrs["grid.boundaries.atmosphere.height"]),
    label=r"$n_{\rm max} \exp{\{-x/h\}}$",
)
plt.title(rf"$t={{{f.t:.2f}}}$")
plt.ylabel(r"$n_-+n_+$")
plt.xlabel(r"$x$")
plt.xlim(0, 2)
plt.yscale("log")
plt.legend()
plt.ylim(1e-2, 20)
