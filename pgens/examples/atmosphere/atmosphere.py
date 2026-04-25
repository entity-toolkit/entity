import nt2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (6, 2)
plt.rcParams["font.family"] = "serif"

data = nt2.Data("atmosphere")

data.fields.coords["xh"] = (
    data.fields.coords["x"] / data.attrs["grid.boundaries.atmosphere.height"]
)

t = 1

data.fields.N_1_2.sel(t=t, method="nearest").plot(x="xh", lw=1)

xs = np.linspace(0, 20, 100)
plt.plot(
    xs,
    data.attrs["grid.boundaries.atmosphere.density"] * np.exp(-xs),
    label=r"$n_{\rm max} \exp{\{-x/h\}}$",
    lw=1,
    ls="--",
    c="k",
)
plt.title(rf"$t={{{t:.2f}}}$")
plt.ylabel(r"$n_-+n_+$")
plt.xlabel(r"$x/h$")
plt.xlim(0, 20)
plt.yscale("log")
plt.ylim(1e-2, 20)

plt.savefig("atmosphere.png", bbox_inches="tight")
