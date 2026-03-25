import nt2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (6, 2)
plt.rcParams["font.family"] = "serif"

data_normal = nt2.Data("atmosphere")
data_no_g = nt2.Data("atmosphere_no_g")
data_no_inject = nt2.Data("atmosphere_no_reinject")

for d in [data_normal, data_no_g, data_no_inject]:
    d.fields.coords["xh"] = (
        d.fields.coords["x"] / d.attrs["grid.boundaries.atmosphere.height"]
    )

t = 1

data_normal.fields.N_1_2.sel(t=t, method="nearest").plot(label="normal", x="xh", lw=1)
data_no_g.fields.N_1_2.sel(t=t, method="nearest").plot(label="no gravity", x="xh", lw=1)
data_no_inject.fields.N_1_2.sel(t=t, method="nearest").plot(
    label="no reinjection", x="xh", lw=1
)

xs = np.linspace(0, 20, 100)
plt.plot(
    xs,
    data_normal.attrs["grid.boundaries.atmosphere.density"] * np.exp(-xs),
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
plt.legend()
plt.ylim(1e-2, 20)

plt.savefig("atmosphere.png", bbox_inches="tight")
