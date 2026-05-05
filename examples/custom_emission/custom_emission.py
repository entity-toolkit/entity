import nt2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True

data = nt2.Data("custom_emission")


def plot(t, data):
    fig = plt.figure(figsize=(8, 4), dpi=150)
    ax1 = fig.add_subplot(121)
    prtls = data.particles.sel(t=slice(t - 0.5, t)).load()
    prtls.plot.scatter(
        ax=ax1,
        x="x",
        y="y",
        color=["r" if sp == 1 else "b" for sp in prtls["sp"]],
        ec=None,
        alpha=(1 - (t - prtls["t"]) / 0.5) ** 2,
        s=(1 - (t - prtls["t"]) / 0.5) * 5,
    )
    ax1.set(xlabel=r"$x$", ylabel=r"$y$", xlim=(-1, 1), ylim=(-1, 1), aspect=1)

    ax2 = fig.add_subplot(122)
    prtls1 = data.particles.sel(sp=1, t=slice(None, t)).load()
    prtls2 = data.particles.sel(sp=2, t=slice(None, t)).load()
    e1 = (
        prtls1.assign(
            e=np.sqrt(1 + prtls1["ux"] ** 2 + prtls1["uy"] ** 2 + prtls1["uz"] ** 2)
        )
        .groupby("t", as_index=False)["e"]
        .sum()
    )
    e2 = (
        prtls2.assign(
            e=np.sqrt(prtls2["ux"] ** 2 + prtls2["uy"] ** 2 + prtls2["uz"] ** 2)
        )
        .groupby("t", as_index=False)["e"]
        .sum()
    )
    e = e1.merge(e2, on="t", how="outer", suffixes=("_1", "_2")).fillna(0)
    ax2.plot(e.t, e.e_1, label="emitters", c="r")
    ax2.plot(e.t, e.e_2, label="emitted", c="b")
    ax2.plot(e.t, e.e_1 + e.e_2, label="total", c="k")
    ax2.set(xlabel=r"$t$", ylabel=r"total energy", xlim=(0, 5), ylim=(0, 3))
    ax2.axvline(t, color="gray", ls="--")
    ax2.legend(loc="center left")


data.makeMovie(plot, framerate=30)
