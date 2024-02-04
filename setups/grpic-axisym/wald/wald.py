import matplotlib.pyplot as plt
import numpy as np
import nt2.read as nt2r

data = nt2r.Data("<PATH-TO>/Wald.h5")


def plot(ti, data):
    selector = lambda d: d.isel(t=ti).sel(r=slice(None, 10, 2), th=slice(None, None, 2))

    fig = plt.figure(figsize=(5, 8), dpi=150)
    ax = fig.add_subplot(111)

    selector(data).Dr.polar.pcolor(ax=ax, vmin=-0.1, vmax=0.1, cmap="RdBu_r")
    selector(data).Aph.polar.contour(
        ax=ax, colors="black", levels=np.linspace(0, 10), linewidths=0.5
    )
