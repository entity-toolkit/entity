import sys

sys.path.append("../")

import nt2.read as nt2r
import nt2.export as nt2e
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

data = nt2r.Data("wald.h5")

t = 22
select = lambda dat, t: dat.isel(t=t).sel(r=slice(0, 5))

fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

select(data.Bth, t).polar.pcolor(cmap="RdBu_r", vmin=-25, vmax=25, ax=ax)

ax.add_artist(
    plt.Circle((0, 0), data.attrs["rh"], color="k", fill=False, linewidth=0.5)
)
rs, ths = np.meshgrid(select(data, t).coords["r"], select(data, t).coords["th"])
xs = rs * np.sin(ths)
ys = rs * np.cos(ths)
ax.contour(xs, ys, select(data.Aph, t), levels=20, colors="k", linewidths=0.5)