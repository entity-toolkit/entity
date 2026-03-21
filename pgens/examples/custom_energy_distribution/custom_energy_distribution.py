import nt2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True

data = nt2.Data("custom_energy_distribution")

fig = plt.figure(figsize=(10, 5), dpi=150)
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor("k")
data.particles.sel(sp=1).isel(t=0).phase_plot(ax=ax1, cmap="inferno")
ax1.set(xlabel="$x$", ylabel="$u_x$", ylim=(-1.5, 1.5))

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor("k")
data.particles.sel(sp=2).isel(t=0).phase_plot(
    ax=ax2, y_quantity=lambda df: df["uy"], cmap="inferno"
)
ax2.set(xlabel="$x$", ylabel="$u_y$", ylim=(-1.5, 1.5))

plt.show()
