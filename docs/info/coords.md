---
hide:
  - footer
---

## Axisymmetric grid

Below for visualization purposes we demonstrate three different axisymmetric grids: a regular spherical grid $(r,\theta,\phi)$, an equal area grid $(x_1,x_2,\phi)$, where $x_1 = \log{(r)}$, and $x_2 = -\cos{\theta}$, and a "quasi-spherical" grid $(x_1,x_2,\phi)$, where $x_1 = \log{(r - r_0)}$, and $\theta = x_2 + 2h x_2 (\pi - 2 x_2) (\pi - x_2) / \pi^2$, where $r_0$ and $h$ are user-controlled parameters.

<div id="plot_ax_01" class="p5canvas"></div>

<!-- <script src="/how/coord_sys_ax.js" type="text/javascript"></script> -->