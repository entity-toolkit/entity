---
hide:
  - footer
---

## Axisymmetric grid

Below for visualization purposes we demonstrate three different axisymmetric grids: a regular spherical grid $(r,\theta,\phi)$, an equal area grid $(x_1,x_2,\phi)$, where $x_1 = \log{(r)}$, and $x_2 = -\cos{\theta}$, and a "quasi-spherical" grid $(x_1,x_2,\phi)$, where $x_1 = \log{(r - r_0)}$, and $\theta = x_2 + 2h x_2 (\pi - 2 x_2) (\pi - x_2) / \pi^2$, where $r_0$ and $h$ are user-controlled parameters.

??? note "Code insight"

    In the `entity` we implement the regular spherical and the quasi-spherical grids. The choice of either is specified in the input file:

    ```toml
    [domain]
    coord_sys       = "spherical"   # or "qspherical"
    qsph_r0         = 0.3           # r_0 parameter (ignored when "spherical")
    qsph_h          = 0.9           # h parameter (ignored when "spherical")
    ```

    Notice, that one has to also configure the code with the `-curv` flag to use either of grids.

<div id="plot_ax_01" class="p5canvas"></div>

<!-- <script src="/how/coord_sys_ax.js" type="text/javascript"></script> -->