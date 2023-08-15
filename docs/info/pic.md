---
hide:
  - footer
---

Here we demonstrate the full particle-in-cell (PIC) algorithm in the most general form for both flat (curvilinear) space-time and GR.

!!! note
    
    We use the monospace font to emphasize that all the arrays have the same exact values as the ones in the code (i.e., code units). $\texttt{b}_i$ is the same as `Meshblock.em(..., em::bx1)` etc.


=== "`ntt::PIC`"

    Full set of equations in flat space-time (with an arbitrary diagonal metric $h_{ij}=\textrm{diag}(h_1, h_2, h_3)$) in code units (and not in the order we actually integrate them):

    $$
    \begin{aligned}
    \text{field solvers}&
    \begin{cases}
    \frac{\Delta\texttt{b}_i}{\Delta \texttt{t}} = -\frac{1}{\sqrt{h}} \left[\Delta_j (h_{k}\texttt{e}_k) - \Delta_k (h_{j}\texttt{e}_j)\right]\\[1em]
    \frac{\Delta \texttt{e}_i}{\Delta \texttt{t}} = \frac{1}{\sqrt{h}} \left[\Delta_j (h_{k}\texttt{b}_k) - \Delta_k (h_{j}\texttt{b}_j)\right] - \frac{C_0}{\sqrt{h}}\texttt{j}_i
    \end{cases}\\[4em]
    \text{velocity update}&
    \begin{cases}
    \texttt{e},~\texttt{b} \xrightarrow[\text{interpolation}]{} \texttt{e}_p,~\texttt{b}_p\\[1em]
    \texttt{e}_p,~\texttt{b}_p \xrightarrow[\text{to global XYZ}]{\text{contravariant}} \hat{\texttt{e}}_p,~\hat{\texttt{b}}_p\\[1em]
    \gamma = \sqrt{1 + \hat{\texttt{u}}_i^2 + \hat{\texttt{u}}_j^2 + \hat{\texttt{u}}_k^2}\\[1em]
    \frac{\Delta \hat{\texttt{u}}_i}{\Delta t} = \frac{\tilde{q}_p}{\tilde{m}_p}B_0\left(
      \hat{\texttt{e}}_i 
      + \frac{\hat{\texttt{u}}_j}{\gamma} \hat{\texttt{b}}_k
      - \frac{\hat{\texttt{u}}_k}{\gamma} \hat{\texttt{b}}_j
    \right)
    \end{cases}\\[6em]
    \text{position update}&
    \begin{cases}
    \hat{\texttt{u}}_i \xrightarrow[\text{to contravariant}]{\text{global XYZ}} u^i\\[1em]
    \frac{\Delta \texttt{x}_i}{\Delta \texttt{t}} = \frac{u^i}{\gamma}
    \end{cases}\\[3em]
    \text{current deposition}&~~~
    \texttt{j}_i = \sum\limits_p \tilde{q}_p (\Delta \texttt{x}_i / \Delta \texttt{t})
    \end{aligned}
    $$

=== "`ntt::GRPIC`"

<!-- ## Particle pusher

## Charge-conservative current deposition

To ensure charge conservation for discrete set of particles we must then define their shape functions, $S_p(x^i-x_p^i)$ in the following way (see also the section about [the current deposition](../3p1/#current-deposition)):

$$
\begin{aligned}
\tilde{\rho} &= \sum\limits_p q_p S_p(x^i - x_p^i)\\
\bm{\mathcal{J}}^i &= \sum\limits_p q_p \frac{dx^i_p}{dt} S_p(x^i - x_p^i)
\end{aligned}
$$

where $q_p$ is the charge of the particle $p$ in its rest frame. $dx^i_p/dt$ is the particle three-velocity defined in agreement with [the equation of motion](../3p1/#equations-of-motion-for-particles): in practice it is $\left((x^i_p)^{\rm (new)} - (x^i_p)^{\rm (old)}\right) / \Delta t$. After the deposition, we can then recover the physical contravariant currents that go into the Maxwell's equations: $\bm{J}^i = \bm{\mathcal{J}}^i / \sqrt{h}$.

Full deposition loop can be expressed with the following pseudocode (actual array names and structures are different).

=== "`ntt::PIC`"

    ```go
    // e, j <-- 3D array of e-fields & currents (in either of the dimensions)
    // species <-- 1D array of species
    // species[s].prtls <-- 1D array of particles of species `s`
    // dt <-- timestep

    /* -------------------------------------------------------------------------- */
    /*                              0. reset currents                             */
    /* -------------------------------------------------------------------------- */

    j[:] = 0

    /* -------------------------------------------------------------------------- */
    /*                             1. deposit currents                            */
    /* -------------------------------------------------------------------------- */

    for s := range species {
      charge_s := species[s].charge
      for p := range prtl {
        // (1)
        // computing the Lorentz-factor
        gamma_p := sqrt(1 + p.u**2)
        // computing the contravariant 4-velocity
        uCntrv_p := convert_Cart2Cntrv(p.x, p.u)
        // computing x_old - x_new
        dx_p = p.x - uCntrv_p * dt / gamma_p

        // (2)
        j[...] += charge_s * dx_p / dt
      }
    }

    /* -------------------------------------------------------------------------- */
    /*                    2. recovering the physical currents &                   */
    /*                    adding as sources in the Ampere's law                   */
    /* -------------------------------------------------------------------------- */

    coeff := -rho0 / (n0 * d0**2) // (3)
    for i := range domain {
      e[i] += coeff * j[i] / (alpha(i) * sqrt_det_h(i))
    }
    ```

    1. :warning: Particle velocities are in global Cartesian basis, but the coordinates are in code units (i.e., contravariant $x^i$).
    2. :grey_exclamation: In reality we perform the zig-zag deposition routine, where we deposit into multiple cell edges depending on how the particle moves.
    3. :grey_exclamation: See the [code units](/how/units) chapter for more details.

=== "`ntt::GRPIC`"

    NA -->

## PIC algorithm loop

=== "`ntt::PIC`"

    For the non-GR case we use an explicit leapfrog integrator for both fields and the particles. All the fields, as well as particle coordinates/velocities are defined in the general curvilinear (orthonormal) coordinate system.

    *Initial configuration $t=t^{(n)}$:*
    <div id="plot0"></div>

    *Final configuration $t=t^{(n+1)}$:*
    <div id="plot6"></div>


    === "1. first EM substep"

        * 1.1. first Faraday half-step

        $$
        \frac{1}{c}\frac{\partial B_i}{\partial t} = -\frac{1}{h_1 h_2 h_3}h_i\varepsilon_{ijk}\partial^j\left(h^k E^k\right)
        $$

        <div id="plot1"></div>

        $$
        B^{(n-1/2)}\xrightarrow[\qquad E^{(n)}\qquad]{\Delta t/2} B^{(n)}
        $$

        ```c++
        // overwriting
        Simulation::Meshblock.em_fields(i, j, k, fld::bx1);
        Simulation::Meshblock.em_fields(i, j, k, fld::bx2);
        Simulation::Meshblock.em_fields(i, j, k, fld::bx3);
        ```

    === "2. particle push"

        * 2.1. velocity update (particle push)

        <div id="plot2_1"></div>

        $$
        u^{(n-1/2)}\xrightarrow[\qquad E(x^n),~B(x^n)\qquad]{\Delta t} u^{(n+1/2)}
        $$

        * 2.2. coordinate update (particle move)

        $$
        \frac{\mathrm{d} x_i}{\mathrm{d} t} = \frac{u_i}{\gamma}
        $$

        <div id="plot2_2"></div>

        $$
        x^{(n)}\xrightarrow[\qquad u^{(n+1/2)}\qquad]{\Delta t} x^{(n+1)}
        $$

    === "3. current deposition"

        * 3.1. displacement recovery

        <div id="plot3_1"></div>

        * 3.2. current deposition

        <div id="plot3_2"></div>

        ```c++
        // currents
        Simulation::Meshblock.cur(i, j, k, fld::jx1);
        Simulation::Meshblock.cur(i, j, k, fld::jx2);
        Simulation::Meshblock.cur(i, j, k, fld::jx3);
        ```

    === "4. second EM substep"

        * 4.1. second Faraday half-step

        <div id="plot4"></div>

        * 4.2. Ampere substep

        $$
        \frac{1}{c}\frac{\partial E_i}{\partial t} = \frac{1}{h_1 h_2 h_3}h_i\varepsilon_{ijk}\partial^j\left(h^k B^k\right) - \frac{4\pi}{c} J_i
        $$

        <div id="plot5"></div>

        $$
        E^{(n)}\xrightarrow[\qquad B^{(n+1/2)},~J^{(n+1/2)}\qquad]{\Delta t} E^{(n+1)}
        $$

        ```c++
        // overwriting
        Simulation::Meshblock.em_fields(i, j, k, fld::ex1);
        Simulation::Meshblock.em_fields(i, j, k, fld::ex2);
        Simulation::Meshblock.em_fields(i, j, k, fld::ex3);
        ```

=== "`ntt::GRPIC`"

    *Initial configuration $t=t^{(n)}$:*
    <div id="grplot0"></div>

    *Final configuration $t=t^{(n+1)}$:*
    <div id="grplot5"></div>

    === "1. first EM substep"

        * 1.1. intermediate interpolation

        $$
        \begin{aligned}
        D^{(n-1/2)} &= \frac{1}{2}\left(D^{(n-1)}+D^{(n)}\right),\\\\
        B^{(n-1)} &= \frac{1}{2}\left(B^{(n-3/2)}+B^{(n-1/2)}\right)
        \end{aligned}
        $$

        <div id="grplot1_1"></div>

        * 1.2. auxiliary field recovery

        $$
        E^{(n-1/2)} = \alpha D^{(n-1/2)} + \beta\times B^{(n-1/2)}
        $$

        <div id="grplot1_2"></div>

        * 1.3. auxiliary Faraday substep

        <div id="grplot1_3"></div>

        $$
        B^{(n-1)}\xrightarrow[\qquad E^{(n-1/2)}\qquad]{\Delta t} B^{(n)}
        $$

    === "2. particle push"

        <div id="grplot2_1"></div>

    === "3. current deposition"

        <div id="grplot3"></div>

    === "4. second EM substep"

        * 4.1. auxiliary field recovery

        $$
        E^{(n)} = \alpha D^{(n)} + \beta\times B^{(n)}
        $$

        $$
        H^{(n)} = \alpha B^{(n)} - \beta\times D^{(n)}
        $$

        <div id="grplot4_1"></div>

        * 4.2. Faraday substep

        <div id="grplot4_2"></div>

        $$
        B^{(n-1/2)}\xrightarrow[\qquad E^{(n)}\qquad]{\Delta t} B^{(n+1/2)}
        $$

        * 4.3. intermediate current interpolation

        $$
        J^{(n)} = \frac{1}{2}\left(J^{(n-1/2)}+J^{(n+1/2)}\right)
        $$

        <div id="grplot4_3"></div>

        * 4.4. auxiliary Ampere substep

        $$
        D^{(n-1/2)}\xrightarrow[\qquad H^{(n)},~J^{(n)}\qquad]{\Delta t} D^{(n+1/2)}
        $$

        <div id="grplot4_4"></div>

        * 4.5. auxiliary field recovery

        $$
        H^{(n+1/2)} = \alpha B^{(n+1/2)} - \beta\times D^{(n+1/2)}
        $$

        <div id="grplot4_5"></div>

        * 4.6. Ampere substep

        <div id="grplot4_6"></div>

        $$
        D^{(n)}\xrightarrow[\qquad H^{(n+1/2)},~J^{(n+1/2)}\qquad]{\Delta t} D^{(n+1)}
        $$

<div id="pic_scheme"></div>