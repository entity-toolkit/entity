Here we demonstrate the full particle-in-cell (PIC) algorithm in the most general form for both flat (curvilinear) space-time and GR.

## Non-GR

For the non-GR case we use an explicit leapfrog integrator for both fields and the particles. All the fields, as well as particle coordinates/velocities are defined in the general curvilinear (orthonormal) coordinate system.

```cpp
// fields (1)
int i, j, k;
Simulation::Meshblock.em_fields(i, j, k, fld::ex1);
Simulation::Meshblock.em_fields(i, j, k, fld::ex2);
Simulation::Meshblock.em_fields(i, j, k, fld::ex3);

Simulation::Meshblock.em_fields(i, j, k, fld::bx1);
Simulation::Meshblock.em_fields(i, j, k, fld::bx2);
Simulation::Meshblock.em_fields(i, j, k, fld::bx3);

// particles (2)
int species_id, prtl_id;
Simulation::Meshblock.particles[species_id].m_i1(prtl_id);
Simulation::Meshblock.particles[species_id].m_i2(prtl_id);
Simulation::Meshblock.particles[species_id].m_i3(prtl_id);
Simulation::Meshblock.particles[species_id].m_dx1(prtl_id);
Simulation::Meshblock.particles[species_id].m_dx2(prtl_id);
Simulation::Meshblock.particles[species_id].m_dx3(prtl_id);

Simulation::Meshblock.particles[species_id].m_ux1(prtl_id);
Simulation::Meshblock.particles[species_id].m_ux2(prtl_id);
Simulation::Meshblock.particles[species_id].m_ux3(prtl_id);
```

1. :grey_exclamation: Even though we employ `(i, j, k)` indexing in the code the field components are staggered not only in time but also spatially.
2. :grey_exclamation: Particle velocities are staggered in time w.r.t. the coordinates.


*Initial configuration:*
<div id="plot0"></div>


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

*Final configuration:*
<div id="plot6"></div>

---

## GR

*Initial configuration:*
<div id="grplot0"></div>

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

*Final configuration:*
<div id="grplot5"></div>

<div id="pic_scheme"></div>