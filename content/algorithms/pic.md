---
title: Particle-in-cell
---

{{< toc >}}

{{<d3header>}}
{{<mathjaxheader>}}

Here we demonstrate the full particle-in-cell (PIC) algorithm in the most general form for both flat (curvilinear) space-time and GR.

## Non-GR

{{< hint info >}}
**Substeps are clickable!**
{{< /hint >}}

For the non-GR case we use an explicit leapfrog integrator for both fields and the particles. All the fields, as well as particle coordinates/velocities are defined in the general curvilinear (orthonormal) coordinate system.


<div id="plot0"></div>

```c++
// fields
int i, j, k;
Simulation::Meshblock.em_fields(i, j, k, fld::ex1);
Simulation::Meshblock.em_fields(i, j, k, fld::ex2);
Simulation::Meshblock.em_fields(i, j, k, fld::ex3);

Simulation::Meshblock.em_fields(i, j, k, fld::bx1);
Simulation::Meshblock.em_fields(i, j, k, fld::bx2);
Simulation::Meshblock.em_fields(i, j, k, fld::bx3);

// Note: even though we employ (i, j, k) indexing ...
// ... in the code the field components are staggered ...
// ... not only in time but also spatially.

// particles
int species_id, prtl_id;
Simulation::Meshblock.particles[species_id].m_x1(prtl_id);
Simulation::Meshblock.particles[species_id].m_x2(prtl_id);
Simulation::Meshblock.particles[species_id].m_x3(prtl_id);

Simulation::Meshblock.particles[species_id].m_ux1(prtl_id);
Simulation::Meshblock.particles[species_id].m_ux2(prtl_id);
Simulation::Meshblock.particles[species_id].m_ux3(prtl_id);
// Note: particle velocities are staggered in time ...
// ... w.r.t. the coordinates
```

{{< expand "1. first EM substep" >}}

#### 1.1. first Faraday half-step

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

{{< /expand >}}

{{< expand "2. particle push" >}}

#### 2.1. velocity update (particle push)

<div id="plot2_1"></div>

$$
u^{(n-1/2)}\xrightarrow[\qquad E(x^n),~B(x^n)\qquad]{\Delta t} u^{(n+1/2)}
$$

#### 2.2. coordinate update (particle move)

$$
\frac{\mathrm{d} x_i}{\mathrm{d} t} = \frac{u_i}{\gamma}
$$

<div id="plot2_2"></div>

$$
x^{(n)}\xrightarrow[\qquad u^{(n+1/2)}\qquad]{\Delta t} x^{(n+1)}
$$

{{< /expand >}}

<!-- {{< expand "2. particle push" >}}

#### 2.1. field interpolation

<div id="plot2_1"></div>

#### 2.2. field/velocity transformations (to Cartesian, denoted by "${}^c$")

$$
u^c_i = u_j\mathcal{J}^{ij},~~~ E^c_i = E_j\mathcal{J}^{ij}, ~~~ B^c_i = B_j\mathcal{J}^{ij}
$$

<div id="plot2_2"></div>

#### 2.3. velocity update (particle push)

$$
F_{\rm L}^c = q\left(\boldsymbol{E}^c + \frac{1}{c\gamma^c}\boldsymbol{u}^c\times\boldsymbol{B}^c\right)
$$

<div id="plot2_3"></div>

$$
u^{c~(n-1/2)}\xrightarrow[\qquad E^c(x^n),~B^c(x^n)\qquad]{\Delta t} u^{c~(n+1/2)}
$$

#### 2.4. velocity back-transformation (to curvilinear)

$$
???
$$

<div id="plot2_4"></div>

#### 2.5. coordinate update (particle move)

$$
\frac{\mathrm{d} x_i}{\mathrm{d} t} = \frac{u_i}{\gamma}
$$

<div id="plot2_5"></div>

$$
x^{(n)}\xrightarrow[\qquad u^{(n+1/2)}\qquad]{\Delta t} x^{(n+1)}
$$

{{< /expand >}} -->

{{< expand "3. current deposition" >}}

#### 3.1. displacement recovery

<div id="plot3_1"></div>

#### 3.2. current deposition

<div id="plot3_2"></div>

```c++
// currents
Simulation::Meshblock.j_fields(i, j, k, fld::jx1);
Simulation::Meshblock.j_fields(i, j, k, fld::jx2);
Simulation::Meshblock.j_fields(i, j, k, fld::jx3);
```

<!-- #### 3.3. current filtering

<div id="plot3_3"></div>

#### 3.4. current transformation

$$
J_i = \frac{1}{h_j h_k} j_i
$$

<div id="plot3_4"></div> -->

{{< /expand >}}

{{< expand "4. second EM substep" >}}

#### 4.1. second Faraday half-step

<div id="plot4"></div>

#### 4.2. Ampere substep

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

{{< /expand >}}

<div id="plot6"></div>

{{<jsfile "nongr.js">}}

---
<!--
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 -->

## GR

<div id="grplot0"></div>

{{< expand "1. first EM substep" >}}

#### 1.1. intermediate interpolation

$$
\begin{aligned}
D^{(n-1/2)} &= \frac{1}{2}\left(D^{(n-1)}+D^{(n)}\right),\\\\
B^{(n-1)} &= \frac{1}{2}\left(B^{(n-3/2)}+B^{(n-1/2)}\right)
\end{aligned}
$$

<div id="grplot1_1"></div>

#### 1.2. auxiliary field recovery

$$
E^{(n-1/2)} = \alpha D^{(n-1/2)} + \beta\times B^{(n-1/2)}
$$

<div id="grplot1_2"></div>

#### 1.3. auxiliary Faraday substep

<div id="grplot1_3"></div>

$$
B^{(n-1)}\xrightarrow[\qquad E^{(n-1/2)}\qquad]{\Delta t} B^{(n)}
$$

{{< /expand >}}

{{< expand "2. particle push" >}}

#### ??? intermediate steps skipped

<div id="grplot2_1"></div>

{{< /expand >}}

{{< expand "3. current deposition" >}}

<div id="grplot3"></div>

{{< /expand >}}

{{< expand "4. second EM substep" >}}

#### 4.1. auxiliary field recovery

$$
E^{(n)} = \alpha D^{(n)} + \beta\times B^{(n)}
$$

$$
H^{(n)} = \alpha B^{(n)} - \beta\times D^{(n)}
$$

<div id="grplot4_1"></div>

#### 4.2. Faraday substep

<div id="grplot4_2"></div>

$$
B^{(n-1/2)}\xrightarrow[\qquad E^{(n)}\qquad]{\Delta t} B^{(n+1/2)}
$$

#### 4.3. intermediate current interpolation

$$
J^{(n)} = \frac{1}{2}\left(J^{(n-1)}+J^{(n+1/2)}\right)
$$

<div id="grplot4_3"></div>

#### 4.4. auxiliary Ampere substep

$$
D^{(n-1/2)}\xrightarrow[\qquad H^{(n)},~J^{(n)}\qquad]{\Delta t} D^{(n+1/2)}
$$

<div id="grplot4_4"></div>

#### 4.5. auxiliary field recovery

$$
H^{(n+1/2)} = \alpha B^{(n+1/2)} - \beta\times D^{(n+1/2)}
$$

<div id="grplot4_5"></div>

#### 4.6. Ampere substep

<div id="grplot4_6"></div>

$$
D^{(n)}\xrightarrow[\qquad H^{(n+1/2)},~J^{(n+1/2)}\qquad]{\Delta t} D^{(n+1)}
$$

{{< /expand >}}

<div id="grplot5"></div>

{{<jsfile "gr.js">}}
