---
hide:
  - footer
---

## Physical units

!!! note

    Here in the section we use $\bm{E}$, and $\bm{B}$ to denote the electric and magnetic fields in the orthonormal coordinate basis. For the purposes of this section flat space-time with trivial metric is assumed.

Most of the time the user will only need to interact with quantities in the so-called physical units. For example, when specifying `larmor0 = 42.0` in the input file, the value `42.0` is in physical units (we discuss what `larmor0` means shortly). The `extent` parameter in the input, similarly, is the extent of the simulation box in physical units.

In basic particle-in-cell algorithm we need to take care of three different equation sets: the Maxwell's equations on EM fields, equations of motion for the particles, and the current deposition. Let us write down all these in CGS (with all the vectors being defined in an orthonormal basis):

$$
\begin{aligned}
\text{Maxwell's equations}&
\begin{cases}
\frac{\partial\bm{B}}{c\partial t} = -\nabla\times\bm{E}\\\\
\frac{\partial\bm{E}}{c\partial t} = \nabla\times\bm{B} - \frac{4\pi}{c} \bm{J}
\end{cases}\\\\
\text{EoM for the {\it i}-th particle}&
\begin{cases}
\frac{d\left(\bm{\beta}_i\gamma_i\right)}{c dt} = \frac{q_i}{m_i c^2} \left(\bm{E} + \bm{\beta}_i\times\bm{B}\right)\\\\
\frac{d\bm{x}_i}{c dt} = \bm{\beta}_i
\end{cases}\\\\
\text{Deposited current}:&~
\bm{J}=\frac{1}{V}\sum\limits_{i\in V} q_i w_i \bm{\beta}_i c
\end{aligned}
$$

Here $q_i$, $m_i$, $w_i$, $\beta_i$ are the charges, the masses, the weights, and the dimensionless three-velocities of the macroparticles.

We now introduce the fiducial quantities which will help rescale all the equations from CGS to physical units. First of all, let us introduce a fiducial particle: one that has a charge $q_0>0$ and a mass $m_0$. And let us also define $B_0$ to be the fiducial magnetic field strength. If the fiducial particle moves in the uniform and constant magnetic field of strength $B_0$ in the perpendicular plane with a velocity $\bm{\beta}\gamma=1$, then it's Larmor radius is: $\rho_0=m_0 c^2/\left(q_0 B_0\right)$. 

In Gaussian units there is a fundamental freedom to pick $q_0/m_0 \equiv 1$, and $c\equiv 1$. Thus, $B_0 \equiv 1/\rho_0$.

If we now have plasma consisting of static particles (ions) of charge $-q_0$ and fiducial particles of charge $q_0$ with both species having a number density $n_0$ (fiducial number density), then this plasma will have a fundamental oscillation frequency, $\omega_0^2 = 4\pi n_0 q_0^2 / m_0$, and an equivalent lengthscale (fiducial skin depth): $d_0 \equiv 1/\omega_0$. Further we will see, that it is useful to define a fiducial current density as $J_0 \equiv  4\pi q_0 n_0$.

Because we are dealing with a discretized space, we also need to define a fiducial cell volume, $V_0$, fiducial number of particles per cell, $\texttt{PPC}_0$. Then the fiducial number density from above can be chosen to be $n_0 \equiv \texttt{PPC}_0 / V_0$.

| Symbol              | Description                   | Definition                                | In the code                         |
| ---                 | ---                           | ---                                       | ---                                 |
| $c$                 | speed of light                | $\equiv 1$                                | --                                  |
| $\texttt{PPC}_0$    | fiducial number of p.p.c.     | fundamental                               | `Simulation::params().ppc0()`       |
| $d_0$               | fiducial skin-depth           | fundamental                               | `Simulation::params().skindepth0()` |
| $\rho_0$            | fiducial Larmor radius        | fundamental                               | `Simulation::params().larmor0()`    |
| $V_0$               | fiducial volume size          | (see below)                               | `Simulation::params().V0()`         |
| $n_0$               | fiducial number density       | $\equiv\texttt{PPC}_0 / V_0$              | `Simulation::params().n0()`         |
| $4\pi q_0$          | fiducial particle charge      | $\equiv \left(n_0 d_0^2\right)^{-1}$      | `Simulation::params().q0()`         |
| $m_0$               | fiducial particle masses      | $\equiv q_0$                              |                                     |
| $\sigma_0$          | fiducial magnetization        | $\equiv \left(d_0/\rho_0\right)^2$        | `Simulation::params().sigma0()`     |
| $B_0$               | fiducial field strength       | $\equiv \rho_0^{-1}$                      | `Simulation::params().B0()`         |
| $J_0$               | fiducial current density      | $\equiv 4\pi q_0 n_0$                  |                                     |

We can then rewrite our equations in the "dimensionless" form, by renormalizing everything to fiducial units.

$$
\begin{aligned}
\text{with}~~~&\bm{e}\equiv \bm{E}/B_0,~~~\bm{b}\equiv \bm{B}/B_0,~~~\bm{j}\equiv \bm{J}/J_0,\\
&\tilde{q}_i \equiv q_i/q_0,~~~\tilde{m}_i \equiv m_i/m_0\\\\
\text{Maxwell's equations}&
\begin{cases}
\frac{\partial\bm{b}}{\partial t} = -\nabla\times\bm{e}\\\\
\frac{\partial\bm{e}}{\partial t} = \nabla\times\bm{b} - \frac{J_0}{B_0} \bm{j}
\end{cases}\\\\
\text{EoM for the {\it i}-th particle}&
\begin{cases}
\frac{d\left(\bm{\beta}_i\gamma_i\right)}{dt} = \frac{\tilde{q}_i}{\tilde{m}_i}B_0 \left(\bm{e} + \bm{\beta}_i\times\bm{b}\right)\\\\
\frac{d\bm{x}_i}{dt} = \bm{\beta}_i
\end{cases}\\\\
\text{Deposited current}:&~
\bm{j}=\frac{1}{V}\sum\limits_{i\in V} \tilde{q}_i w_i \bm{\beta}_i
\end{aligned}
$$

!!! note
  
    Fields and quantities contained in the data output are all normalized to their corresponding fiducial values. For a given physical setup, these normalized quantities are insensitive to either the resolution of the simulation, or the particle sampling ($\texttt{PPC}_0$). In other words, if you measure 10 plasma oscillations in the time interval $0 < t < 1$ with $\texttt{PPC}_0 = 10$, and resolution $128^3$, then you will measure the same number of oscillations with the same amplitude for $\texttt{PPC}_0 = 1000$, and resolution $512^3$. If you'd like to increase the "physical" density of your plasma, you should drop the value of $d_0$. Similarly, if you want to weaken the strength of your field, you should increase $\rho_0$.

### Fiducial volume and number density

To associate the number of simulation particles to the fiducial number density, $n_0$, we need to define the fiducial cell volume $V_0$. For Cartesian geometry, where all the cells have exactly the same size, it is defined simply as $V_0 \equiv (\Delta x)^D$, with $D$ being the dimension of the simulation. For spherical geometries, $V_0$ is defined as the volume of the first cell near the pole: 

$$
V_0 \equiv \begin{cases}
(\Delta x)^D&,~\text{for Cartesian}\\\\
\sqrt{\det{h}}\bigg\rvert_{r=\Delta r/2,~\theta=\Delta \theta/2}&,~\text{for spherical}
\end{cases}
$$

The interpretation of this is quite simple: if you initialize $\textrm{PPC}$ particles per each cell with weights $w_p$ moving in the $x_1$ direction with a velocity $\beta^{\hat{i}}$, their number density and the current density the impose would be:

$$
\frac{n}{n_0} = \frac{\textrm{PPC}}{\textrm{PPC}_0} \frac{V_0}{\sqrt{h}}w_p ,~~~\frac{J^{\hat{i}}}{J_0} = \frac{\textrm{PPC}}{\textrm{PPC}_0}\frac{V_0}{\sqrt{h}}w_p \beta^{\hat{i}}.
$$

Notice, that these quantities are independent of the resolution of the simulation, and the value of $\textrm{PPC}_0$. 

!!! important

    Knowing the exact value of $V_0$ is not necessary for the end-user. All the factors of $V_0$, $n_0$ etc must be canceled when working in dimensionless units. If you find otherwise, double check your calculations.

### Defining the physical units

The following parameters are parsed from the `input` file (under the `[units]` block) to define the fiducial physical quantities:

* `larmor0`: fiducial Larmor radius, $\rho_0$, of a particle with charge $q_0$ and mass $m_0$ moving in a uniform magnetic field of strength $B_0$ in the perpendicular plane with a velocity $\bm{\beta}\gamma=1$.
* `skindepth0`: fiducial skin depth for plasma, $d_0$, consisting of static particles (ions) of charge $-q_0$ and fiducial particles of charge $q_0$ with both species having a number density $n_0$ (fiducial number density).
* `ppc0`: fiducial number of particles per cell, $\textrm{PPC}_0$. If the domain of physical extent of size $1$ is filled with $\textrm{PPC}_0$ simulation particles per each cell with weights $1$, then the average number density of the simulation is $n/n_0 = 1$.

### Conversion to physical units

Equations that rely on pure electromagnetism (e.g., no quantum effects) can be directly expressed through our basis parameters. For that you need to simply make the following substitutions 

$$
n\to \tilde{n} n_0,~~~m\to \tilde{m}m_0,~~~q\to \tilde{q}q_0\\
\bm{B}\to \bm{b} B_0,~~~\bm{E}\to \bm{e} B_0,~~~\bm{J}\to \bm{j} J_0\\
ct\to t
$$

and then use equivalence relations in (1) to reduce all the fiducial unknowns to $\rho_0$, and $d_0$. If done correctly, no additional factors should remain (e.g., $B_0$, $c$, or $n_0$). Most of the quantities either plotted in the GUI or written in the output are in normalized physical units. For example, the output contains $\bm{b}$ and $\bm{e}$ field components (for non-GR simulations these are defined in the orthonormal basis).
    

!!! example

    For instance, we can compare the rest-mass energy density of plasma with number density $n_p$ consisting of particles with masses $m_p$ and charges $\pm q_p$ with the energy density of the magnetic field of strength $B$:

    $$
    \frac{U_B}{\rho_p c^2}\equiv \frac{B^2/(8\pi)}{n_p m_p c^2} = \frac{b}{2\tilde{n}_p\tilde{m}_p} \left(\frac{d_0}{\rho_0}\right)^2 = \frac{b}{2\tilde{n}_p\tilde{m}_p} \sigma_0,
    $$

    where $\tilde{n}_p = n_p / n_0$, $\tilde{m}_p = m_p / m_0$, and $b = B/B_0$.

    As another example, suppose we try to estimate Goldreich-Julian number density for a magnetosphere with a light cylinder defined as $R_{\rm LC}=c/\Omega$, a magnetic field of strength $B$ near the surface and pair plasma of charge $\pm q_p$. 

    $$
    \frac{n_{\rm GJ}}{n_0}\equiv \frac{\Omega B}{2\pi c |q_p|} = \frac{2 b}{|\tilde{q}_p|}\frac{d_0^2}{\rho_0 R_{\rm LC}}
    $$
    
    where $R_{\rm LC}$ is now measured in physical units (same units as for $\rho_0$ and $d_0$).