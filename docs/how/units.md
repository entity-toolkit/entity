---
hide:
  - footer
---

In this code it is useful to distinguish two different unit and coordinate systems: the **physical units** (not to be confused with the real-world units in, e.g., CGS) and the **code units**.

## Physical units

!!! note

    Here in the section we use $\bm{E}$, and $\bm{B}$ to denote the electric and magnetic fields in the orthonormal coordinate basis. For the purposes of this section flat space-time with trivial metric is assumed.

Most of the time the user will only need to interact with quantities in physical units. For example, when specifying `larmor0 = 42.0` in the input file, the value `42.0` is in physical units (we discuss what `larmor0` means shortly). The `extent` parameter in the input, similarly, is the extent of the simulation box in physical units.

In basic particle-in-cell algorithm we need to take care of three different equation sets: the Maxwell's equations on EM fields, equations of motion for the particles, and the current deposition. Let us write down all these in CGS:

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
\text{Deposited current density in volume {\it V}}:&~
\bm{J}=\frac{1}{V}\sum\limits_{i\in V} q_i \bm{\beta}_i c
\end{aligned}
$$

We now introduce the fiducial quantities which will help rescale all the equations from CGS to physical units. First of all, let us introduce a fiducial particle: one that has a charge $q_0>0$ and a mass $m_0$. And let us also define $B_0$ to be the fiducial magnetic field strength. If the fiducial particle moves in the uniform and constant magnetic field of strength $B_0$ in the perpendicular plane with a velocity $\bm{\beta}\gamma=1$, then it's Larmor radius is: $\rho_0=m_0 c^2/\left(q_0 B_0\right)$. If we now have plasma consisting of static particles (ions) of charge $-q_0$ and fiducial particles of charge $q_0$ with both species having a number density $n_0$ (fiducial number density), then this plasma will have a fundamental oscillation frequency, $\omega_0^2 = 4\pi n_0 q_0^2 / m_0$, and an equivalent lengthscale (fiducial skin depth): $d_0 = c/\omega_0$. Further we will see, that it is useful to define a fiducial current as $J_0 = q_0 c$.

The easiest way to rewrite any equation in our physical units is to use the following equivalence:

\begin{equation}
\begin{aligned}
4\pi q_0 &\equiv\frac{\rho_0 B_0}{n_0 d_0^2}\\
\frac{q_0}{m_0} &\equiv \frac{c^2}{\rho_0}\frac{1}{B_0}\\
J_0&\equiv q_0 c
\end{aligned}
\end{equation}

Without the loss of generality we can also define time in such a way, that $c = 1$ in our physical units. We can then rewrite our equations in the physical units:

$$
\begin{aligned}
\text{Maxwell's equations}&
\begin{cases}
\frac{\partial\bm{b}}{\partial t} = -\nabla\times\bm{e}\\\\
\frac{\partial\bm{e}}{\partial t} = \nabla\times\bm{b} - \frac{\rho_0}{n_0 d_0^2} \bm{j}
\end{cases}\\\\
\text{EoM for the {\it i}-th particle}&
\begin{cases}
\frac{d\left(\bm{\beta}_i\gamma_i\right)}{dt} = \frac{\tilde{q}_i}{\tilde{m}_i}\frac{1}{\rho_0} \left(\bm{e} + \bm{\beta}_i\times\bm{b}\right)\\\\
\frac{d\bm{x}_i}{dt} = \bm{\beta}_i
\end{cases}\\\\
\text{Deposited current density in volume {\it V}}:&~
\bm{j}=\frac{1}{V}\sum\limits_{i\in V} \tilde{q}_i \bm{\beta}_i
\end{aligned}
$$

where we employ $\bm{e}\equiv \bm{E}/B_0$, $\bm{b}\equiv \bm{B}/B_0$, $\bm{j}\equiv \bm{J}/J_0$, $\tilde{q}_i \equiv q_i/q_0$, and $\tilde{m}_i \equiv m_i/m_0$ (notice that $B_0$, $q_0$, or $m_0$ do not enter any of the equations explicitly). Just like in CGS, in our physical units we fix *four* quantities to fully constraint the unit system: $\rho_0$, $d_0$, $n_0$, and $c = 1$.

!!! important
  
    All the dimensional scales in the input (such as the extent of the domain, the fiducial plasma skin depth, etc) are specified in **physical units**.


### Conversion to physical units

Equations that rely on pure electromagnetism (e.g., no quantum effects) can be directly expressed through our basis parameters. For that you need to simply make the following substitutions 

$$
n\to \tilde{n} n_0,~~~m\to \tilde{m}m_0,~~~q\to \tilde{q}q_0\\
\bm{B}\to \bm{b} B_0,~~~\bm{E}\to \bm{e} B_0,~~~\bm{J}\to \bm{j} J_0\\
ct\to t
$$

and then use equivalence relations in (1) to reduce all the fiducial unknowns to $\rho_0$, $d_0$, and $n_0$. If done correctly, no additional factors should remain (e.g., $B_0$ or $c$).
    

!!! example

    For instance, we can compare the rest-mass energy density of plasma with number density $n_p$ consisting of particles with masses $m_p$ and charges $\pm q_p$ with the energy density of the magnetic field of strength $B$:

    $$
    \frac{U_B}{\rho_p c^2}\equiv \frac{B^2/(8\pi)}{n_p m_p c^2} = \frac{b}{2\tilde{n}_p\tilde{m}_p} \left(\frac{d_0}{\rho_0}\right)^2,
    $$

    where $\tilde{n}_p = n_p / n_0$, $\tilde{m}_p = m_p / m_0$, and $b = B/B_0$.

    As another example, suppose we try to estimate Goldreich-Julian number density for a magnetosphere with a light cylinder defined as $R_{\rm LC}=c/\Omega$, a magnetic field of strength $B$ and pair plasma of charge $\pm q_p$. 

    $$
    n_{\rm GJ}\equiv \frac{\Omega B}{2\pi c |q_p|} = \frac{2 b}{|\tilde{q}_p|}\frac{n_0 d_0^2}{\rho_0 R_{\rm LC}}
    $$
    
    where $R_{\rm LC}$ is now measured in physical units.

## Code units

While the user mostly interacts with the code in terms of physical units, the code itself operates in terms of code units. The only difference between the code units and the physical units is that in code units coordinates are rescaled to be in the range `[0, nxi)`, where `nxi` is the number of grid points in the $i$-th direction.