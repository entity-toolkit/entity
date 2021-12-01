---
title: Coordinate systems
---

{{< hint info >}}
**On notations**\
To familiarize with the notations we use as well as find useful calculus formulae for curvilinear coordinate systems see the section [below]({{< relref "coordsys.md#notes" >}}).
{{< /hint >}}

At the moment the code supports several flat space (non-GR) coordinate systems: 1D/2D/3D _Cartesian/quasi-Cartesian_, and 2D _spherical_/_quasi-spherical_ (axisymmetric).

In the former case the orthogonal coordinates {{< katex >}}(\xi, \eta, \zeta){{</katex>}} are given by stretching the corresponding Cartesian coordinates.

{{< katex display >}}
\begin{aligned}
x&=f_x(\xi)\\
y&=f_y(\eta)\\
z&=f_z(\zeta)
\end{aligned}
{{< /katex >}}

The trivial case of Cartesian coordinates is handled separately in the code. In _quasi-Cartesian_ 3D case we define three separate functions, {{< katex >}}f_x{{</katex>}}, {{< katex >}}f_y{{</katex>}}, {{< katex >}}f_z{{</katex>}}, their inverses, {{< katex >}}f_x^{-1}{{</katex>}}, {{< katex >}}f_y^{-1}{{</katex>}}, {{< katex >}}f_z^{-1}{{</katex>}}, and three non-zero Jacobian coefficients {{< katex >}}\partial f_x/\partial\xi{{</katex>}}, {{< katex >}}\partial f_y/\partial\eta{{</katex>}}, {{< katex >}}\partial f_z/\partial\zeta{{</katex>}}.

## Note on curvilinear coordinates {#notes}

Suppose a transformation from Cartesian {{< katex >}}\bm{r}=(x,y,z){{< /katex >}} to an arbitrary orthogonal curvilinear coordinate space {{< katex >}}\bm{\rho}=(\xi,\eta,\zeta){{< /katex >}}, where {{< katex >}}x = f_x(\xi,\eta,\zeta){{< /katex >}}, {{< katex >}}y = f_y(\xi,\eta,\zeta){{< /katex >}} and {{< katex >}}z = f_z(\xi,\eta,\zeta){{< /katex >}}. We define

{{< katex display >}}
h_\xi = \left|\frac{\partial\bm{r}}{\partial\xi}\right|,~~~
h_\eta = \left|\frac{\partial\bm{r}}{\partial\eta}\right|,~~~
h_\zeta = \left|\frac{\partial\bm{r}}{\partial\zeta}\right|
{{< /katex >}}

and the unit vectors in the new coordinate system:

{{< katex display >}}
\hat{\bm{e}}_\xi=\frac{1}{h_\xi}\frac{\partial\bm{r}}{\partial\xi},~~~
\hat{\bm{e}}_\eta=\frac{1}{h_\eta}\frac{\partial\bm{r}}{\partial\eta},~~~
\hat{\bm{e}}_\zeta=\frac{1}{h_\zeta}\frac{\partial\bm{r}}{\partial\zeta}
{{< /katex >}}

Velocities (or any other vector) can be converted (from curvilinear to Cartesian) via:

{{< katex display >}}
\bm{v} = v_\xi \frac{\partial \bm{r}}{\partial \xi} + v_\eta \frac{\partial \bm{r}}{\partial \eta}+v_\zeta \frac{\partial \bm{r}}{\partial \zeta}
{{< /katex >}}

These {{< katex >}}\partial\bm{r}/\partial\xi{{< /katex >}} etc are just the columns of the Jacobi matrix:

{{< katex display >}}
J=\begin{bmatrix}
\frac{\partial\bm{r}}{\partial\xi}
&
\frac{\partial\bm{r}}{\partial\eta}
&
\frac{\partial\bm{r}}{\partial\zeta}
\end{bmatrix}
=\begin{bmatrix}
\frac{\partial x}{\partial\xi}
&
\frac{\partial x}{\partial\eta}
&
\frac{\partial x}{\partial\zeta}\\[0.5em]
\frac{\partial y}{\partial\xi}
&
\frac{\partial y}{\partial\eta}
&
\frac{\partial y}{\partial\zeta}\\[0.5em]
\frac{\partial z}{\partial\xi}
&
\frac{\partial z}{\partial\eta}
&
\frac{\partial z}{\partial\zeta}

\end{bmatrix}
{{< /katex >}}

Gradient of a scalar function {{< katex >}}f{{< /katex >}} in the new coordinate system:

{{< katex display >}}
\nabla f=
\frac{1}{h_\xi}\frac{\partial f}{\partial\xi}\hat{\bm{e}}_\xi+
\frac{1}{h_\eta}\frac{\partial f}{\partial\eta}\hat{\bm{e}}_\eta+
\frac{1}{h_\zeta}\frac{\partial f}{\partial\zeta}\hat{\bm{e}}_\zeta
{{< /katex >}}

Divergence of a vector function {{< katex >}}\bm{v}{{< /katex >}}:

{{< katex display >}}
\nabla\cdot\bm{v}=
\frac{1}{h_\xi h_\eta h_\zeta}\left[
\frac{\partial}{\partial\xi}\left(v_\xi h_\eta h_\zeta\right)+
\frac{\partial}{\partial\eta}\left(v_\eta h_\xi h_\zeta\right)+
\frac{\partial}{\partial\zeta}\left(v_\zeta h_\xi h_\eta\right)
\right]
{{< /katex >}}

Laplacian of a scalar function {{< katex >}}f{{< /katex >}}:

{{< katex display >}}
\Delta f=\frac{1}{h_\xi h_\eta h_\zeta}\left[
\frac{\partial}{\partial \xi}
\left(\frac{h_\eta h_\zeta}{h_\xi}\frac{\partial f}{\partial\xi}\right)+
\frac{\partial}{\partial \eta}
\left(\frac{h_\xi h_\zeta}{h_\eta}\frac{\partial f}{\partial\eta}\right)+
\frac{\partial}{\partial \zeta}
\left(\frac{h_\xi h_\eta}{h_\zeta}\frac{\partial f}{\partial\zeta}\right)

\right]
{{< /katex >}}

Curl of a vector function {{< katex >}}\bm{v}{{< /katex >}}:

{{< katex display >}}
\begin{split}
\nabla\times\bm{v}=&\frac{1}{h_\xi h_\eta h_\zeta}\cdot
\\[0.5em]
\cdot\Biggl[&h_\xi \hat{\bm{e}}_\xi
\left\{\frac{\partial}{\partial\eta}\left(h_\zeta v_\zeta\right)-\frac{\partial}{\partial\zeta}\left(h_\eta v_\eta\right)\right\}
\\[0.5em]
-&h_\eta \hat{\bm{e}}_\eta
\left\{\frac{\partial}{\partial\xi}\left(h_\zeta v_\zeta\right)-\frac{\partial}{\partial\zeta}\left(h_\xi v_\xi\right)\right\}
\\[0.5em]
&h_\zeta \hat{\bm{e}}_\zeta
\left\{\frac{\partial}{\partial\xi}\left(h_\eta v_\eta\right)-\frac{\partial}{\partial\eta}\left(h_\xi v_\xi\right)\right\}\Biggr]
\end{split}
{{< /katex >}}
