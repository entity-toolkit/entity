## class ntt::Metric< D >



---

```c++
template<Dimension D> Inline auto alpha (const coord_t< D > & x)
```
Compute lapse function.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- alpha. 

---

```c++
template<Dimension D> Inline auto beta1u (const coord_t< D > & x)
```
Compute radial component of shift vector.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- beta^1 (contravariant). 

---

```c++
template<Dimension D> auto findSmallestCell ()
```
Note:
Since kokkos disallows virtual inheritance, we have to include vector transformations for a non-diagonal metric here (and not in the base class).Compute minimum effective cell size for a given metric (in physical units). 

**Returns:**
- Minimum cell size of the grid [physical units]. 

---

```c++
template<Dimension D> Inline auto h_11 (const coord_t< D > & x)
```
Compute metric component 11.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- h_11 (covariant, lower index) metric component. 

---

```c++
template<Dimension D> Inline auto h_13 (const coord_t< D > & x)
```
Compute metric component 13.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- h_13 (covariant, lower index) metric component. 

---

```c++
template<Dimension D> Inline auto h_22 (const coord_t< D > & x)
```
Compute metric component 22.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- h_22 (covariant, lower index) metric component. 

---

```c++
template<Dimension D> Inline auto h_33 (const coord_t< D > & x)
```
Compute metric component 33.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- h_33 (covariant, lower index) metric component. 

---

```c++
template<Dimension D> Inline auto polar_area (const coord_t< D > & x)
```
Compute the area at the pole (used in axisymmetric solvers). Approximate solution for the polar area.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- Area at the pole. 

---

```c++
template<Dimension D> Inline auto sqrt_det_h_tilde (const coord_t< D > & x)
```
Compute the square root of the determinant of h-matrix divided by sin(theta).

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- sqrt(det(h))/sin(theta). 

---

```c++
template<Dimension D> Inline void v_Cntr2SphCntrv (const coord_t< D > &, const vec_t< Dimension::THREE_D > & vi_cntrv, vec_t< Dimension::THREE_D > & vsph_cntrv)
```
Vector conversion from contravariant to spherical contravariant.

**Parameters**
- `xi` coordinate array in code units (size of the array is D). 
- `vi_cntrv` vector in contravariant basis (size of the array is 3). 
- `vsph_cntrv` vector in spherical contravariant basis (size of the array is 3). 

---

```c++
template<Dimension D> Inline void x_Cart2Code (const coord_t< D > & x, coord_t< D > & xi)
```
Coordinate conversion from Cartesian physical units to code units.

**Parameters**
- `x` coordinate array in Cartesian coordinates in physical units (size of the array is D). 
- `xi` coordinate array in code units (size of the array is D). 

---

```c++
template<Dimension D> Inline void x_Code2Cart (const coord_t< D > & xi, coord_t< D > & x)
```
Coordinate conversion from code units to Cartesian physical units.

**Parameters**
- `xi` coordinate array in code units (size of the array is D). 
- `x` coordinate array in Cartesian physical units (size of the array is D). 

---

```c++
template<Dimension D> Inline void x_Code2Sph (const coord_t< D > & xi, coord_t< D > & x)
```
Coordinate conversion from code units to Spherical physical units.

**Parameters**
- `xi` coordinate array in code units (size of the array is D). 
- `x` coordinate array in Spherical coordinates in physical units (size of the array is D). 

---

```c++
template<Dimension D> Inline void x_Sph2Code (const coord_t< D > & x, coord_t< D > & xi)
```
Coordinate conversion from Spherical physical units to code units.

**Parameters**
- `x` coordinate array in Spherical coordinates in physical units (size of the array is D). 
- `xi` coordinate array in code units (size of the array is D). 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

