## class ntt::Metric< D >



---

```c++
template<Dimension D> auto findSmallestCell ()
```
Compute minimum effective cell size for a given metric (in physical units).

**Returns:**
- Minimum cell size of the grid [physical units]. 

---

```c++
template<Dimension D> Inline auto h_11 (const coord_t< D > &)
```
Compute metric component 11.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- h_11 (covariant, lower index) metric component. 

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
Compute the area at the pole (used in axisymmetric solvers).

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- Area at the pole. 

---

```c++
template<Dimension D> Inline auto sqrt_det_h (const coord_t< D > & x)
```
Compute the square root of the determinant of h-matrix.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- sqrt(det(h_ij)). 

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
Note:
Since kokkos disallows virtual inheritance, we have to include vector transformations for a diagonal metric here (and not in the base class).Coordinate conversion from code units to Cartesian physical units.

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

