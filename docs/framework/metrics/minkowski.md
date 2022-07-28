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
template<Dimension D> Inline auto h_22 (const coord_t< D > &)
```
Compute metric component 22.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- h_22 (covariant, lower index) metric component. 

---

```c++
template<Dimension D> Inline auto h_33 (const coord_t< D > &)
```
Compute metric component 33.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- h_33 (covariant, lower index) metric component. 

---

```c++
template<Dimension D> Inline auto sqrt_det_h (const coord_t< D > &)
```
Compute the square root of the determinant of h-matrix.

**Parameters**
- `x` coordinate array in code units (size of the array is D). 

**Returns:**
- sqrt(det(h_ij)). 

---

```c++
template<Dimension D> Inline void v_Cart2Cntrv (const coord_t< D > & xi, const vec_t< Dimension::THREE_D > & vi_cart, vec_t< Dimension::THREE_D > & vi_cntrv)
```
Vector conversion from global Cartesian to contravariant basis.

**Parameters**
- `xi` coordinate array in code units (size of the array is D). 
- `vi_cart` vector in global Cartesian basis (size of the array is 3). 
- `vi_cntrv` vector in contravariant basis (size of the array is 3). 

---

```c++
template<Dimension D> Inline void v_Cart2Cov (const coord_t< D > & xi, const vec_t< Dimension::THREE_D > & vi_cart, vec_t< Dimension::THREE_D > & vi_cov)
```
Vector conversion from global Cartesian to covariant basis.

**Parameters**
- `xi` coordinate array in code units (size of the array is D). 
- `vi_cart` vector in global Cartesian basis (size of the array is 3). 
- `vi_cov` vector in covariant basis (size of the array is 3). 

---

```c++
template<Dimension D> Inline void v_Cntrv2Cart (const coord_t< D > & xi, const vec_t< Dimension::THREE_D > & vi_cntrv, vec_t< Dimension::THREE_D > & vi_cart)
```
Vector conversion from contravariant to global Cartesian basis.

**Parameters**
- `xi` coordinate array in code units (size of the array is D). 
- `vi_cntrv` vector in contravariant basis (size of the array is 3). 
- `vi_cart` vector in global Cartesian basis (size of the array is 3). 

---

```c++
template<Dimension D> Inline void v_Cov2Cart (const coord_t< D > & xi, const vec_t< Dimension::THREE_D > & vi_cov, vec_t< Dimension::THREE_D > & vi_cart)
```
Vector conversion from covariant to global Cartesian basis.

**Parameters**
- `xi` coordinate array in code units (size of the array is D). 
- `vi_cov` vector in covariant basis (size of the array is 3). 
- `vi_cart` vector in global Cartesian basis (size of the array is 3). 

---

```c++
template<Dimension D> Inline void x_Cart2Code (const coord_t< D > &, coord_t< D > &)
```
Coordinate conversion from Cartesian physical units to code units.

**Parameters**
- `x` coordinate array in Cartesian coordinates in physical units (size of the array is D). 
- `xi` coordinate array in code units (size of the array is D). 

---

```c++
template<Dimension D> Inline void x_Code2Cart (const coord_t< D > &, coord_t< D > &)
```
Note:
Since kokkos disallows virtual inheritance, we have to include vector transformations for a diagonal metric here (and not in the base class).Coordinate conversion from code units to Cartesian physical units.

**Parameters**
- `xi` coordinate array in code units (size of the array is D). 
- `x` coordinate array in Cartesian physical units (size of the array is D). 

---

```c++
template<Dimension D> Inline void x_Code2Sph (const coord_t< D > &, coord_t< D > &)
```
Coordinate conversion from code units to Spherical physical units.

**Parameters**
- `xi` coordinate array in code units (size of the array is D). 
- `x` coordinate array in Spherical coordinates in physical units (size of the array is D). 

---

```c++
template<Dimension D> Inline void x_Sph2Code (const coord_t< D > &, coord_t< D > &)
```
Coordinate conversion from Spherical physical units to code units.

**Parameters**
- `xi` coordinate array in Spherical coordinates in physical units (size of the array is D). 
- `x` coordinate array in code units (size of the array is D). 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

