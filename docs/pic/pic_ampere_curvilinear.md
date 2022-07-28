## class ntt::AmpereCurvilinear< D >

Algorithm for the Ampere's law: \fCdE/dt = curl B in curvilinear space.  

---

```c++
template<Dimension D> AmpereCurvilinear (const Meshblock< D, SimulationType::PIC > & mblock, const real_t & coeff)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 
- `coeff` Coefficient to be multiplied by dE/dt = coeff * curl B. 

---

```c++
template<Dimension D> Inline void operator() (index_t i1, index_t i2)
```
2D version of the algorithm. 

**Parameters**
- `i1` index. 
- `i2` index. 

---

```c++
template<Dimension D> Inline void operator() (index_t i1, index_t i2, index_t i3)
```
3D version of the algorithm. 

**Parameters**
- `i1` index. 
- `i2` index. 
- `i3` index. 

---

## class ntt::AmpereCurvilinearPoles< D >

Algorithm for the Ampere's law: \fCdE/dt = curl B in curvilinear space near the polar axes (integral form).  

---

```c++
template<Dimension D> AmpereCurvilinearPoles (const Meshblock< D, SimulationType::PIC > & mblock, const real_t & coeff)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 
- `coeff` Coefficient to be multiplied by dE/dt = coeff * curl B. 

---

```c++
template<Dimension D> Inline void operator() (index_t i)
```
Implementation of the algorithm. 

**Parameters**
- `i` radial index. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

