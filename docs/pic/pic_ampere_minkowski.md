## class ntt::AmpereMinkowski< D >

Algorithm for the Ampere's law: \fCdE/dt = curl B in Minkowski space.  

---

```c++
template<Dimension D> AmpereMinkowski (const Meshblock< D, SimulationType::PIC > & mblock, const real_t & coeff)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 
- `coeff` Coefficient to be multiplied by dE/dt = coeff * curl B. 

---

```c++
template<Dimension D> Inline void operator() (index_t)
```
1D implementation of the algorithm. 

**Parameters**
- `i1` index. 

---

```c++
template<Dimension D> Inline void operator() (index_t, index_t)
```
2D implementation of the algorithm. 

**Parameters**
- `i1` index. 
- `i2` index. 

---

```c++
template<Dimension D> Inline void operator() (index_t, index_t, index_t)
```
3D implementation of the algorithm. 

**Parameters**
- `i1` index. 
- `i2` index. 
- `i3` index. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

