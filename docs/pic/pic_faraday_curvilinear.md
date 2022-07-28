## class ntt::FaradayCurvilinear< D >

Algorithm for the Faraday's law: \fCdB/dt = -curl E in Curvilinear space (diagonal metric).  

---

```c++
template<Dimension D> FaradayCurvilinear (const Meshblock< D, SimulationType::PIC > & mblock, const real_t & coeff)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 
- `coeff` Coefficient to be multiplied by dB/dt = coeff * -curl E. 

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

