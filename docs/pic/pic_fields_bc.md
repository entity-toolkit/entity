## class ntt::FieldBC_rmax< D >

Algorithms for PIC field boundary conditions.  

---

```c++
template<Dimension D> FieldBC_rmax (const Meshblock< D, SimulationType::PIC > & mblock, const ProblemGenerator< D, SimulationType::PIC > & pgen, real_t r_absorb, real_t r_max)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 
- `pgen` Problem generator. 
- `rabsorb` Absorbing radius. 
- `rmax` Maximum radius. 

---

```c++
template<Dimension D> Inline void operator() (index_t, index_t)
```
2D implementation of the algorithm. 

**Parameters**
- `i1` index. 
- `i2` index. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

