## class ntt::TransformCurrentsSubstep< D >

Transform currents to the coordinate basis.  

---

```c++
template<Dimension D> TransformCurrentsSubstep (const Meshblock< D, SimulationType::PIC > & mblock)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 

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

