## class ntt::AddCurrentsSubstep< D >

Add the currents to the E field.  

---

```c++
template<Dimension D> AddCurrentsSubstep (const Meshblock< D, SimulationType::PIC > & mblock)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 

---

```c++
template<Dimension D> Inline void operator() (index_t i1)
```
1D version of the add current. 

**Parameters**
- `i1` index. 

---

```c++
template<Dimension D> Inline void operator() (index_t i1, index_t i2)
```
2D version of the add current. 

**Parameters**
- `i1` index. 
- `i2` index. 

---

```c++
template<Dimension D> Inline void operator() (index_t i1, index_t i2, index_t i3)
```
3D version of the add current. 

**Parameters**
- `i1` index. 
- `i2` index. 
- `i3` index. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

