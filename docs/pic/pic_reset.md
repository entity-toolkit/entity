## class ntt::ResetCurrents< D >

Reset all the currents to zero.  

---

```c++
template<Dimension D> ResetCurrents (const Meshblock< D, SimulationType::PIC > & mblock)
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

## class ntt::ResetFields< D >

Reset all the fields to zero.  

---

```c++
template<Dimension D> ResetFields (const Meshblock< D, SimulationType::PIC > & mblock)
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

## class ntt::ResetParticles< D >

Reset all particles of a particular species.  

---

```c++
template<Dimension D> ResetParticles (const Meshblock< D, SimulationType::PIC > & mblock, const Particles< D, SimulationType::PIC > & particles)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 
- `particles` Particles. 

---

```c++
template<Dimension D> Inline void operator() (index_t p)
```
Loop iteration. 

**Parameters**
- `p` index 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

