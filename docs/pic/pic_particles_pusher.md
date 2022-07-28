## class ntt::BorisBwd_t



---

## class ntt::BorisFwd_t



---

## class ntt::Photon_t



---

## class ntt::Pusher< D >

Algorithm for the Particle pusher.  

---

```c++
template<Dimension D> Pusher (const Meshblock< D, SimulationType::PIC > & mblock, const Particles< D, SimulationType::PIC > & particles, const real_t & coeff, const real_t & dt)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 
- `particles` Particles. 
- `coeff` Coefficient to be multiplied by dE/dt = coeff * curl B. 
- `dt` Time step. 

---

```c++
template<Dimension D> Inline void BorisUpdate (index_t & p, vec_t< Dimension::THREE_D > & e0, vec_t< Dimension::THREE_D > & b0)
```
Boris algorithm. 
Note:
Fields are modified inside the function and cannot be reused. 

**Parameters**
- `p` index of the particle. 
- `e` interpolated e-field vector of size 3 [modified]. 
- `b` interpolated b-field vector of size 3 [modified]. 

---

```c++
template<Dimension D> Inline void getParticleCoordinate (index_t &, coord_t< D > &)
```
Transform particle coordinate from code units i+di to \fCreal_t type. 

**Parameters**
- `p` index of the particle. 
- `coord` coordinate of the particle as a vector (of size D). 

---

```c++
template<Dimension D> Inline void interpolateFields (index_t &, vec_t< Dimension::THREE_D > &, vec_t< Dimension::THREE_D > &)
```
First order Yee mesh field interpolation to particle position. 

**Parameters**
- `p` index of the particle. 
- `e` interpolated e-field vector of size 3 [return]. 
- `b` interpolated b-field vector of size 3 [return]. 

---

```c++
template<Dimension D> Inline void operator() (const BorisFwd_t &, index_t p)
```
Pusher for the forward Boris algorithm. 

**Parameters**
- `p` index. 

---

```c++
template<Dimension D> Inline void operator() (const Photon_t &, index_t p)
```
Pusher for the photon pusher. 

**Parameters**
- `p` index. 

---

```c++
template<Dimension D> Inline void positionUpdate (index_t &, const vec_t< Dimension::THREE_D > &)
```
Update particle positions according to updated velocities. 

**Parameters**
- `p` index of the particle. 
- `v` particle 3-velocity. 

---

```c++
template<Dimension D> Inline void positionUpdate_x1 (index_t & p, const real_t & vx1)
```
Update each position component. 

**Parameters**
- `p` index of the particle. 
- `v` corresponding 3-velocity component. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

