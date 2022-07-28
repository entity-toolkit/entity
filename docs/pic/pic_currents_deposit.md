## class ntt::Deposit< D >

Algorithm for the current deposition.  

---

```c++
template<Dimension D> Deposit (const Meshblock< D, SimulationType::PIC > & mblock, const Particles< D, SimulationType::PIC > & particles, const RealScatterFieldND< D, 3 > & scatter_cur, const real_t & coeff, const real_t & dt)
```
Constructor. 

**Parameters**
- `mblock` Meshblock. 
- `particles` Particles. 
- `scatter_cur` Scatter array of the currents. 
- `coeff` Coefficient to be multiplied by dE/dt = coeff * curl B. 
- `dt` Time step. 

---

```c++
template<Dimension D> Inline void depositCurrentsFromParticle (const vec_t< Dimension::THREE_D > & vp, const tuple_t< int, D > & Ip_f, const tuple_t< int, D > & Ip_i, const coord_t< D > & xp_f, const coord_t< D > & xp_i, const coord_t< D > & xp_r)
```
Deposit currents from a single particle. 

**Parameters**
- `vp` Particle 3-velocity. 
- `Ip_f` Final position of the particle (cell index). 
- `Ip_i` Initial position of the particle (cell index). 
- `xp_f` Final position. 
- `xp_i` Previous step position. 
- `xp_r` Intermediate point used in zig-zag deposit. 

---

```c++
template<Dimension D> Inline void getDepositInterval (index_t & p, vec_t< Dimension::THREE_D > & vp, tuple_t< int, D > & Ip_f, tuple_t< int, D > & Ip_i, coord_t< D > & xp_f, coord_t< D > & xp_i, coord_t< D > & xp_r)
```
Get particle position in \fCcoord_t form. 

**Parameters**
- `p` Index of particle. 
- `vp` Particle 3-velocity. 
- `Ip_f` Final position of the particle (cell index). 
- `Ip_i` Initial position of the particle (cell index). 
- `xp_f` Final position. 
- `xp_i` Previous step position. 
- `xp_r` Intermediate point used in zig-zag deposit. 

---

```c++
template<Dimension D> Inline void operator() (index_t p)
```
Iteration of the loop over particles. 

**Parameters**
- `p` index. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

