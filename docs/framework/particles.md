## class ntt::ParticleSpecies

Container for the information about the particle species.  

---

```c++
ParticleSpecies (std::string label, const float & m, const float & ch, const std::size_t & maxnpart, const ParticlePusher & pusher)
```
Constructor for the particle species container. 

**Parameters**
- `label` species label. 
- `m` species mass. 
- `ch` species charge. 
- `maxnpart` max number of allocated particles. 
- `pusher` pusher assigned for the species. 

---

```c++
ParticleSpecies (std::string, const float &, const float &, const std::size_t &)
```
Constructor for the particle species container which deduces the pusher itself. This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts. 

**Parameters**
- `label` species label. 
- `m` species mass. 
- `ch` species charge. 
- `maxnpart` max number of allocated particles. 

---

```c++
ParticleSpecies (const ParticleSpecies &)
```
Copy constructor for the particle species container. This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts. 

**Parameters**
- `spec` 

---

## class ntt::Particles< D, S >

Container class to carry particle information for a specific species.  

---

```c++
template<Dimension D, SimulationType S> Particles (const std::string &, const float &, const float &, const std::size_t &)
```
Constructor for the particle container. 

**Parameters**
- `label` species label. 
- `m` species mass. 
- `ch` species charge. 
- `maxnpart` max number of allocated particles. 

---

```c++
template<Dimension D, SimulationType S> Particles (const ParticleSpecies &)
```
Constructor for the particle container. This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts. 

**Parameters**
- `spec` species container. 

---

```c++
template<Dimension D, SimulationType S> auto loopParticles ()
```
Loop over all active particles. 

**Returns:**
- 1D Kokkos range policy of size of \fCnpart. 

---

```c++
template<Dimension D, SimulationType S> void set_npart (const std::size_t & N)
```
Set the number of particles. 

**Parameters**
- `npart` number of particles. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

