## class ntt::Mesh< D >

Container for the meshgrid information (cell ranges etc).  

---

```c++
template<Dimension D> Mesh (const std::vector< unsigned int > & res, const std::vector< real_t > & ext, const real_t * params)
```
Constructor for the mesh container, sets the active cell sizes and ranges. 

**Parameters**
- `res` resolution vector of size D (dimension). 
- `ext` extent vector of size 2 * D. 
- `params` metric-/domain-specific parameters (max: 10). 

---

```c++
template<Dimension D> auto loopActiveCells ()
```
Loop over all active cells (disregard ghost cells). 

**Returns:**
- Kokkos range policy with proper min/max indices and dimension. 

---

```c++
template<Dimension D> auto loopAllCells ()
```
Loop over all cells. 

**Returns:**
- Kokkos range policy with proper min/max indices and dimension. 

---

## class ntt::Meshblock< D, S >

Container for the fields, particles and coordinate system. This is the main subject of the simulation.  

---

```c++
template<Dimension D, SimulationType S> Meshblock (const std::vector< unsigned int > & res, const std::vector< real_t > & ext, const real_t * params, const std::vector< ParticleSpecies > & species)
```
Constructor for the meshblock. 

**Parameters**
- `res` resolution vector of size D (dimension). 
- `ext` extent vector of size 2 * D. 
- `params` metric-/domain-specific parameters (max: 10). 
- `species` vector of particle species parameters. 

---

```c++
template<Dimension D, SimulationType S> void set_min_cell_size (const real_t & min_cell_size)
```
Set the minimum cell size of the meshblock. 

**Parameters**
- `min_cell_size` minimum cell size in physical units. 

---

```c++
template<Dimension D, SimulationType S> void set_timestep (const real_t & timestep)
```
Set the timestep of the meshblock. 

**Parameters**
- `timestep` timestep in physical units. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

