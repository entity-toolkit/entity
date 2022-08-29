## class ntt::Simulation< D, S >

Main class of the simulation containing all the necessary methods and configurations.  

---

```c++
template<Dimension D, SimulationType S> Simulation (const toml::value & inputdata)
```
Constructor for simulation class. 

**Parameters**
- `inputdata` toml-object with parsed toml parameters. 

---

```c++
template<Dimension D, SimulationType S> auto rangeActiveCells ()
```
Loop over all active cells (disregard ghost cells). 

**Returns:**
- Kokkos range policy with proper min/max indices and dimension. 

---

```c++
template<Dimension D, SimulationType S> auto rangeAllCells ()
```
Loop over all cells. 

**Returns:**
- Kokkos range policy with proper min/max indices and dimension. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

