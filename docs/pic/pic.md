## class ntt::PIC< D >

Class for PIC simulations, inherits from \fCSimulation<D, SimulationType::PIC>.  

---

```c++
template<Dimension D> PIC (const toml::value & inputdata)
```
Constructor for PIC class. 

**Parameters**
- `inputdata` toml-object with parsed toml parameters. 

---

```c++
template<Dimension D> void addCurrentsSubstep (const real_t & t)
```
Add computed and filtered currents to the E-field. 

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void ampereSubstep (const real_t & t, const real_t & f)
```
Advance E-field using Ampere's law (without currents). 

**Parameters**
- `t` time in physical units. 
- `f` coefficient that gets multiplied by the timestep (e.g., 1.0). 

---

```c++
template<Dimension D> void depositCurrentsSubstep (const real_t & t)
```
Deposit currents from particles. 

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void faradaySubstep (const real_t & t, const real_t & f)
```
Advance B-field using Faraday's law. 

**Parameters**
- `t` time in physical units. 
- `f` coefficient that gets multiplied by the timestep (e.g., 0.5). 

---

```c++
template<Dimension D> void fieldBoundaryConditions (const real_t & t)
```
Apply boundary conditions for fields. 

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void filterCurrentsSubstep (const real_t & t)
```
Spatially filter all the deposited currents. 

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void initial_step (const real_t &)
```
Dummy function to match with GRPIC. 

**Parameters**
- `time` in physical units 

---

```c++
template<Dimension D> void particleBoundaryConditions (const real_t & t)
```
Apply boundary conditions for particles. 

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void pushParticlesSubstep (const real_t & t, const real_t & f)
```
Advance particle positions and velocities. 

**Parameters**
- `t` time in physical units. 
- `f` coefficient that gets multiplied by the timestep (e.g., 1.0). 

---

```c++
template<Dimension D> void resetCurrents (const real_t & t)
```
Reset current arrays. 

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void resetFields (const real_t & t)
```
Reset field arrays. 

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void resetParticles (const real_t & t)
```
Reset particles. 

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void step_backward (const real_t & t)
```
Advance the simulation forward for one timestep. 

**Parameters**
- `t` time in physical units 

---

```c++
template<Dimension D> void step_forward (const real_t & t)
```
Advance the simulation forward for one timestep. 

**Parameters**
- `t` time in physical units 

---

```c++
template<Dimension D> void transformCurrentsSubstep (const real_t & t)
```
Transform the deposited currents to coordinate basis. 

**Parameters**
- `t` time in physical units. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

