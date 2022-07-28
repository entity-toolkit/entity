## class ntt::Compute_Aphi< D >



---

## class ntt::GRPIC< D >



---

```c++
template<Dimension D> GRPIC (const toml::value & inputdata)
```
Constructor for GRPIC class.

**Parameters**
- `inputdata` toml-object with parsed toml parameters. 

---

```c++
template<Dimension D> void addCurrentsSubstep (const real_t &)
```
Add computed and filtered currents to the E-field.

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void ampereSubstep (const real_t &, const real_t &, const gr_ampere &)
```
Advance D-field using Ampere's law.

**Parameters**
- `t` time in physical units. 
- `c` coefficient that gets multiplied by the timestep (e.g., 0.5). 
- `f` either initial, intermediate or the main substep [\fCgr_ampere::init, \fCgr_ampere::aux, \fCgr_ampere::main]. 

---

```c++
template<Dimension D> void auxFieldBoundaryConditions (const real_t &, const gr_bc &)
```
Apply boundary conditions for auxiliary fields.

**Parameters**
- `t` time in physical units. 
- `f` select field to apply boundary conditions to [\fCgr_bc::Efield, \fCgr_bc::Hfield]. 

---

```c++
template<Dimension D> void computeAuxESubstep (const real_t &, const gr_getE &)
```
Compute E field.

**Parameters**
- `t` time in physical units. 
- `f` flag to use D0 and B or D and B0 [\fCgr_getE::D0_B, \fCgr_getE::D_B0]. 

---

```c++
template<Dimension D> void computeAuxHSubstep (const real_t &, const gr_getH &)
```
Compute H field.

**Parameters**
- `t` time in physical units. 
- `f` flat to use D0 and B0 or D and B0 [\fCgr_getH::D_B0, \fCgr_getH::D0_B0]. 

---

```c++
template<Dimension D> void computeVectorPotential ()
```
Computes Aphi. 

---

```c++
template<Dimension D> void copyFieldsGR ()
```
Copies em fields into em0. 

---

```c++
template<Dimension D> void depositCurrentsSubstep (const real_t &)
```
Deposit currents from particles.

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void faradaySubstep (const real_t &, const real_t &, const gr_faraday &)
```
Advance B-field using Faraday's law.

**Parameters**
- `t` time in physical units. 
- `c` coefficient that gets multiplied by the timestep (e.g., 0.5). 
- `f` either intermediate substep or the main one [\fCgr_faraday::aux, \fCgr_faraday::main]. 

---

```c++
template<Dimension D> void fieldBoundaryConditions (const real_t &, const gr_bc &)
```
Apply boundary conditions for fields.

**Parameters**
- `t` time in physical units. 
- `f` select field to apply boundary conditions to [\fCgr_bc::Dfield, \fCgr_bc::Bfield]. 

---

```c++
template<Dimension D> void filterCurrentsSubstep (const real_t &)
```
Spatially filter all the deposited currents.

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void initial_step (const real_t &)
```
Advance the simulation forward for one timestep.

**Parameters**
- `t` time in physical unitsFrom the initial fields, advances the first time steps.
- `t` time in physical units 

---

```c++
template<Dimension D> void mainloop ()
```
Advance the simulation forward for a specified amount of timesteps, keeping track of time. 

---

```c++
template<Dimension D> void process ()
```
Process the simulation (calling initialize, verify, mainloop, etc). 

---

```c++
template<Dimension D> void resetCurrents (const real_t &)
```
Reset current arrays.

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void resetFields (const real_t &)
```
Reset field arrays.

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void resetParticles (const real_t &)
```
Reset particles.

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void step_forward (const real_t &)
```
Advance the simulation forward for one timestep.

**Parameters**
- `t` time in physical units 

---

```c++
template<Dimension D> void timeAverageDBSubstep (const real_t &)
```
Time average EM fields.

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void timeAverageJSubstep (const real_t &)
```
Time average currents.

**Parameters**
- `t` time in physical units. 

---

```c++
template<Dimension D> void transformCurrentsSubstep (const real_t &)
```
Transform the deposited currents to coordinate basis.

**Parameters**
- `t` time in physical units. 

---

###### API documentation generated using [Doxygenmd](https://github.com/d99kris/doxygenmd)

