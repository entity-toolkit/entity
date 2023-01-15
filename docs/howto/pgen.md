---
hide:
  - footer
---

# Writing a problem generator

Problem generators are the main entry point for the user to specify the initial conditions of the simulation, any intervention routines (like injection or driving), as well as specific boundary conditions. All the problem generators are located in `src/<engine>/pgen/` directory. The problem generator is specified using the `-D pgen=<PROBLEM_GENERATOR>` flag when configuring the code with `cmake`. 

All problem generators should inherit from the master class `PGen<D, S>` defined in `src/framework/archetypes.hpp`. There are several basic routines that optionally need to be implemented in the derived class:

* `PGen<D, S>::UserInitParticles`: initializes the particle data (e.g. positions, velocities, etc.; called only once in the beginning).
* `PGen<D, S>::UserInitFields`: initializes the field data ($E$, and $B$; called only once in the beginning).
* `PGen<D, S>::UserDriveParticles`: drives the particles (e.g. injects particles, applies external forces, etc.; called every time step after `ParticlesPush` and before `CurrentsDeposit`).
* `PGen<D, S>::UserDriveFields`: drives the fields (e.g. custom boundary conditions, etc.; called every time step after every `Ampere` and `Faraday` call, together with `FieldsBoundaryConditions`).

## Example 1 (Weibel)

In this example we will be implementing a simple problem generator to simulate the Weibel instability with the `PIC` engine in flat space-time.

### Preparation

First, we create a file named `weibel_custom.hpp` in the `src/pic/pgen/` directory. We then create a new class template named `ProblemGenerator` which inerits from `PGen`:

```cpp title="weibel_custom.hpp"
#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h" // (1)!

#include "sim_params.h" // (2)!

#include "archetypes.hpp" // (3)!

namespace ntt {
  template <Dimension D, SimulationEngine S> // (4)!
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {} // (5)!
  };
}

#endif
```

1. this header should always be included
2. for the `SimulationParams` struct that contains all the parameters of the simulation
3. for the `PGen` class template
4. template arguments for the `ProblemGenerator` class specifying the dimension and the simulation engine
5. empty contructor

Right now the problem generator doesn't do much. When the simulation calls `UserInitParticles`, the default implementation from `PGen` is invoked which does not do anything. To override this, we can simply add:

```cpp title="Specialized implementation of the `UserInitParticles`" hl_lines="7 9-13"
// ...
namespace ntt {
  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    // ...
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {} // (1)!
  };

  template <> // (2)!
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    // ... custom implementation
  }
}
```

1. overriding the default implementation
2. customizing for 2D PIC simulation

Now any time `ProblemGenerator<Dim2, PICEngine>::UserInitParticles` is called, it will use the custom implementation, while the default implementation will be used for any other combination of template arguments.

### Initializing particles (the hard way)

As one can see, `UserInitParticles` receives two parameters: `SimulationParams` and `Meshblock`. The former contains all the constant parameters read from the user-specified input. The `Meshblock` object, on the other hand, contains all the field arrays, particle arrays, metric functions, as well as other (deduced) parameters, such as the timestep duration, the number of particles of each species, etc. 

We can "inject" particles into the simulation by simply doing:

```cpp title="Setting the number of active particles" hl_lines="5 7"
template <>
inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
  const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
  // for species #1
  mblock.particles[0].setNpart(1000);
  // for species #2
  mblock.particles[1].setNpart(1000);
}
```

This is rather unremarkable, since it will simply tell the simulation that there are 1000 active particles of each of the two particle species, but these particles by default will have zeros for all their coordinates and velocities. 

There are then two ways to initialize the particles: the hard way and the easy way. For educational purposes, let us first do that manually, the hard way:

```cpp title="Initializing particles the hard way"
template <>
inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
  const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
  auto& electrons = mblock.particles[0]; // (1)!
  auto& positrons = mblock.particles[1];
  electrons.setNpart(1000);
  positrons.setNpart(1000);

  auto random_pool = *(mblock.random_pool_ptr); // (2)!
  real_t Xmin = mblock.metric.x1_min; // (3)!
  real_t Xmax = mblock.metric.x1_max;
  real_t Ymin = mblock.metric.x2_min;
  real_t Ymax = mblock.metric.x2_max;

  Kokkos::parallel_for( // (4)!
    "SetWeibelParticles", // <-- optional name
    1000, // <-- loop range
    Lambda(index_t p) { // <-- loop index
      typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state(); // (5)!
      real_t rx { rand_gen.frand(Xmin, Xmax) }; // (6)!
      real_t ry { rand_gen.frand(Ymin, Ymax) };

      coord_t<Dim2> x_CU;
      mblock.metric.x_Cart2Code({ rx, ry }, x_CU); // (7)!

      int i = static_cast<int>(x_CU[0]), j = static_cast<int>(x_CU[1]); // (8)!
      float dx = static_cast<float>(x_CU[0]) - static_cast<float>(i);
      float dy = static_cast<float>(x_CU[1]) - static_cast<float>(j);

      electrons.i1(p) = i; // (9)!
      electrons.i2(p) = j;
      electrons.dx1(p) = dx;
      electrons.dx2(p) = dy;
      
      positrons.i1(p) = i; // (10)!
      positrons.i2(p) = j;
      positrons.dx1(p) = dx;
      positrons.dx2(p) = dy;

      electrons.ux1(p) = -0.1; // (11)!
      positrons.ux1(p) = 0.1;

      random_pool.free_state(rand_gen); // (12)!
  });
}
```

1. for simplicity defining the references to particle species
2. we will be using random number generator so we access the random pool stored on the meshblock
3. we also need to know the extent of the meshblock in physical units
4. loop over 1000 active particles
5. get the random state
6. generate random numbers in the range `[Xmin, Xmax[` and `[Ymin, Ymax[`
7. the generated coordinates were in physical units, so we need to convert them to code units
8. `x_CU` need to further be converted, since particles store their coordinates in the form of integer indices and fractional offsets
9. now finally we can store these into particle data
10. initialize the second species at the exact same position (important for charge conservation)
11. the velocities for PIC engine are stored in global Cartesian basis so in our case it is quite trivial. the value of the velocity can be read from the input (shown later), for now we simply make two counterstreaming beams in `x`. the rest of the components are left to zero
12. it's important to release the state back to the pool

There are a few ways one may simplify this. For example, we provide the most commonly used code patterns as c++ macros, which can be found in `src/framework/particle_macros.h`. Using some of these macros one could simplify the above code to:

```cpp title="Initializing particles the hard way (slighly simplified)" hl_lines="13-14"
#include "particle_macros.h"

// ...

Kokkos::parallel_for(
  "SetWeibelParticles",
  1000,
  Lambda(index_t p) {
    typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();
    real_t rx { rand_gen.frand(Xmin, Xmax) };
    real_t ry { rand_gen.frand(Ymin, Ymax) };

    init_prtl_2d(mblock, electrons, p, rx, ry, 0.1, 0.0, 0.0); // (1)!
    init_prtl_2d(mblock, positrons, p, rx, ry, -0.1, 0.0, 0.0);

    random_pool.free_state(rand_gen);
});
```

1. no need to convert the coordinates to code units. the macro accepts 2 coordinate components in physical units + 3 velocity components in global Cartesian basis

This is already much simpler and cleaner, however in more complex scenarios, when one needs to inject a particular distribution function, or inject particles based on certain criterion, it may quickly become cumbersome to implement these routines. 

### Initializing particles (the easy way)

`Entity` provides a multitude of pre-fabricated tools to make common operations, like the initialization of particles, easier. In this particular example, we will be relying on routines defined in the `src/framework/injector.hpp`, and our final code should look something like this:

```cpp title="Initializing particles the easy way (final)" hl_lines="6"
#include "injector.hpp"

template <>
inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
  const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
  InjectUniform<Dim2, PICEngine>(params, mblock, { 1, 2 }, params.ppc0() * 0.5); // (1)!
  //                                                 ^
  //                                                 |
  //                                        species to be injected
  //                         numbers are consistent with the ones defined in the input
}
```

1. here instead of injecting 1000 particles, we now inject on average `ppc0` particles (both species combined) per cell.

Notice, that we used the `InjectUniform` routine, which injects a uniformly distributed pre-defined number of particles per cell. Notice, however, that we have not specified the velocity distribution here. By default the velocity distribution is set to zero.

To generate a non-trivial distribution of plasma, we need to provide an additional template argument to the `InjectUniform` routine:

```cpp title="Initializing particles the easy way (final)" hl_lines="6"
InjectUniform<Dim2, PICEngine, WeibelDrift>(params, mblock, { 1, 2 }, params.ppc0() * 0.5);
```

This `WeibelDrift` is a template class, which we will define further, which specifies the velocity distribution for the injected particles. The `WeibelDrift` class should inherit from the `EnergyDistribution<D, S>` class and override the `void operator()(...)`.

```cpp title="Defining a custom energy distribution"
template <Dimension D, SimulationEngine S>
struct WeibelDrift : public EnergyDistribution<D, S> {
  WeibelDrift(const SimulationParams& params, 
              const Meshblock<D, S>& mblock)
    : EnergyDistribution<D, S>(params, mblock) {}
  Inline void operator()(const coord_t<D>&,
                          vec_t<Dim3>& v,
                          const int&   species) const override { // (1)!
    if (species == 0) { // (2)!
      v[0] = 0.1;
    } else {
      v[0] = -0.1;
    }
  }
};
```

1. accepts three arguments: the coordinate, the velocity to be set, and the species number (0 or 1)
2. set positive for electrons and negative for positrons

All we need to do now is to pass that class template to the `InjectUniform` routine as shown above. We can also read the drift velocity from the input file:

```cpp title="Reading the drift velocity from the input file" hl_lines="6 11 13 17-18"
template <Dimension D, SimulationEngine S>
struct WeibelDrift : public EnergyDistribution<D, S> {
  WeibelDrift(const SimulationParams& params, 
              const Meshblock<D, S>& mblock)
    : EnergyDistribution<D, S>(params, mblock),
      udrift { readFromInput<real_t>(params.inputdata(), "problem", "udrift", 0.1) } {} // (1)!
  Inline void operator()(const coord_t<D>&,
                          vec_t<Dim3>& v,
                          const int&   species) const override {
    if (species == 0) {
      v[0] = udrift;
    } else {
      v[0] = -udrift;
    }
  }

private:
  const real_t udrift;
};
```

1. the `0.1` here is the optional default value, which will be used if the parameter is not specified in the input file

### Final result

That's it, the final result is shown below. To select your newly created problem generator during compilation, simply configure the code using `-D pgen=weibel_custom` (or whatever the filename was chosen before). To run the code with the new problem generator you will also need an input file, which you can learn how to write [here](/howto/input).

```cpp title="Weibel problem generator (final)"
#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "input.h"
#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct WeibelDrift : public EnergyDistribution<D, S> {
    WeibelDrift(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        udrift { readFromInput<real_t>(params.inputdata(), "problem", "udrift", 0.1) } {}
    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      if (species == 0) {
        v[0] = udrift;
      } else {
        v[0] = -udrift;
      }
    }

  private:
    const real_t udrift;
  };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}
  };

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    InjectUniform<Dim2, PICEngine, WeibelDrift>(params, mblock, { 1, 2 }, params.ppc0() * 0.5);
  }
} // namespace ntt

#endif
```