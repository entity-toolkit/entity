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
    if (species == 1) { // (2)!
      v[0] = 0.1;
    } else {
      v[0] = -0.1;
    }
  }
};
```

1. accepts three arguments: the coordinate, the velocity to be set, and the species index
2. set positive for electrons and negative for positrons

All we need to do now is to pass that class template to the `InjectUniform` routine as shown above. We can also read the drift velocity from the input file:

```cpp title="Reading the drift velocity from the input file" hl_lines="6 11 13 17-18"
template <Dimension D, SimulationEngine S>
struct WeibelDrift : public EnergyDistribution<D, S> {
  WeibelDrift(const SimulationParams& params, 
              const Meshblock<D, S>& mblock)
    : EnergyDistribution<D, S>(params, mblock),
      udrift { params.get<real_t>("problem", "udrift", 0.1) } {} // (2)!
  Inline void operator()(const coord_t<D>&,
                          vec_t<Dim3>& v,
                          const int&   species) const override {
    if (species == 1) {
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

That's it, the final result is shown below. To select your newly created problem generator during compilation, simply configure the code using `-D pgen=weibel_custom` (or whatever the filename was chosen before). To run the code with the new problem generator you will also need an input file, which you can learn how to write [here](../input).

```cpp title="Weibel problem generator (final)"
#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct WeibelDrift : public EnergyDistribution<D, S> {
    WeibelDrift(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        udrift { params.get<real_t>("problem", "udrift", 0.1) } {}
    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      if (species == 1) {
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

## Example 2 (pulsar)

As another example let us also write a problem generator for curvilinear coordinates; the pulsar magnetosphere is a perfect example as it will teach how to properly initialize fields, set boundary conditions and inject particles on-the-fly.

As discussed in the first example, start by creating a file in `src/pic/pgen/` directory.

```cpp title="src/pic/pgen/mypulsar.hpp"
#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"

namespace ntt {
  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {}
    inline void UserDriveFields(const real_t&,
                                const SimulationParams&,
                                Meshblock<D, S>&) override {}
    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}
  };

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> { // (1)!
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : TargetFields<D, S>(params, mblock) {}
    Inline real_t operator()(const em&, const coord_t<D>&) const override {
      return ZERO;
    }
  };

}    // namespace ntt

#endif
```

1. this will be used to drive the fields in the external absorbing region

In this example we will be focusing on 2D axisymmetric PIC case. Let us start by initializing a dipolar magnetic field filling the intire simulation domain. We will again be doing this in two ways (for educational purposes). Components of the magnetic field in the orthonormal (hatted) basis are given by

$$ 
\begin{aligned}
B^{\hat{r}} &= 2B_{\rm surf} \cos{\theta}\left(\frac{r_{\rm min}}{r}\right)^3,\\
B^{\hat{\theta}} &= B_{\rm surf} \sin{\theta}\left(\frac{r_{\rm min}}{r}\right)^3
\end{aligned}
$$

```cpp title="Initializing B-field (the hard way)"
template <>
inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
  const SimulationParams&, Meshblock<Dim2, PICEngine>& mblock) {
  auto   rmin  = mblock.metric.x1_min;    // (1)!
  real_t bsurf = 1.0;
  Kokkos::parallel_for(
    "UserInitFields",
    mblock.rangeActiveCells(), // (2)!
    Lambda(index_t i1, index_t i2) {
      const real_t i1_ { static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS) }; // (3)!
      const real_t i2_ { static_cast<real_t>(static_cast<int>(i2) - N_GHOSTS) };
      
      vec_t<Dim3>   b_cntrv { ZERO }, b_hat { ZERO };
      coord_t<Dim2> x_ph { ZERO };

      mblock.metric.x_Code2Sph({ i_, j_ + HALF }, x_ph); // (4)!
      b_hat[0] = TWO * math::cos(x_ph[1]) / CUBE(x_ph[0]);
      b_hat[1] = math::sin(x_ph[1]) / CUBE(x_ph[0]); // (5)!
      mblock.metric.v_Hat2Cntrv({ i_, j_ + HALF }, b_hat, b_cntrv);
      mblock.em(i, j, em::bx1) = bsurf * CUBE(rmin) * b_cntrv[0];

      mblock.metric.x_Code2Sph({ i_ + HALF, j_ }, x_ph); // (6)!
      b_hat[0] = TWO * math::cos(x_ph[1]) / CUBE(x_ph[0]);
      b_hat[1] = math::sin(x_ph[1]) / CUBE(x_ph[0]);
      mblock.metric.v_Hat2Cntrv({ i_ + HALF, j_ }, b_hat, b_cntrv);
      mblock.em(i, j, em::bx2) = bsurf * CUBE(rmin) * b_cntrv[1];
  });
}
```

1. in 2D axisymmetric the minimum value of the radial coordinate (rmin) is stored in `x1_min` field of the `metric` structure
2. traverse all the active cells in the meshblock
3. convert cell indices to real_t and shift by the number of ghost cells
4. `B^r` is defined at `i, j+1/2`. `x_ph` is now the coordinate in physical units in spherical coordinates (`r`, `theta`)
5. these fields are given in the orthonormal basis. we further convert them to contravariant basis
6. the same procedure has to be performed for the `B^theta` component, as it is defined at `i+1/2, j`

There are several ways `Entity` can simplify this workflow. First of all, let us put the definition of the dipolar field in a separate routine

```cpp title="Dipole field routine"
Inline void dipoleField(const coord_t<Dim2>& x_ph,
                          vec_t<Dim3>&, // (1)!
                          vec_t<Dim3>& b_out,
                          real_t       rmin,
                          real_t       bsurf) { // (2)!
  b_out[0] = bsurf * TWO * math::cos(x_ph[1]) / CUBE(x_ph[0] / rmin);
  b_out[1] = bsurf * math::sin(x_ph[1]) / CUBE(x_ph[0] / rmin);
}
```

1. dummy argument `e_out` for the electric field which we will not be using
2. the number of additional arguments here can be arbitrary. for now, we will only be using `rmin` and `bsurf`

To avoid making all the conversions and Yee-mesh specific shifting manually, `Entity` provides a convenient macro to streamline this process. Our new initialization routine should look like this:

```cpp title="Initializing B-field (the easy way)"
#include "field_macros.h"

// ...

template <>
inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
  const SimulationParams&, Meshblock<Dim2, PICEngine>& mblock) {
  const auto rmin = mblock.metric.x1_min;
  const real_t bsurf = 1.0;
  Kokkos::parallel_for(
    "UserInitFields", 
    mblock.rangeActiveCells(), 
    Lambda(index_t i1, index_t u2) {
      set_em_fields_2d(mblock, i1, i2, dipoleField, rmin, bsurf);
    });
}
```

And that is pretty much it. All the shifting, conversion etc is done implicitly by the macro. The only thing we have to do is to provide the field routine and the additional arguments. 

If you compile and run what we already have, you will see a rather unremarkable stationary dipole. To breath some life into it, we will add a rotation, imposed by the boundary conditions near $r_{\rm min}$. First, let us define a function, similar to `dipoleField` which defines the rotation electric field.

```cpp title="Rotating field routine"
Inline void surfaceRotationField(const coord_t<Dim2>& x_ph,
                                  vec_t<Dim3>&        e_out,
                                  vec_t<Dim3>&        b_out,
                                  real_t              rmin,
                                  real_t              bsurf,
                                  real_t              omega) {
  dipoleField(x_ph, e_out, b_out, rmin, bsurf); // (1)!
  e_out[1] = omega * bsurf * math::sin(x_ph[1]); // (2)!
  e_out[2] = 0.0;
}
```

1. `B^r` will be driven on the surface to the dipole field
2. we will only be driving the tangential electric field (`theta` and `phi` components)

To actually impose these boundary conditions, we need to implement the `UserDriveFields` function.

```cpp title="Field boundary conditions"
template <>
inline void ProblemGenerator<Dim2, PICEngine>::UserDriveFields(
  const real_t& time, const SimulationParams&, Meshblock<Dim2, PICEngine>& mblock) {
  {
    const auto bsurf      = 1.0; // (1)!
    const auto omega      = 1.0;
    const auto rmin       = mblock.metric.x1_min;
    const auto i1_min     = mblock.i1_min();
    const auto buff_cells = 5; // (2)!
    const auto i1_max     = mblock.i1_min() + buff_cells;
    NTTHostErrorIf(buff_cells > mblock.Ni1(), "buff_cells > ni1"); // (3)!

    Kokkos::parallel_for(
      "UserDriveFields_rmin", // (4)!
      CreateRangePolicy<Dim2>({ i1_min, mblock.i2_min() }, { i1_max, mblock.i2_max() }),
      Lambda(index_t i1, index_t i2) {
        if (i1 < i1_max - 1) { // (5)!
          mblock.em(i1, i2, em::ex1) = ZERO;
        }
        set_ex2_2d(mblock, i1, i2, surfaceRotationField, rmin, bsurf, omega);  // (6)!
        set_ex3_2d(mblock, i1, i2, surfaceRotationField, rmin, bsurf, omega);
        set_bx1_2d(mblock, i1, i2, surfaceRotationField, rmin, bsurf, omega);
      });
  }
  {
    const auto i1_max = mblock.i1_max();
    Kokkos::parallel_for(
      "UserDriveFields_rmax",
      CreateRangePolicy<Dim1>({ mblock.i2_min() }, { mblock.i2_max() }),
      Lambda(index_t i2) {
        mblock.em(i1_max, i2, em::ex2) = 0.0; // (7)!
        mblock.em(i1_max, i2, em::ex3) = 0.0;
        mblock.em(i1_max, i2, em::bx1) = 0.0;
      });
  }
}
```

1. we will be reading this from the parameters object further
2. it's always a good idea to drive the field boundary conditions within at least a few cells above `rmin` to avoid numerical artifacts
3. some sanity checks are always a good idea
4. boundary condition at `rmin`
5. resetting the `E^r` to zero is not strictly necessary, but it is a good idea to avoid numerical artifacts when interpolating the field to the particle positions
6. setting normal component of the `B` and tangential components of the `E` to the rotating field
7. similarly setting fields to zero at the uppermost boundary (not strictly necessary). notice that in this case the loop is 1D

At the outer boundary near $r_{\rm max}$ we often have an absorber for the EM fields. The absorber will attempt to damp the fields to particular user-defined values, specified in the `PgenTargetFields` class. In our case, we will set the B-fields to dipole and E-fields to zero. 

```cpp title="Absorber boundary conditions"
template <Dimension D, SimulationEngine S>
struct PgenTargetFields : public TargetFields<D, S> { // (1)!
  PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
    : TargetFields<D, S>(params, mblock), rmin { mblock.metric.x1_min }, bsurf { 1.0 } {}
  Inline real_t operator()(const em& comp, const coord_t<D>& xi) const override {
    if (comp == em::bx1 || comp == em::bx2) {
      vec_t<Dim3> e_out { ZERO }, b_out { ZERO };
      coord_t<D> x_ph { ZERO };
      (this->m_mblock).metric.x_Code2Sph(xi, x_ph); // (2)!
      dipoleField(x_ph, e_out, b_out, rmin, bsurf);
      return (comp == em::bx1) ? b_out[0] : b_out[1];
    } else {
      return ZERO;
    }
  }

private:
  const real_t rmin, bsurf;
};
```

1. must inherit from `TargetFields<D, S>` archetype
2. need to manually convert coordinates from code units to physical units

Vacuum pulsar magnetospheres are quite boring. Let us fill the simulation with some plasma. Unfortunately, simply initializing the plasma won't work, since it will just fly out of the simulation domain. Instead, we will need to constantly inject plasma from the inner boundary near $r_{\rm min}$. As in the Weibel example before, we will be using a pre-defined routine which does all the conversions etc. for us.

```cpp title="Plasma injection"
#include "injector.hpp"

// ...

template <>
inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
  const real_t& time, const SimulationParams& params, Meshblock<D, S>& mblock) override {
  auto nppc_per_spec = (real_t)(params.ppc0()) * 0.1; // (2)!
  InjectInVolume<D, S, ...>(params, mblock, { 1, 2 }, nppc_per_spec); // (1)!
}
```

1. instead of calling `InjectUniform` we now call `InjectInVolume`, which works with arbitrary spatial distributions without pre-defined number of particles to be injected
2. each timestep we inject a fraction of `ppc0`

As template arguments to the `InjectInVolume` function we may then pass three different classes: the energy distribution (inherited from `EnergyDistribution<D, S>`), spatial distribution (inherited from `SpatialDistribution<D, S>`), and the injection criterion (inherited from `InjectionCriterion<D, S>`). If not specified, the energy distribution will default to `ColdDist` (all velocities to zeros), the spatial distribution will default to uniform distribution, and the injection criterion -- to being always true. 

For our purposes we will be using the following classes:

```cpp title="Distributions and criteria"
template <Dimension D, SimulationEngine S>
struct RadialKick : public EnergyDistribution<D, S> {
  RadialKick(const SimulationParams& params, const Meshblock<D, S>& mblock)
    : EnergyDistribution<D, S>(params, mblock) {}
  Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v, const int&) const override {
    v[0] = 0.5; // (1)!
  }
};

template <Dimension D, SimulationEngine S>
struct InjectionShell : public SpatialDistribution<D, S> {
  explicit InjectionShell(const SimulationParams& params, Meshblock<D, S>& mblock)
    : SpatialDistribution<D, S>(params, mblock) {
    inj_rmax              = 1.5 * mblock.metric.x1_min;
    const int  buff_cells = 5;
    coord_t<D> xcu { ZERO }, xph { ZERO };
    xcu[0] = (real_t)buff_cells;
    mblock.metric.x_Code2Sph(xcu, xph);
    inj_rmin = xph[0];
  }
  Inline real_t operator()(const coord_t<D>& x_ph) const { // (2)!
    return ((x_ph[0] <= inj_rmax) && (x_ph[0] > inj_rmin)) ? ONE : ZERO;
  }

private:
  real_t inj_rmax, inj_rmin;
};

template <Dimension D, SimulationEngine S>
struct MaxDensCrit : public InjectionCriterion<D, S> {
  explicit MaxDensCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
    : InjectionCriterion<D, S>(params, mblock),
      inj_maxDens { 5.0 } {}
  Inline bool operator()(const coord_t<D>&) const {
    return false;
  }

private:
  const real_t inj_maxDens;
};

template <>
Inline bool MaxDensCrit<Dim2, PICEngine>::operator()(const coord_t<Dim2>& xph) const {
  coord_t<Dim2> xi { ZERO };
  (this->m_mblock).metric.x_Sph2Code(xph, xi);
  std::size_t i1 = (std::size_t)(xi[0] + N_GHOSTS);
  std::size_t i2 = (std::size_t)(xi[1] + N_GHOSTS);
  return (this->m_mblock).buff(i1, i2, fld::dens) < inj_maxDens; // (3)!
}
```

1. defining a slight radial kick to the particles
2. returns `1` (full density) or `0` (no density) depending on the radial position
3. return `false` when the local density is greater than `5 ppc0`

Now we can feed these structures to our injection routine.

```cpp title="Plasma injection" hl_lines="7-8"
template <>
inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
  const real_t&               time,
  const SimulationParams&     params,
  Meshblock<Dim2, PICEngine>& mblock) {
  auto nppc_per_spec = (real_t)(params.ppc0()) * 0.1; // (1)!
  InjectInVolume<Dim2, PICEngine, RadialKick, InjectionShell, MaxDensCrit>( // (2)!
    params, mblock, { 1, 2 }, nppc_per_spec);
}
```

1. in locations where `InjectionShell` returns `1` we inject `0.1 ppc0` particles per cell on average. in other locations this number will be scaled accordingly
2. the order of template arguments here is important: first goes the energy distribution, then the spatial distribution, and finally the injection criterion


Below is the final version with all the parameters read from the input file.

```cpp title="Final version"
#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

namespace ntt {
  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params)
      : bsurf { params.get<real_t>("problem", "bsurf", (real_t)(1.0)) },
        inj_fraction { params.get<real_t>("problem", "inj_fraction", (real_t)(0.1)) } {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {}
    inline void UserDriveFields(const real_t&,
                                const SimulationParams&,
                                Meshblock<D, S>&) override {}
    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

  private:
    const real_t bsurf, inj_fraction;
  };

  Inline void dipoleField(
    const coord_t<Dim2>& x_ph, vec_t<Dim3>&, vec_t<Dim3>& b_out, real_t rmin, real_t bsurf) {
    b_out[0] = bsurf * TWO * math::cos(x_ph[1]) / CUBE(x_ph[0] / rmin);
    b_out[1] = bsurf * math::sin(x_ph[1]) / CUBE(x_ph[0] / rmin);
  }

  Inline void surfaceRotationField(const coord_t<Dim2>& x_ph,
                                   vec_t<Dim3>&         e_out,
                                   vec_t<Dim3>&         b_out,
                                   real_t               rmin,
                                   real_t               bsurf,
                                   real_t               omega) {
    dipoleField(x_ph, e_out, b_out, rmin, bsurf);
    e_out[1] = omega * bsurf * math::sin(x_ph[1]);
    e_out[2] = 0.0;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    const auto rmin   = mblock.metric.x1_min;
    const auto bsurf_ = bsurf;
    Kokkos::parallel_for(
      "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, dipoleField, rmin, bsurf_);
      });
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveFields(
    const real_t& time, const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    {
      const auto omega      = params.get<real_t>("problem", "spin_omega");
      const auto rmin       = mblock.metric.x1_min;
      const auto bsurf_     = bsurf;
      const auto i1_min     = mblock.i1_min();
      const auto buff_cells = 5;
      const auto i1_max     = mblock.i1_min() + buff_cells;
      NTTHostErrorIf(buff_cells > mblock.Ni1(), "buff_cells > ni1");

      Kokkos::parallel_for(
        "UserDriveFields_rmin",
        CreateRangePolicy<Dim2>({ i1_min, mblock.i2_min() }, { i1_max, mblock.i2_max() }),
        Lambda(index_t i1, index_t i2) {
          if (i1 < i1_max - 1) {
            mblock.em(i1, i2, em::ex1) = ZERO;
          }
          set_ex2_2d(mblock, i1, i2, surfaceRotationField, rmin, bsurf_, omega);
          set_ex3_2d(mblock, i1, i2, surfaceRotationField, rmin, bsurf_, omega);
          set_bx1_2d(mblock, i1, i2, surfaceRotationField, rmin, bsurf_, omega);
        });
    }
    {
      const auto i1_max = mblock.i1_max();
      Kokkos::parallel_for(
        "UserDriveFields_rmax",
        CreateRangePolicy<Dim1>({ mblock.i2_min() }, { mblock.i2_max() }),
        Lambda(index_t i2) {
          mblock.em(i1_max, i2, em::ex2) = 0.0;
          mblock.em(i1_max, i2, em::ex3) = 0.0;
          mblock.em(i1_max, i2, em::bx1) = 0.0;
        });
    }
  }

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : TargetFields<D, S>(params, mblock),
        rmin { mblock.metric.x1_min },
        bsurf { params.get<real_t>("problem", "bsurf", (real_t)(1.0)) } {}
    Inline real_t operator()(const em& comp, const coord_t<D>& xi) const override {
      if (comp == em::bx1 || comp == em::bx2) {
        vec_t<Dim3> e_out { ZERO }, b_out { ZERO };
        coord_t<D>  x_ph { ZERO };
        (this->m_mblock).metric.x_Code2Sph(xi, x_ph);
        dipoleField(x_ph, e_out, b_out, rmin, bsurf);
        return (comp == em::bx1) ? b_out[0] : b_out[1];
      } else {
        return ZERO;
      }
    }

  private:
    const real_t rmin, bsurf;
  };

  template <Dimension D, SimulationEngine S>
  struct RadialKick : public EnergyDistribution<D, S> {
    RadialKick(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        u_kick { params.get<real_t>("problem", "u_kick", ZERO) } {}
    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v, const int&) const override {
      v[0] = u_kick;
    }

  private:
    const real_t u_kick;
  };

  template <Dimension D, SimulationEngine S>
  struct InjectionShell : public SpatialDistribution<D, S> {
    explicit InjectionShell(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {
      inj_rmax = params.get<real_t>("problem", "inj_rmax", 1.5 * mblock.metric.x1_min);
      const int  buff_cells = 5;
      coord_t<D> xcu { ZERO }, xph { ZERO };
      xcu[0] = (real_t)buff_cells;
      mblock.metric.x_Code2Sph(xcu, xph);
      inj_rmin = xph[0];
    }
    Inline real_t operator()(const coord_t<D>& x_ph) const {
      return ((x_ph[0] <= inj_rmax) && (x_ph[0] > inj_rmin)) ? ONE : ZERO;
    }

  private:
    real_t inj_rmax, inj_rmin;
  };

  template <Dimension D, SimulationEngine S>
  struct MaxDensCrit : public InjectionCriterion<D, S> {
    explicit MaxDensCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock),
        inj_maxDens { params.get<real_t>("problem", "inj_maxDens", 5.0) } {}
    Inline bool operator()(const coord_t<D>&) const {
      return false;
    }

  private:
    const real_t inj_maxDens;
  };

  template <>
  Inline bool MaxDensCrit<Dim2, PICEngine>::operator()(const coord_t<Dim2>& xph) const {
    coord_t<Dim2> xi { ZERO };
    (this->m_mblock).metric.x_Sph2Code(xph, xi);
    std::size_t i1 = (std::size_t)(xi[0] + N_GHOSTS);
    std::size_t i2 = (std::size_t)(xi[1] + N_GHOSTS);
    return (this->m_mblock).buff(i1, i2, fld::dens) < inj_maxDens;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
    const real_t& time, const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    auto nppc_per_spec = (real_t)(params.ppc0()) * inj_fraction;
    InjectInVolume<Dim2, PICEngine, RadialKick, InjectionShell, MaxDensCrit>(
      params, mblock, { 1, 2 }, nppc_per_spec);
  }

}    // namespace ntt

#endif
```