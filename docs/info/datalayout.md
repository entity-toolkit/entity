!!! code

    Field quantities and particle coordinates are stored in the simulation object with following structure.

    ```cpp
    // fields (1)
    Simulation::Meshblock.em(i, j, k, fld::ex1);
    Simulation::Meshblock.em(i, j, k, fld::ex2);
    Simulation::Meshblock.em(i, j, k, fld::ex3);

    Simulation::Meshblock.em(i, j, k, fld::bx1);
    Simulation::Meshblock.em(i, j, k, fld::bx2);
    Simulation::Meshblock.em(i, j, k, fld::bx3);
    //                          ^  ^
    //                          |   \
    //                          |    not present in 1D/2D
    //                  not present in 1D

    // particles (2)
    Simulation::Meshblock.particles[species_id].i1(prtl_id);
    Simulation::Meshblock.particles[species_id].i2(prtl_id);  // not present in 1D
    Simulation::Meshblock.particles[species_id].i3(prtl_id);  // not present in 1D/2D
    Simulation::Meshblock.particles[species_id].dx1(prtl_id);
    Simulation::Meshblock.particles[species_id].dx2(prtl_id); // not present in 1D
    Simulation::Meshblock.particles[species_id].dx3(prtl_id); // not present in 1D/2D

    Simulation::Meshblock.particles[species_id].ux1(prtl_id);
    Simulation::Meshblock.particles[species_id].ux2(prtl_id);
    Simulation::Meshblock.particles[species_id].ux3(prtl_id);
    ```

    1. :grey_exclamation: Even though we employ `(i, j, k)` indexing in the code the field components are staggered not only in time but also spatially.
    2. :grey_exclamation: Particle velocities are staggered in time w.r.t. the coordinates.

