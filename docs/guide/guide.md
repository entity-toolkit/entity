---
hide:
  - footer
---

## Ranges

`Kokkos` loops accept `Kokkos::RangePolicy` or `Kokkos::MDRangePolicy` objects that specify the n-dimensional range of the loop. `Entity` embeds around these range objects and provides convenient aliases to the most common use cases. 

#### `[Simulation].m_mblock` methods

=== "`.rangeAllCells()`"

    | dimension | return object | ranges |
    |-----------|---------------|-------------|
    | 1D        | `Kokkos::RangePolicy<>` | $i_1\in [0,N_1)$ | 
    | 2D        | `Kokkos::MDRangePolicy<Rank<2>>` | $i_1\in [0,N_1)$, $i_2\in [0,N_2)$ | 
    | 3D        | `Kokkos::MDRangePolicy<Rank<3>>` | $i_1\in [0,N_1)$, $i_2\in [0,N_2)$, $i_3\in [0,N_3)$ | 

=== "`.rangeActiveCells()`"

    | dimension | return object | ranges |
    |-----------|---------------|-------------|
    | 1D        | `Kokkos::RangePolicy<>` | $i_1\in [N_G,N_1-N_G)$ | 
    | 2D        | `Kokkos::MDRangePolicy<Rank<2>>` | $i_1\in [N_G,N_1-N_G)$, $i_2\in [N_G,N_2-N_G)$ | 
    | 3D        | `Kokkos::MDRangePolicy<Rank<3>>` | $i_1\in [N_G,N_1-N_G)$, $i_2\in [N_G,N_2-N_G)$, $i_3\in [N_G,N_3-N_G)$ | 

=== "`.rangeCells(const boxRegion<D>&)`"

    `boxRegion<D>` is an auxiliary object (basically a tuple of size `D`) to specify a custom region of cells. It contains `CellLayer` objects by the number of dimensions. For example:
    ```c++
    boxRegion<2> left{CellLayer::minActiveLayer, CellLayer::allActiveLayer};
    ```


<!-- `Simulation::Meshblock` object has built-in `rangeAllCells()` and `rangeActiveCells()` methods that return, respectively, the range of all cells and the range of all the active cells.  -->



    ``` mermaid
    graph TD
      subgraph Meshblock
        direction LR
        subgraph Particles["Particle species"]
            subgraph Coordinates
                direction LR
                coord1["i1, i2, i3"]
                coord2["dx1, dx2, dx3"]
            end
            
            subgraph Velocities
                vel["ux1, ux2, ux3"]
            end
            wei
        end
        subgraph Fields["Field arrays"]
          direction LR
          subgraph MainFields["Main EM fields"]
            em1["em(*, em::ex1)"]
            em2["em(*, em::ex2)"]
            em3["em(*, em::ex3)"]
            em4["em(*, em::bx1)"]
            em5["em(*, em::bx2)"]
            em6["em(*, em::bx3)"]
          end
          subgraph BackupFields["Backup fields"]
            bckp1["bckp(*, em::ex1)"]
            bckp2["bckp(*, em::ex2)"]
            bckp3["bckp(*, em::ex3)"]
            bckp4["bckp(*, em::bx1)"]
            bckp5["bckp(*, em::bx2)"]
            bckp6["bckp(*, em::bx3)"]
          end
          subgraph CurrFields["Main currents"]
            cur1["cur(*, em::cur1)"]
            cur2["cur(*, em::cur2)"]
            cur3["cur(*, em::cur3)"]
          end
          subgraph BuffFields["Buffer fields"]
            buff1["buff(*, 1)"]
            buff2["buff(*, 2)"]
            buff3["buff(*, 3)"]
          end
        end
      end
    ```