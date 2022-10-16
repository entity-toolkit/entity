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