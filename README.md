# `Entity` 

tl;dr: One particle-in-cell code to rule them all. 

`Entity` is an open-source coordinate-agnostic particle-in-cell (PIC) code written in C++17 specifically targeted to study plasma physics in relativistic astrophysical systems. The main algorithms of the code are written in covariant form, allowing to easily implement arbitrary grid geometries. The code is highly modular, and is written in the architecture-agnostic way using the [`Kokkos`](https://kokkos.org/kokkos-core-wiki/) performance portability library, allowing the code to efficiently use device parallelization on CPU and GPU architectures of different types. The multi-node parallelization is implemented using the `MPI` library, and the data output is done via the [`ADIOS2`](https://github.com/ornladios/ADIOS2) library which supports multiple output formats, including `HDF5` and `BP5`.

`Entity` is part of the `Entity toolkit` framework, which also includes a Python library for fast and efficient data analysis and visualization of the simulation data: [`nt2py`](https://pypi.org/project/nt2py/).

Our [detailed documentation](https://entity-toolkit.github.io/) includes everything you need to know to get started with using and/or contributing to the `Entity toolkit`. If you find bugs or issues, please feel free to add a GitHub issue or submit a pull request. Users with significant contributions to the code will be added to the list of developers, and assigned an emoji of their choice (important).

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Contributors (alphabetical)

* :guitar: Ludwig Böss {[@LudwigBoess](https://github.com/LudwigBoess)}
* :eyes: Yangyang Cai {[@StaticObserver](https://github.com/StaticObserver)}
* :person_tipping_hand: Alexander Chernoglazov {[@SChernoglazov](https://github.com/SChernoglazov)}
* :tea: Benjamin Crinquand {[@bcrinquand](https://github.com/bcrinquand)}
* :bubble_tea: Alisa Galishnikova {[@alisagk](https://github.com/alisagk)}
* :locomotive: Evgeny Gorbunov {[@Alcauchy](https://github.com/Alcauchy)}
* :coffee: Hayk Hakobyan {[@haykh](https://github.com/haykh)}
* :potato: Jens Mahlmann {[@jmahlmann](https://github.com/jmahlmann)}
* :dolphin: Sasha Philippov {[@sashaph](https://github.com/sashaph)}
* :radio: Siddhant Solanki {[@sidruns30](https://github.com/sidruns30)}
* :shrug: Arno Vanthieghem {[@vanthieg](https://github.com/vanthieg)}
* :cat: Muni Zhou {[@munizhou](https://github.com/munizhou)}

## Branch policy

Master branch contains the latest stable version of the code which has already been released. Development on the core is done on branches starting with `dev/`, while fixing bugs is done in branches that start with `bug/`. User-specific modifications (i.e., new problem generators plus perhaps minor corrections in the core) are done on branches starting with `pgen/`. Before merging to the master branch, all the branches must first be merged to the latest release-candidate branch, which ends with `rc`, via a pull request. After which, when all the release goals are met, the `rc` branch is merged to the master and released as a new stable version. Stale branches will be archived with a tag starting with `archive/` (can still be accessed via the "Tags" tab) and removed.
