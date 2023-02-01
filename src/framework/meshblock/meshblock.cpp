#include "meshblock.h"

#include "wrapper.h"

#include "particles.h"
#include "sim_params.h"

namespace ntt {
  template <Dimension D>
  Mesh<D>::Mesh(const std::vector<unsigned int>& res,
                const std::vector<real_t>&       ext,
                const real_t*                    params)
    : m_i1min { res.size() > 0 ? N_GHOSTS : 0 },
      m_i1max { res.size() > 0 ? N_GHOSTS + (int)(res[0]) : 1 },
      m_i2min { res.size() > 1 ? N_GHOSTS : 0 },
      m_i2max { res.size() > 1 ? N_GHOSTS + (int)(res[1]) : 1 },
      m_i3min { res.size() > 2 ? N_GHOSTS : 0 },
      m_i3max { res.size() > 2 ? N_GHOSTS + (int)(res[2]) : 1 },
      m_Ni1 { res.size() > 0 ? (int)(res[0]) : 1 },
      m_Ni2 { res.size() > 1 ? (int)(res[1]) : 1 },
      m_Ni3 { res.size() > 2 ? (int)(res[2]) : 1 },
      metric { res, ext, params } {}

  template <>
  auto Mesh<Dim1>::rangeAllCells() -> range_t<Dim1> {
    boxRegion<Dim1> region { CellLayer::allLayer };
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim2>::rangeAllCells() -> range_t<Dim2> {
    boxRegion<Dim2> region { CellLayer::allLayer, CellLayer::allLayer };
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim3>::rangeAllCells() -> range_t<Dim3> {
    boxRegion<Dim3> region { CellLayer::allLayer, CellLayer::allLayer, CellLayer::allLayer };
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim1>::rangeActiveCells() -> range_t<Dim1> {
    boxRegion<Dim1> region { CellLayer::activeLayer };
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim2>::rangeActiveCells() -> range_t<Dim2> {
    boxRegion<Dim2> region { CellLayer::activeLayer, CellLayer::activeLayer };
    return rangeCells(region);
  }
  template <>
  auto Mesh<Dim3>::rangeActiveCells() -> range_t<Dim3> {
    boxRegion<Dim3> region { CellLayer::activeLayer,
                             CellLayer::activeLayer,
                             CellLayer::activeLayer };
    return rangeCells(region);
  }

  template <Dimension D>
  auto Mesh<D>::rangeCells(const boxRegion<D>& region) -> range_t<D> {
    tuple_t<int, D> imin, imax;
    for (short i = 0; i < (short)D; i++) {
      switch (region[i]) {
      case CellLayer::allLayer:
        imin[i] = 0;
        imax[i] = Ni(i) + 2 * N_GHOSTS;
        break;
      case CellLayer::activeLayer:
        imin[i] = i_min(i);
        imax[i] = i_max(i);
        break;
      case CellLayer::minGhostLayer:
        imin[i] = 0;
        imax[i] = i_min(i);
        break;
      case CellLayer::minActiveLayer:
        imin[i] = i_min(i);
        imax[i] = i_min(i) + N_GHOSTS;
        break;
      case CellLayer::maxActiveLayer:
        imin[i] = i_max(i) - N_GHOSTS;
        imax[i] = i_max(i);
        break;
      case CellLayer::maxGhostLayer:
        imin[i] = i_max(i);
        imax[i] = Ni(i) + 2 * N_GHOSTS;
        break;
      default:
        NTTHostError("Invalid cell layer");
      }
    }
    return CreateRangePolicy<D>(imin, imax);
  }

  // !TODO: too ugly, implement a better solution (combine with device)
  template <Dimension D>
  auto Mesh<D>::rangeCellsOnHost(const boxRegion<D>& region) -> range_h_t<D> {
    tuple_t<int, D> imin, imax;
    for (short i = 0; i < (short)D; i++) {
      switch (region[i]) {
      case CellLayer::allLayer:
        imin[i] = 0;
        imax[i] = Ni(i) + 2 * N_GHOSTS;
        break;
      case CellLayer::activeLayer:
        imin[i] = i_min(i);
        imax[i] = i_max(i);
        break;
      case CellLayer::minGhostLayer:
        imin[i] = 0;
        imax[i] = i_min(i);
        break;
      case CellLayer::minActiveLayer:
        imin[i] = i_min(i);
        imax[i] = i_min(i) + N_GHOSTS;
        break;
      case CellLayer::maxActiveLayer:
        imin[i] = i_max(i) - N_GHOSTS;
        imax[i] = i_max(i);
        break;
      case CellLayer::maxGhostLayer:
        imin[i] = i_max(i);
        imax[i] = Ni(i) + 2 * N_GHOSTS;
        break;
      default:
        NTTHostError("Invalid cell layer");
      }
    }
    return CreateRangePolicyOnHost<D>(imin, imax);
  }

  template <>
  auto Mesh<Dim1>::rangeAllCellsOnHost() -> range_h_t<Dim1> {
    boxRegion<Dim1> region { CellLayer::allLayer };
    return rangeCellsOnHost(region);
  }
  template <>
  auto Mesh<Dim2>::rangeAllCellsOnHost() -> range_h_t<Dim2> {
    boxRegion<Dim2> region { CellLayer::allLayer, CellLayer::allLayer };
    return rangeCellsOnHost(region);
  }
  template <>
  auto Mesh<Dim3>::rangeAllCellsOnHost() -> range_h_t<Dim3> {
    boxRegion<Dim3> region { CellLayer::allLayer, CellLayer::allLayer, CellLayer::allLayer };
    return rangeCellsOnHost(region);
  }
  template <>
  auto Mesh<Dim1>::rangeActiveCellsOnHost() -> range_h_t<Dim1> {
    boxRegion<Dim1> region { CellLayer::activeLayer };
    return rangeCellsOnHost(region);
  }
  template <>
  auto Mesh<Dim2>::rangeActiveCellsOnHost() -> range_h_t<Dim2> {
    boxRegion<Dim2> region { CellLayer::activeLayer, CellLayer::activeLayer };
    return rangeCellsOnHost(region);
  }
  template <>
  auto Mesh<Dim3>::rangeActiveCellsOnHost() -> range_h_t<Dim3> {
    boxRegion<Dim3> region { CellLayer::activeLayer,
                             CellLayer::activeLayer,
                             CellLayer::activeLayer };
    return rangeCellsOnHost(region);
  }

  template <Dimension D>
  auto Mesh<D>::rangeCells(const tuple_t<tuple_t<int, Dim2>, D>& ranges) -> range_t<D> {
    tuple_t<int, D> imin, imax;
    for (short i = 0; i < (short)D; i++) {
      if ((ranges[i][0] < -N_GHOSTS) || (ranges[i][1] > N_GHOSTS)) {
        NTTHostError("Invalid cell layer picked");
      }
      imin[i] = i_min(i) + ranges[i][0];
      imax[i] = i_max(i) + ranges[i][1];
      if (imin[i] >= imax[i]) {
        NTTHostError("Invalid cell layer picked");
      }
    }
    return CreateRangePolicy<D>(imin, imax);
  }

  template <>
  auto Mesh<Dim1>::extent() const -> std::vector<real_t> {
    return { metric.x1_min, metric.x1_max };
  }

  template <>
  auto Mesh<Dim2>::extent() const -> std::vector<real_t> {
    return { metric.x1_min, metric.x1_max, metric.x2_min, metric.x2_max };
  }

  template <>
  auto Mesh<Dim3>::extent() const -> std::vector<real_t> {
    return { metric.x1_min, metric.x1_max, metric.x2_min,
             metric.x2_max, metric.x3_min, metric.x3_max };
  }

  template <Dimension D, SimulationEngine S>
  Meshblock<D, S>::Meshblock(const std::vector<unsigned int>&    res,
                             const std::vector<real_t>&          ext,
                             const real_t*                       params,
                             const std::vector<ParticleSpecies>& species)
    : Mesh<D>(res, ext, params), Fields<D, S>(res) {
    for (auto& part : species) {
      particles.emplace_back(part);
    }
  }

  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::Verify() {
    // verifying that the correct particle arrays are allocated for a given dimension ...
    // ... and a given simulation engine
    for (auto& species : particles) {
      if constexpr (D == Dim1) {
        NTTHostErrorIf(
          (species.i2.extent(0) != 0) || (species.i3.extent(0) != 0)
            || (species.dx2.extent(0) != 0) || (species.dx3.extent(0) != 0)
            || (species.i2_prev.extent(0) != 0) || (species.i3_prev.extent(0) != 0)
            || (species.dx2_prev.extent(0) != 0) || (species.dx3_prev.extent(0) != 0),
          "Wrong particle arrays allocated for 1D mesh");
        if constexpr (S == PICEngine) {
          NTTHostErrorIf((species.i1_prev.extent(0) != 0) || (species.dx1_prev.extent(0) != 0),
                         "Wrong particle arrays allocated for 1D mesh PIC");
        }
#ifdef MINKOWSKI_METRIC
        NTTHostErrorIf(species.phi.extent(0) != 0,
                       "Wrong particle arrays allocated for 1D mesh MINKOWSKI");
#endif
      } else if constexpr (D == Dim2) {
        NTTHostErrorIf((species.i3.extent(0) != 0) || (species.dx3.extent(0) != 0)
                         || (species.i3_prev.extent(0) != 0)
                         || (species.dx3_prev.extent(0) != 0),
                       "Wrong particle arrays allocated for 2D mesh");
        if constexpr (S == PICEngine) {
          NTTHostErrorIf((species.i1_prev.extent(0) != 0) || (species.dx1_prev.extent(0) != 0)
                           || (species.i2_prev.extent(0) != 0)
                           || (species.dx2_prev.extent(0) != 0),
                         "Wrong particle arrays allocated for 2D mesh PIC");
        }
#ifdef MINKOWSKI_METRIC
        NTTHostErrorIf(species.phi.extent(0) != 0,
                       "Wrong particle arrays allocated for 2D mesh MINKOWSKI");
#endif
      } else {
        if constexpr (S == PICEngine) {
          NTTHostErrorIf(
            (species.i1_prev.extent(0) != 0) || (species.dx1_prev.extent(0) != 0)
              || (species.i2_prev.extent(0) != 0) || (species.dx2_prev.extent(0) != 0)
              || (species.i3_prev.extent(0) != 0) || (species.dx3_prev.extent(0) != 0),
            "Wrong particle arrays allocated for 2D mesh PIC");
        }
#ifdef MINKOWSKI_METRIC
        NTTHostErrorIf(species.phi.extent(0) != 0,
                       "Wrong particle arrays allocated for 2D mesh MINKOWSKI");
#endif
      }
    }
  }

  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::SynchronizeHostDevice(const SynchronizeFlags& flags) {
    if (flags & Synchronize_em) {
      Kokkos::deep_copy(this->em_h, this->em);
      this->em_h_content = this->em_content;
    }
    if (flags & Synchronize_cur) {
      Kokkos::deep_copy(this->cur_h, this->cur);
      this->cur_h_content = this->cur_content;
    }
    if (flags & Synchronize_buff) {
      Kokkos::deep_copy(this->buff_h, this->buff);
      this->buff_h_content = this->buff_content;
    }
    if (flags & Synchronize_bckp) {
      Kokkos::deep_copy(this->bckp_h, this->bckp);
      this->bckp_h_content = this->bckp_content;
    }
  }
}    // namespace ntt

template class ntt::Mesh<ntt::Dim1>;
template class ntt::Mesh<ntt::Dim2>;
template class ntt::Mesh<ntt::Dim3>;

#ifdef PIC_ENGINE
template class ntt::Meshblock<ntt::Dim1, ntt::PICEngine>;
template class ntt::Meshblock<ntt::Dim2, ntt::PICEngine>;
template class ntt::Meshblock<ntt::Dim3, ntt::PICEngine>;
#elif defined(GRPIC_ENGINE)
template class ntt::Meshblock<ntt::Dim2, ntt::SimulationEngine::GRPIC>;
template class ntt::Meshblock<ntt::Dim3, ntt::SimulationEngine::GRPIC>;
#endif