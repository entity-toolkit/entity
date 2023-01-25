#include "fields.h"

#include "wrapper.h"

#include <plog/Log.h>

#include <vector>

namespace ntt {
  void AssertEmptyContent(const std::vector<Content>& content) {
    for (auto c : content) {
      NTTHostErrorIf(c != Content::empty, "Content is not empty.");
    }
  }

  void AssertContent(const std::vector<Content>& content, const std::vector<Content>& target) {
    NTTHostErrorIf(content.size() != target.size(), "Content size mismatch.");
    for (unsigned int i = 0; i < content.size(); ++i) {
      NTTHostErrorIf(content[i] != target[i], "Content mismatch.");
    }
  }

  void ImposeContent(std::vector<Content>& content, const std::vector<Content>& target) {
    NTTHostErrorIf(content.size() != target.size(), "Content size mismatch.");
    for (unsigned int i = 0; i < content.size(); ++i) {
      content[i] = target[i];
    }
  }

  void ImposeEmptyContent(std::vector<Content>& content) {
    for (auto& c : content) {
      c = Content::empty;
    }
  }

  using resolution_t = std::vector<unsigned int>;

#ifdef PIC_ENGINE
  // * * * * * * * * * * * * * * * * * * * *
  // PIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Fields<Dim1, PICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS },
      bckp { "bckp", res[0] + 2 * N_GHOSTS } {
    PLOGD << "Allocated field arrays.";
    em_h   = Kokkos::create_mirror(em);
    cur_h  = Kokkos::create_mirror(cur);
    buff_h = Kokkos::create_mirror(buff);
    bckp_h = Kokkos::create_mirror(bckp);
  }

  template <>
  Fields<Dim2, PICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      bckp { "bckp", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS } {
    PLOGD << "Allocated field arrays.";
    em_h   = Kokkos::create_mirror(em);
    cur_h  = Kokkos::create_mirror(cur);
    buff_h = Kokkos::create_mirror(buff);
    bckp_h = Kokkos::create_mirror(bckp);
  }

  template <>
  Fields<Dim3, PICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      bckp { "bckp", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS } {
    PLOGD << "Allocated field arrays.";
    em_h   = Kokkos::create_mirror(em);
    cur_h  = Kokkos::create_mirror(cur);
    buff_h = Kokkos::create_mirror(buff);
    bckp_h = Kokkos::create_mirror(bckp);
  }

  template <Dimension D, SimulationEngine S>
  void Fields<D, S>::SynchronizeHostDevice() {
    Kokkos::deep_copy(em_h, em);
    em_h_content = em_content;
    Kokkos::deep_copy(cur_h, cur);
    cur_h_content = cur_content;
    Kokkos::deep_copy(buff_h, buff);
    buff_h_content = buff_content;
    Kokkos::deep_copy(bckp_h, bckp);
    bckp_h_content = bckp_content;
  }

#elif defined(GRPIC_ENGINE)
  // * * * * * * * * * * * * * * * * * * * *
  // GRPIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Fields<Dim2, TypeGRPIC>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      aux { "AUX", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      em0 { "EM0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      aphi { "APHI", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS } {
    PLOGD << "Allocated field arrays.";
    em_h   = Kokkos::create_mirror(em);
    cur_h  = Kokkos::create_mirror(cur);
    buff_h = Kokkos::create_mirror(buff);
    aphi_h = Kokkos::create_mirror(aphi);
  }

  template <>
  Fields<Dim3, TypeGRPIC>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      aux { "AUX", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      em0 { "EM0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      aphi { "APHI", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS } {
    PLOGD << "Allocated field arrays.";
    em_h   = Kokkos::create_mirror(em);
    cur_h  = Kokkos::create_mirror(cur);
    buff_h = Kokkos::create_mirror(buff);
    aphi_h = Kokkos::create_mirror(aphi);
  }

  template <Dimension D, SimulationEngine S>
  void Fields<D, S>::SynchronizeHostDevice() {
    Kokkos::deep_copy(em_h, em);
    Kokkos::deep_copy(cur_h, cur);
    Kokkos::deep_copy(buff_h, buff);
    Kokkos::deep_copy(bckp_h, bckp);
    Kokkos::deep_copy(aphi_h, aphi);
  }

#endif

}    // namespace ntt

#ifdef PIC_ENGINE
template struct ntt::Fields<ntt::Dim1, ntt::PICEngine>;
template struct ntt::Fields<ntt::Dim2, ntt::PICEngine>;
template struct ntt::Fields<ntt::Dim3, ntt::PICEngine>;
#elif defined(GRPIC_ENGINE)
template struct ntt::Fields<ntt::Dim2, ntt::SimulationEngine::GRPIC>;
template struct ntt::Fields<ntt::Dim3, ntt::SimulationEngine::GRPIC>;
#endif