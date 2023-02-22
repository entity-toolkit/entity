#include "fields.h"

#include "wrapper.h"

#include <plog/Log.h>

#include <vector>

namespace ntt {
  void AssertEmptyContent_(const std::vector<Content>& content) {
    for (auto c : content) {
      NTTHostErrorIf(c != Content::empty, "Content is not empty.");
    }
  }

  void AssertContent_(const std::vector<Content>& content, const std::vector<Content>& target) {
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
  void ImposeContent(Content& content, const Content& target) {
    content = target;
  }

  void ImposeEmptyContent(std::vector<Content>& content) {
    for (auto& c : content) {
      c = Content::empty;
    }
  }
  void ImposeEmptyContent(Content& content) {
    content = Content::empty;
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
    NTTLog();
  }

  template <>
  Fields<Dim2, PICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      bckp { "bckp", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS } {
    NTTLog();
  }

  template <>
  Fields<Dim3, PICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      bckp { "bckp", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS } {
    NTTLog();
  }

#elif defined(GRPIC_ENGINE)
  // * * * * * * * * * * * * * * * * * * * *
  // GRPIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Fields<Dim2, GRPICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      aux { "AUX", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      em0 { "EM0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS },
      aphi { "APHI", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS } {
    NTTLog();
  }

  template <>
  Fields<Dim3, GRPICEngine>::Fields(resolution_t res)
    : em { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      cur { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      buff { "J0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      aux { "AUX", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      em0 { "EM0", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS },
      aphi { "APHI", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS } {
    NTTLog();
  }

#endif

}    // namespace ntt

/**
 * Engine specific instantiations
 */
#if defined(PIC_ENGINE)
template struct ntt::Fields<ntt::Dim1, ntt::PICEngine>;
template struct ntt::Fields<ntt::Dim2, ntt::PICEngine>;
template struct ntt::Fields<ntt::Dim3, ntt::PICEngine>;
#elif defined(GRPIC_ENGINE)
template struct ntt::Fields<ntt::Dim2, ntt::GRPICEngine>;
template struct ntt::Fields<ntt::Dim3, ntt::GRPICEngine>;
#endif