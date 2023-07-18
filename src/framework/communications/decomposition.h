#ifndef FRAMEWORK_COMM_DECOMPOSITION_H
#define FRAMEWORK_COMM_DECOMPOSITION_H

#include "wrapper.h"

#include <fmt/core.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace ntt {
  namespace {
    /**
     * @brief Decompose a 1D domain into ndomains domains roughly equally
     * @param ndomains Number of domains
     * @param ncells Number of cells
     */
    auto decompose1D(const unsigned int& ndomains, const unsigned int& ncells)
      -> std::vector<unsigned int> {
      auto size          = (unsigned int)((double)ncells / (double)ndomains);
      auto ncells_domain = std::vector<unsigned int>(ndomains, size);
      for (std::size_t i { 0 }; i < ncells - size * ndomains; ++i) {
        ncells_domain[i] += 1;
      }
      auto sum = std::accumulate(ncells_domain.begin(), ncells_domain.end(), 0);
      NTTHostErrorIf(sum != (int)ncells, "Decomposition error: sum != ncells");
      NTTHostErrorIf(ncells_domain.size() != ndomains, "Decomposition error: size != ndomains");
      return ncells_domain;
    }

    /**
     * @brief Distribute a 2D domain into ntot domains with rough proportions s1 and s2
     * @param ntot Number of domains
     * @param s1 Proportion of the first dimension
     * @param s2 Proportion of the second dimension
     */
    auto divideInProportions2D(const unsigned int& ntot,
                               const unsigned int& s1,
                               const unsigned int& s2)
      -> std::tuple<unsigned int, unsigned int> {
      auto n1 = (unsigned int)(std::sqrt((double)ntot * (double)s1 / (double)s2));
      if (n1 == 0) {
        return { 1, ntot };
      } else if (n1 > ntot) {
        return { ntot, 1 };
      } else {
        while (ntot % n1 != 0) {
          n1++;
          NTTHostErrorIf(n1 > ntot, "Decomposition2D error: n1 > ntot");
        }
        auto n2 = ntot / n1;
        return { n1, n2 };
      }
    }

    /**
     * @brief Distribute a 3D domain into ntot domains with rough proportions s1, s2 and s3
     * @param ntot Number of domains
     * @param s1 Proportion of the first dimension
     * @param s2 Proportion of the second dimension
     * @param s3 Proportion of the third dimension
     */
    auto divideInProportions3D(const unsigned int& ntot,
                               const unsigned int& s1,
                               const unsigned int& s2,
                               const unsigned int& s3)
      -> std::tuple<unsigned int, unsigned int, unsigned int> {
      auto n1
        = (unsigned int)(std::cbrt((double)ntot * (double)(SQR(s1)) / (double)(s2 * s3)));
      if (n1 > ntot) {
        return { ntot, 1, 1 };
      } else {
        if (n1 == 0) {
          n1 = 1;
        }
        while (ntot % n1 != 0) {
          n1++;
          NTTHostErrorIf(n1 > ntot, "Decomposition3D error: n1 > ntot");
        }
        auto n23      = ntot / n1;
        auto [n2, n3] = divideInProportions2D(n23, s2, s3);
        return { n1, n2, n3 };
      }
    }
  }    // anonymous namespace

  /**
   * @brief Decompose a domain into ndomains domains
   * @param ndomains Number of domains
   * @param ncells Number of cells in each dimension
   * @param decomposition Number of domains in each dimension
   *
   * @return A vector of vectors containing the number of cells in each domain in each
   * dimension
   *
   * @note If decomposition is empty, the domain is decomposed automatically
   */
  inline auto Decompose(const unsigned int&              ndomains,
                        const std::vector<unsigned int>& ncells,
                        const std::vector<unsigned int>& decomposition = {})
    -> std::vector<std::vector<unsigned int>> {
    const auto dimension      = ncells.size();
    const auto auto_decompose = decomposition.empty();
    // sanity check
    NTTHostErrorIf(dimension < 1 || dimension > 3,
                   "Decomposition error: dimension < 1 || dimension > 3");
    NTTHostErrorIf(!auto_decompose && decomposition.size() != dimension,
                   "Decomposition error: decomposition.size() != dimension");
    NTTHostErrorIf(ndomains == 0, "Decomposition error: ndomains == 0");
    NTTHostErrorIf(
      !auto_decompose
        && std::accumulate(decomposition.begin(), decomposition.end(), 1, std::multiplies<>())
             != (int)ndomains,
      fmt::format(
        "Decomposition error: sum(decomposition) != ndomains {} {}",
        std::accumulate(decomposition.begin(), decomposition.end(), 1, std::multiplies<>()),
        (int)ndomains));
    if (dimension == 1) {
      return { decompose1D(ndomains, ncells[0]) };
    } else if (dimension == 2) {
      if (auto_decompose) {
        auto [n1, n2] = divideInProportions2D(ndomains, ncells[0], ncells[1]);
        return { decompose1D(n1, ncells[0]), decompose1D(n2, ncells[1]) };
      } else {
        return { decompose1D(decomposition[0], ncells[0]),
                 decompose1D(decomposition[1], ncells[1]) };
      }
    } else {
      if (auto_decompose) {
        auto [n1, n2, n3] = divideInProportions3D(ndomains, ncells[0], ncells[1], ncells[2]);
        return { decompose1D(n1, ncells[0]),
                 decompose1D(n2, ncells[1]),
                 decompose1D(n3, ncells[2]) };
      } else {
        return { decompose1D(decomposition[0], ncells[0]),
                 decompose1D(decomposition[1], ncells[1]),
                 decompose1D(decomposition[2], ncells[2]) };
      }
    }
  }
}    // namespace ntt

#endif    // FRAMEWORK_COMM_DECOMPOSITION_H