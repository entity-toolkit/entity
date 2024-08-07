/**
 * @file utils/tools.h
 * @brief Helper functions for general use
 * @implements
 *   - tools::TensorProduct<> -> boundaries_t<T>
 *   - tools::decompose1D -> std::vector<std::size_t>
 *   - tools::divideInProportions2D -> std::tuple<unsigned int, unsigned int>
 *   - tools::divideInProportions3D -> std::tuple<unsigned int, unsigned int, unsigned int>
 *   - tools::Decompose -> std::vector<std::vector<std::size_t>>
 * @namespaces:
 *   - tools::
 */

#ifndef UTILS_HELPERS_H
#define UTILS_HELPERS_H

#include "global.h"

#include "utils/error.h"
#include "utils/numeric.h"

#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

namespace tools {

  /**
   * @brief Compute a tensor product of a list of vectors
   * @param list List of vectors
   * @return Tensor product of list
   */
  template <typename T>
  inline auto TensorProduct(const std::vector<std::vector<T>>& list)
    -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> result = { {} };
    for (const auto& sublist : list) {
      std::vector<std::vector<T>> temp;
      for (const auto& element : sublist) {
        for (const auto& r : result) {
          temp.push_back(r);
          temp.back().push_back(element);
        }
      }
      result = temp;
    }
    return result;
  }

  /**
   * @brief Decompose a 1D domain into ndomains domains roughly equally
   * @param ndomains Number of domains
   * @param ncells Number of cells
   */
  inline auto decompose1D(unsigned int ndomains, std::size_t ncells)
    -> std::vector<std::size_t> {
    auto size          = (std::size_t)((double)ncells / (double)ndomains);
    auto ncells_domain = std::vector<std::size_t>(ndomains, size);
    for (std::size_t i { 0 }; i < ncells - size * ndomains; ++i) {
      ncells_domain[i] += 1;
    }
    auto sum = std::accumulate(ncells_domain.begin(),
                               ncells_domain.end(),
                               (std::size_t)0);
    raise::ErrorIf(sum != ncells, "Decomposition error: sum != ncells", HERE);
    raise::ErrorIf(ncells_domain.size() != (std::size_t)ndomains,
                   "Decomposition error: size != ndomains",
                   HERE);
    for (unsigned int d = 0; d < ndomains; ++d) {
      raise::ErrorIf(ncells_domain[d] < 5, "ncells < 5", HERE);
    }
    return ncells_domain;
  }

  /**
   * @brief Distribute a 2D domain into ntot domains with rough proportions s1 and s2
   * @param ntot Number of domains
   * @param s1 Proportion of the first dimension
   * @param s2 Proportion of the second dimension
   */
  inline auto divideInProportions2D(unsigned int ntot, unsigned int s1, unsigned int s2)
    -> std::tuple<unsigned int, unsigned int> {
    auto n1 = (unsigned int)(std::sqrt((double)ntot * (double)s1 / (double)s2));
    if (n1 == 0) {
      return { 1, ntot };
    } else if (n1 > ntot) {
      return { ntot, 1 };
    } else {
      while (ntot % n1 != 0) {
        n1++;
        raise::ErrorIf(n1 > ntot, "Decomposition2D error: n1 > ntot", HERE);
      }
      return { n1, ntot / n1 };
    }
  }

  /**
   * @brief Distribute a 3D domain into ntot domains with rough proportions s1, s2 and s3
   * @param ntot Number of domains
   * @param s1 Proportion of the first dimension
   * @param s2 Proportion of the second dimension
   * @param s3 Proportion of the third dimension
   */
  inline auto divideInProportions3D(unsigned int ntot,
                                    unsigned int s1,
                                    unsigned int s2,
                                    unsigned int s3)
    -> std::tuple<unsigned int, unsigned int, unsigned int> {
    auto n1 = (unsigned int)(std::cbrt(
      (double)ntot * (double)(SQR(s1)) / (double)(s2 * s3)));
    if (n1 > ntot) {
      return { ntot, 1, 1 };
    } else {
      if (n1 == 0) {
        n1 = 1;
      }
      while (ntot % n1 != 0) {
        n1++;
        raise::ErrorIf(n1 > ntot, "Decomposition3D error: n1 > ntot", HERE);
      }
      auto [n2, n3] = divideInProportions2D(ntot / n1, s2, s3);
      return { n1, n2, n3 };
    }
  }

  /**
   * @brief Decompose a domain into ndomains domains
   * @param ndomains Number of domains
   * @param ncells Number of cells in each dimension
   * @param decomposition Number of domains in each dimension
   *
   * @return A vector of vectors containing the number of cells in each domain
   * in each dimension
   *
   * @note If decomposition has -1, it will be calculated automatically
   */
  inline auto Decompose(unsigned int                    ndomains,
                        const std::vector<std::size_t>& ncells,
                        const std::vector<int>&         decomposition)
    -> std::vector<std::vector<std::size_t>> {
    const auto dimension = ncells.size();
    raise::ErrorIf(dimension != decomposition.size(),
                   "Decomposition error: dimension != decomposition.size",
                   HERE);
    if (dimension == 1) {
      /* 1D ----------------------------------------------------------------- */
      return { decompose1D(ndomains, ncells[0]) };
    } else if (dimension == 2) {
      /* 2D ----------------------------------------------------------------- */
      unsigned int n1 { 0 }, n2 { 0 };
      if (decomposition[0] > 0 && decomposition[1] > 0) {
        n1 = decomposition[0];
        n2 = decomposition[1];
      } else if (decomposition[0] > 0 && decomposition[1] < 0) {
        n1 = decomposition[0];
        raise::ErrorIf(ndomains % n1 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n2 = ndomains / n1;
      } else if (decomposition[0] < 0 && decomposition[1] > 0) {
        n2 = decomposition[1];
        raise::ErrorIf(ndomains % n2 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n1 = ndomains / n2;
        return { decompose1D(n1, ncells[0]), decompose1D(n2, ncells[1]) };
      } else if (decomposition[0] < 0 && decomposition[1] < 0) {
        std::tie(n1, n2) = divideInProportions2D(ndomains, ncells[0], ncells[1]);
      } else {
        raise::Error("Decomposition error: invalid decomposition", HERE);
      }
      raise::ErrorIf(n1 * n2 != ndomains,
                     "Decomposition error: n1 * n2 != ndomains",
                     HERE);
      return { decompose1D(n1, ncells[0]), decompose1D(n2, ncells[1]) };
    } else {
      /* 3D ----------------------------------------------------------------- */
      unsigned int n1 { 0 }, n2 { 0 }, n3 { 0 };
      if (decomposition[0] > 0 && decomposition[1] > 0 && decomposition[2] > 0) {
        n1 = decomposition[0];
        n2 = decomposition[1];
        n3 = decomposition[2];
      } else if (decomposition[0] < 0 && decomposition[1] > 0 &&
                 decomposition[2] > 0) {
        n2 = decomposition[1];
        n3 = decomposition[2];
        raise::ErrorIf(ndomains % (n2 * n3) != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n1 = ndomains / (n2 * n3);
      } else if (decomposition[0] > 0 && decomposition[1] < 0 &&
                 decomposition[2] > 0) {
        n1 = decomposition[0];
        n3 = decomposition[2];
        raise::ErrorIf(ndomains % (n1 * n3) != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n2 = ndomains / (n1 * n3);
      } else if (decomposition[0] > 0 && decomposition[1] > 0 &&
                 decomposition[2] < 0) {
        n1 = decomposition[0];
        n2 = decomposition[1];
        raise::ErrorIf(ndomains % (n1 * n2) != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        n3 = ndomains / (n1 * n2);
      } else if (decomposition[0] < 0 && decomposition[1] < 0 &&
                 decomposition[2] > 0) {
        n3 = decomposition[2];
        raise::ErrorIf(ndomains % n3 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        std::tie(n1,
                 n2) = divideInProportions2D(ndomains / n3, ncells[0], ncells[1]);
      } else if (decomposition[0] < 0 && decomposition[1] > 0 &&
                 decomposition[2] < 0) {
        n2 = decomposition[1];
        raise::ErrorIf(ndomains % n2 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        std::tie(n1,
                 n3) = divideInProportions2D(ndomains / n2, ncells[0], ncells[2]);
      } else if (decomposition[0] > 0 && decomposition[1] < 0 &&
                 decomposition[2] < 0) {
        n1 = decomposition[0];
        raise::ErrorIf(ndomains % n1 != 0,
                       "Decomposition error: does not divide evenly",
                       HERE);
        std::tie(n2, 
                 n3) = divideInProportions2D(ndomains / n1, ncells[1], ncells[2]);
      } else if (decomposition[0] < 0 && decomposition[1] < 0 &&
                 decomposition[2] < 0) {
        std::tie(n1, n2, n3) = divideInProportions3D(ndomains,
                                                     ncells[0],
                                                     ncells[1],
                                                     ncells[2]);
      }
      raise::ErrorIf(n1 * n2 * n3 != ndomains,
                     "Decomposition error: n1 * n2 * n3 != ndomains",
                     HERE);
      return { decompose1D(n1, ncells[0]),
               decompose1D(n2, ncells[1]),
               decompose1D(n3, ncells[2]) };
    }
  }

} // namespace tools

#endif // UTILS_HELPERS_H
