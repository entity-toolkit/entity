#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include "wrapper.h"

#include <string>
#include <vector>

#include <type_traits>

namespace ntt {
  template <class KeyViewType>
  struct BinBool {
    BinBool() = default;
    template <class ViewType>
    Inline auto bin(ViewType& keys, const int& i) const -> int {
      return keys(i) ? 1 : 0;
    }
    Inline auto max_bins() const -> int {
      return 2;
    }
    template <class ViewType, typename iT1, typename iT2>
    Inline auto operator()(ViewType& keys, iT1& i1, iT2& i2) const -> bool {
      return false;
    }
  };

  namespace c9 {
    /**
     * @link https://github.com/CommitThis/zip-iterator
     */

    template <typename Iter>
    using select_access_type_for
      = std::conditional_t<std::is_same_v<Iter, std::vector<bool>::iterator>
                             || std::is_same_v<Iter, std::vector<bool>::const_iterator>,
                           typename Iter::value_type,
                           typename Iter::reference>;

    template <typename Iter1, typename Iter2>
    class zip_iterator {
    public:
      using value_type
        = std::pair<select_access_type_for<Iter1>, select_access_type_for<Iter2>>;

      zip_iterator() = delete;

      zip_iterator(Iter1 iter_1_begin, Iter2 iter_2_begin)
        : m_iter_1_begin { iter_1_begin }, m_iter_2_begin { iter_2_begin } {}

      auto operator++() -> zip_iterator& {
        ++m_iter_1_begin;
        ++m_iter_2_begin;
        return *this;
      }

      auto operator++(int) -> zip_iterator {
        auto tmp = *this;
        ++*this;
        return tmp;
      }

      auto operator!=(zip_iterator const& other) {
        return !(*this == other);
      }

      auto operator==(zip_iterator const& other) {
        return m_iter_1_begin == other.m_iter_1_begin
               || m_iter_2_begin == other.m_iter_2_begin;
      }

      auto operator*() -> value_type {
        return value_type { *m_iter_1_begin, *m_iter_2_begin };
      }

    private:
      Iter1 m_iter_1_begin;
      Iter2 m_iter_2_begin;
    };

    template <typename T>
    using select_iterator_for = std::conditional_t<std::is_const_v<std::remove_reference_t<T>>,
                                                   typename std::decay_t<T>::const_iterator,
                                                   typename std::decay_t<T>::iterator>;

    template <typename T, typename U>
    class zipper {
    public:
      using Iter1    = select_iterator_for<T>;
      using Iter2    = select_iterator_for<U>;

      using zip_type = zip_iterator<Iter1, Iter2>;

      template <typename V, typename W>
      zipper(V&& a, W&& b) : m_a { a }, m_b { b } {}

      auto begin() -> zip_type {
        return zip_type { std::begin(m_a), std::begin(m_b) };
      }
      auto end() -> zip_type {
        return zip_type { std::end(m_a), std::end(m_b) };
      }

    private:
      T m_a;
      U m_b;
    };

    template <typename T, typename U>
    auto zip(T&& t, U&& u) {
      return zipper<T, U> { std::forward<T>(t), std::forward<U>(u) };
    }

  }    // namespace c9

}    // namespace ntt

#endif