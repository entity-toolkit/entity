#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include "wrapper.h"

#include <regex>
#include <string>
#include <vector>

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

  template <class KeyViewType>
  struct BinTag {
    BinTag() = default;
    template <class ViewType>
    Inline auto bin(ViewType& keys, const int& i) const -> int {
      return static_cast<int>(keys(i));
    }
    Inline auto max_bins() const -> int {
      return 101;
    }
    template <class ViewType, typename iT1, typename iT2>
    Inline auto operator()(ViewType& keys, iT1& i1, iT2& i2) const -> bool {
      return false;
    }
  };

  /**
   * @brief Check if a string is a valid option
   * @param option Option to check
   * @param valid_options Vector of valid options
   * @return True if option is valid, false otherwise
   */
  template <typename T>
  inline void TestValidOption(const T& option, const std::vector<T>& valid_options) {
    for (const auto& valid_option : valid_options) {
      if (option == valid_option) {
        return;
      }
    }
    NTTHostError("Invalid option: " + option);
  }

  /**
   * @brief Split a string into a vector of strings
   * @param str String to split
   * @param delim Delimiter
   * @return Vector of strings
   */
  inline auto SplitString(const std::string str, const std::string delimiter)
    -> std::vector<std::string> {
    std::regex regexz(delimiter);
    return { std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
             std::sregex_token_iterator() };
  }
}    // namespace ntt

#endif