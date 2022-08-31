#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <string.h>

namespace ntt {
  auto zeropad(const std::string& str, const size_t& num) -> std::string {
    std::string out {str};
    if (num > str.size()) { out.insert(0, num - str.size(), '0'); }
    return out;
  }
} // namespace ntt

#endif