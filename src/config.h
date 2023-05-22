#ifndef CONFIG_H
#define CONFIG_H

#include <fmt/core.h>

#include <stdexcept>

#define STRINGIZE(x)        STRINGIZE_DETAIL(x)
#define STRINGIZE_DETAIL(x) #x
#define LINE_STRING         STRINGIZE(__LINE__)

#define NTTLog()                                                                              \
  { PLOGV_(ntt::LogFile); }

#define NTTWarn(msg)                                                                          \
  { PLOGW_(ntt::LogFile) << msg; }

#define NTTFatal()                                                                            \
  { PLOGF_(ntt::LogFile); }

#define NTTHostError(msg)                                                                     \
  {                                                                                           \
    auto err                                                                                  \
      = fmt::format("# ERROR: {}  : filename: {} : line: {}", msg, __FILE__, LINE_STRING);    \
    NTTFatal();                                                                               \
    throw std::runtime_error(err);                                                            \
  }

#define NTTHostErrorIf(condition, msg)                                                        \
  {                                                                                           \
    if ((condition)) {                                                                        \
      NTTHostError(msg);                                                                      \
    }                                                                                         \
  }

#ifdef ENABLE_GPU
#  define NTTError(msg) ({})
#else
#  define NTTError(msg) NTTHostError(msg)
#endif

// Defining precision-based constants and types
using prtldx_t = float;
#ifdef SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

inline constexpr int N_GHOSTS = 2;

#endif    // CONFIG_H
