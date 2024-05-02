/**
 * @file utils/cargs.h
 * @brief Class for extracting the command line arguments
 * @implements
 *   - cargs::CommandLineArguments
 * @cpp:
 *   - cargs.cpp
 * @namespaces:
 *   - cargs::
 */

#ifndef GLOBAL_UTILS_CARGS_H
#define GLOBAL_UTILS_CARGS_H

#include <string_view>
#include <vector>

namespace cargs {

  class CommandLineArguments {
  private:
    bool                          _initialized = false;
    std::vector<std::string_view> _args;

  public:
    void readCommandLineArguments(int argc, char* argv[]);
    [[nodiscard]]
    auto getArgument(std::string_view key, std::string_view def)
      -> std::string_view;
    [[nodiscard]]
    auto getArgument(std::string_view key) -> std::string_view;
    auto isSpecified(std::string_view key) -> bool;
  };

} // namespace cargs

#endif // GLOBAL_UTILS_CARGS_H
