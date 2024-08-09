#include "utils/cargs.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace cargs {

  void CommandLineArguments::readCommandLineArguments(int argc, char* argv[]) {
    if (_initialized) {
      throw std::runtime_error(
        "# Error: command line arguments already parsed.");
    }
    for (int i { 1 }; i < argc; ++i) {
      this->_args.emplace_back(std::string_view(argv[i]));
    }
    _initialized = true;
  }

  auto CommandLineArguments::getArgument(std::string_view key, std::string_view def)
    -> std::string_view {
    if (!_initialized) {
      throw std::runtime_error(
        "# Error: command line arguments have not been parsed.");
    }
    std::vector<std::string_view>::const_iterator itr;
    itr = std::find(this->_args.begin(), this->_args.end(), key);
    if (itr != this->_args.end() && ++itr != this->_args.end()) {
      return *itr;
    }
    return def;
  }

  auto CommandLineArguments::getArgument(std::string_view key) -> std::string_view {
    if (!this->isSpecified(key)) {
      throw std::runtime_error(
        "# Error: unspecified key in command line args.");
    }
    return this->getArgument(key, "");
  }

  auto CommandLineArguments::isSpecified(std::string_view key) -> bool {
    return std::find(this->_args.begin(), this->_args.end(), key) !=
           this->_args.end();
  }

} // namespace cargs
