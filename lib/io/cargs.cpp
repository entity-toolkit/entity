#include "cargs.h"

#include <vector>
#include <string>
#include <string_view>
#include <cassert>
#include <algorithm>

namespace ntt {
  namespace io {
    void CommandLineArguments::readCommandLineArguments(int argc, char *argv[]) {
      assert(!_initialized && "# Error: command line arguments already parsed.");
      for (int i {1}; i < argc; ++i)
        this->_args.push_back(std::string_view(argv[i]));
      _initialized = true;
    }
    std::string_view CommandLineArguments::getArgument (std::string_view key, std::string_view def) {
      assert(_initialized && "# Error: command line arguments have not been parsed.");
      std::vector<std::string_view>::const_iterator itr;
      itr = std::find(this->_args.begin(), this->_args.end(), key);
      if (itr != this->_args.end() && ++itr != this->_args.end()){
        return *itr;
      }
      return def;
    }
    std::string_view CommandLineArguments::getArgument (std::string_view key) {
      assert(this->isSpecified(key) && "# Error: unspecified key in command line args.");
      return this->getArgument(key, "");
    }
    bool CommandLineArguments::isSpecified(std::string_view key) {
      return std::find(this->_args.begin(), this->_args.end(), key) != this->_args.end();
    }
    CommandLineArguments cl_args;
  }
}
