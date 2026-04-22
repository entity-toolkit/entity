#include "utils/param_container.h"

#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include <iostream>
#include <string>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  using namespace ntt;
  GlobalInitialize(argc, argv);
  auto p = prm::Parameters();

  const auto d_vec = std::vector<bool> { true, true, false };
  const auto e_vec = std::vector<std::vector<std::string>> {
    { "a" },
    { "b", "c" }
  };
  const auto nonexist_vec = std::vector<int> { 1, 2, 3 };
  const auto flds_bc_vec = std::vector<FldsBC> { FldsBC::AXIS, FldsBC::PERIODIC };
  const auto prtl_bc_vec = boundaries_t<PrtlBC> {
    { PrtlBC::REFLECT, PrtlBC::PERIODIC },
    { PrtlBC::REFLECT,  PrtlBC::REFLECT }
  };

  p.set("a", 1);
  p.set("b", std::string("hello world"));
  p.set("c", 3.14);
  p.set("d", d_vec);
  p.set("e", e_vec);
  p.set("enum1", Coord(Coord::Cartesian));
  p.set("enum2", flds_bc_vec);
  p.set("enum3", prtl_bc_vec);

  try {
    raise::ErrorIf(p.get<int>("a") != 1, "Failed to get int", HERE);
    raise::ErrorIf(p.get<std::string>("b") != "hello world", "Failed to get string", HERE);
    raise::ErrorIf(p.get<double>("c") != 3.14, "Failed to get double", HERE);
    raise::ErrorIf(p.get<std::vector<bool>>("d") != d_vec,
                   "Failed to get vector of bools",
                   HERE);
    raise::ErrorIf(p.get<std::vector<std::vector<std::string>>>("e") != e_vec,
                   "Failed to get vector of vector of strings",
                   HERE);
    raise::ErrorIf(p.get<std::vector<int>>("nonexist", nonexist_vec) != nonexist_vec,
                   "Failed to fallback to default",
                   HERE);
    raise::ErrorIf(p.get<Coord>("enum1") != Coord::Cartesian,
                   "Failed to get Coord::",
                   HERE);
    raise::ErrorIf(p.get<std::vector<FldsBC>>("enum2") != flds_bc_vec,
                   "Failed to get std::vector<FldsBC>::",
                   HERE);
    raise::ErrorIf(p.get<boundaries_t<PrtlBC>>("enum3", prtl_bc_vec) != prtl_bc_vec,
                   "Failed to get boundaries_t<PrtlBC>::",
                   HERE);

    raise::ErrorIf(p.stringize<int>("a") != "1", "Wrong stringize for int", HERE);
    raise::ErrorIf(p.stringize<std::string>("b") != "hello world",
                   "Wrong stringize for string",
                   HERE);
    raise::ErrorIf(p.stringize<double>("c") != "3.14", "Wrong stringize for double", HERE);
    raise::ErrorIf(p.stringize<bool>("d") != "[true, true, false]",
                   "Wrong stringize for vector of bools",
                   HERE);
    raise::ErrorIf(p.stringize<std::string>("e") != "[{a}, {b, c}]",
                   "Wrong stringize for vector of vector of strings",
                   HERE);

    raise::ErrorIf(p.stringize<Coord>("enum1") != "cart",
                   "Wrong stringize for Coord::",
                   HERE);
    raise::ErrorIf(p.stringize<FldsBC>("enum2") != "[axis, periodic]",
                   "Wrong stringize for std::vector<FldsBC>::",
                   HERE);
    raise::ErrorIf(p.stringize<PrtlBC>("enum3") !=
                     "[{reflect, periodic}, {reflect, reflect}]",
                   "Wrong stringize for boundaries_t<PrtlBC>::",
                   HERE);

    p.promiseToDefine("f");
    raise::ErrorIf(p.promisesFulfilled(), "Promises fulfilled too early", HERE);
    p.promiseToDefine("f"); // check duplicate promise
    p.set("f", 42);
    raise::ErrorIf(!p.isPromised("f"), "Promise not found", HERE);
    raise::ErrorIf(!p.promisesFulfilled(), "Promises not fulfilled", HERE);

    raise::ErrorIf(!p.contains("a"), "Failed to find key a", HERE);
    raise::ErrorIf(p.contains("nonexist"), "Found non-existent key nonexist", HERE);
    raise::ErrorIf(!p.contains("enum3"), "Failed to find key enum3", HERE);

  } catch (std::exception& exc) {
    std::cerr << exc.what() << "\n";
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return 0;
}
