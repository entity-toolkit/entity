#include "utils/param_container.h"

#include "enums.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

auto main() -> int {
  using namespace ntt;
  auto p = prm::Parameters();

  const auto d_vec = std::vector<bool> { true, true, false };
  const auto e_vec = std::vector<std::vector<std::string>> {
    { "a" },
    { "b", "c" }
  };
  const auto nonexist_vec = std::vector<int> { 1, 2, 3 };
  const auto flds_bc_vec = std::vector<FldsBC> { FldsBC::AXIS, FldsBC::PERIODIC };
  const auto prtl_bc_vec = boundaries_t<PrtlBC> {
    {PrtlBC::REFLECT, PrtlBC::PERIODIC},
    {PrtlBC::REFLECT,  PrtlBC::REFLECT}
  };

  p.set("a", 1);
  p.set("b", std::string("hello world"));
  p.set("c", 3.14);
  p.set("d", d_vec);
  p.set("e", e_vec);
  p.set("enum1", Coord(Coord::Cart));
  p.set("enum2", flds_bc_vec);
  p.set("enum3", prtl_bc_vec);

  try {
    errorIf(p.get<int>("a") != 1, "Failed to get int");
    errorIf(p.get<std::string>("b") != "hello world", "Failed to get string");
    errorIf(p.get<double>("c") != 3.14, "Failed to get double");
    errorIf(p.get<std::vector<bool>>("d") != d_vec,
            "Failed to get vector of bools");
    errorIf(p.get<std::vector<std::vector<std::string>>>("e") != e_vec,
            "Failed to get vector of vector of strings");
    errorIf(p.get<std::vector<int>>("nonexist", nonexist_vec) != nonexist_vec,
            "Failed to fallback to default");
    errorIf(p.get<Coord>("enum1") != Coord::Cart, "Failed to get Coord::");
    errorIf(p.get<std::vector<FldsBC>>("enum2") != flds_bc_vec,
            "Failed to get std::vector<FldsBC>::");
    errorIf(p.get<boundaries_t<PrtlBC>>("enum3", prtl_bc_vec) != prtl_bc_vec,
            "Failed to get boundaries_t<PrtlBC>::");

    errorIf(p.stringize<int>("a") != "1", "Wrong stringize for int");
    errorIf(p.stringize<std::string>("b") != "hello world",
            "Wrong stringize for string");
    errorIf(p.stringize<double>("c") != "3.14", "Wrong stringize for double");
    errorIf(p.stringize<bool>("d") != "[true, true, false]",
            "Wrong stringize for vector of bools");
    errorIf(p.stringize<std::string>("e") != "[{a}, {b, c}]",
            "Wrong stringize for vector of vector of strings");

    errorIf(p.stringize<Coord>("enum1") != "cart", "Wrong stringize for Coord::");

    errorIf(p.stringize<FldsBC>("enum2") != "[axis, periodic]",
            "Wrong stringize for std::vector<FldsBC>::");
    errorIf(p.stringize<PrtlBC>("enum3") !=
              "[{reflect, periodic}, {reflect, reflect}]",
            "Wrong stringize for boundaries_t<PrtlBC>::");

    p.promiseToDefine("f");
    errorIf(p.promisesFulfilled(), "Promises fulfilled too early");
    p.promiseToDefine("f"); // check duplicate promise
    p.set("f", 42);
    errorIf(!p.isPromised("f"), "Promise not found");
    errorIf(!p.promisesFulfilled(), "Promises not fulfilled");

    errorIf(!p.contains("a"), "Failed to find key a");
    errorIf(p.contains("nonexist"), "Found non-existent key nonexist");
    errorIf(!p.contains("enum3"), "Failed to find key enum3");

  } catch (std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
  return 0;
}