#include "enums.h"

#include <stdexcept>
#include <string>

#include <type_traits>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

using enum_str_t = const std::vector<std::string>;

template <typename T>
void checkEnum(const enum_str_t& all) {
  for (const auto& c : T::variants) {
    errorIf(not T::contains(T(c).to_string()), "Enum does not contain " + c);
  }
  for (const auto& c : all) {
    errorIf(not T::contains(c.c_str()), "Enum does not contain " + std::string(c));
    errorIf(T::pick(c.c_str()) == T::INVALID,
            "Enum::pick(" + std::string(c) + ") == Enum::INVALID");
  }
  errorIf(all.size() != T::total, "Enum::total is incorrect");
}

auto main() -> int {
  using namespace ntt;
  static_assert(em::ex1 == 0);
  static_assert(em::ex2 == 1);
  static_assert(em::ex3 == 2);

  static_assert(em::dx1 == 0);
  static_assert(em::dx2 == 1);
  static_assert(em::dx3 == 2);

  static_assert(em::bx1 == 3);
  static_assert(em::bx2 == 4);
  static_assert(em::bx3 == 5);

  static_assert(em::hx1 == 3);
  static_assert(em::hx2 == 4);
  static_assert(em::hx3 == 5);

  static_assert(cur::jx1 == 0);
  static_assert(cur::jx2 == 1);
  static_assert(cur::jx3 == 2);

  static_assert(ParticleTag::dead == 0);
  static_assert(ParticleTag::alive == 1);

  static_assert(std::is_convertible_v<ParticleTag, short>);

  using enum_str_t = const std::vector<std::string>;

  enum_str_t all_coords  = { "cart", "sph", "qsph" };
  enum_str_t all_metrics = { "minkowski",   "spherical",    "qspherical",
                             "kerr_schild", "qkerr_schild", "kerr_schild_0" };
  enum_str_t all_simulation_engines = { "srpic", "grpic" };
  enum_str_t all_particle_bcs = { "periodic", "absorb",  "atmosphere", "custom",
                                  "reflect",  "horizon", "axis",       "send" };
  enum_str_t all_fields_bcs   = { "periodic", "absorb",  "atmosphere",
                                  "custom",   "horizon", "axis" };
  enum_str_t all_comm_bcs     = { "physical", "comm" };
  enum_str_t all_particle_pushers = { "boris",   "vay",    "boris,gca",
                                      "vay,gca", "photon", "none" };
  enum_str_t all_coolings         = { "synchrotron", "none" };

  checkEnum<Coord>(all_coords);
  checkEnum<Metric>(all_metrics);
  checkEnum<SimEngine>(all_simulation_engines);
  checkEnum<PrtlBC>(all_particle_bcs);
  checkEnum<FldsBC>(all_fields_bcs);
  checkEnum<CommBC>(all_comm_bcs);
  checkEnum<PrtlPusher>(all_particle_pushers);
  checkEnum<Cooling>(all_coolings);

  // // check Coord::
  // for (const auto& c : Coord::variants) {
  //   errorIf(not Coord::contains(c), "Coord does not contain " + c);
  // }
  // for (const auto& c : all_coords) {
  //   errorIf(not Coord::contains(c.c_str()),
  //           "Coord does not contain " + std::string(c));
  //   errorIf(Coord::pick(c.c_str()) == Coord::INVALID,
  //           "Coord::pick(" + std::string(c) + ") == Coord::INVALID");
  // }
  // errorIf(all_coords.size() != Coord::total, "Coord::total is incorrect");

  // // check Metric::
  // for (const auto& m : Metric::variants) {
  //   auto m_str = std::string(m);
  //   errorIf(not Metric::contains(m_str), "Metric does not contain " + m_str);
  // }
  // for (const auto& m : all_metrics) {
  //   errorIf(not Metric::contains(m), "Metric does not contain " + m);
  //   errorIf(Metric::pick(m) == Metric::INVALID,
  //           "Metric::pick(" + m + ") == Metric::INVALID");
  // }
  // errorIf(all_metrics.size() != Metric::total, "Metric::total is incorrect");

  // // check SimulationEngine::
  // for (const auto& s : SimulationEngine::variants) {
  //   auto s_str = std::string(s);
  //   errorIf(not SimulationEngine::contains(s_str),
  //           "SimulationEngine does not contain " + s_str);
  // }
  // for (const auto& s : all_simulation_engines) {
  //   errorIf(not SimulationEngine::contains(s),
  //           "SimulationEngine does not contain " + s);
  //   errorIf(SimulationEngine::pick(s) == SimulationEngine::INVALID,
  //           "SimulationEngine::pick(" + s + ") == SimulationEngine::INVALID");
  // }
  // errorIf(all_simulation_engines.size() != SimulationEngine::total,
  //         "SimulationEngine::total is incorrect");

  // // check ParticleBC::
  // for (const auto& p : ParticleBC::variants) {
  //   auto p_str = std::string(p);
  //   errorIf(not ParticleBC::contains(p_str),
  //           "ParticleBC does not contain " + p_str);
  // }
  // for (const auto& p : all_particle_bcs) {
  //   errorIf(not ParticleBC::contains(p), "ParticleBC does not contain " + p);
  //   errorIf(ParticleBC::pick(p) == ParticleBC::INVALID,
  //           "ParticleBC::pick(" + p + ") == ParticleBC::INVALID");
  // }
  // errorIf(all_particle_bcs.size() != ParticleBC::total,
  //         "ParticleBC::total is incorrect");

  // // check FieldsBC::
  // for (const auto& f : FieldsBC::variants) {
  //   auto f_str = std::string(f);
  //   errorIf(not FieldsBC::contains(f_str), "FieldsBC does not contain " + f_str);
  // }
  // for (const auto& f : all_fields_bcs) {
  //   errorIf(not FieldsBC::contains(f), "FieldsBC does not contain " + f);
  //   errorIf(FieldsBC::pick(f) == FieldsBC::INVALID,
  //           "FieldsBC::pick(" + f + ") == FieldsBC::INVALID");
  // }
  // errorIf(all_fields_bcs.size() != FieldsBC::total, "FieldsBC::total is incorrect");

  // // check CommBC::
  // for (const auto& c : CommBC::variants) {
  //   auto c_str = std::string(c);
  //   errorIf(not CommBC::contains(c_str), "CommBC does not contain " + c_str);
  // }
  // for (const auto& c : all_comm_bcs) {
  //   errorIf(not CommBC::contains(c), "CommBC does not contain " + c);
  //   errorIf(CommBC::pick(c) == CommBC::INVALID,
  //           "CommBC::pick(" + c + ") == CommBC::INVALID");
  // }
  // errorIf(all_comm_bcs.size() != CommBC::total, "CommBC::total is incorrect");

  // // check ParticlePusher::
  // for (const auto& p : ParticlePusher::variants) {
  //   auto p_str = std::string(p);
  //   errorIf(not ParticlePusher::contains(p_str),
  //           "ParticlePusher does not contain " + p_str);
  // }
  // for (const auto& p : all_particle_pushers) {
  //   errorIf(not ParticlePusher::contains(p),
  //           "ParticlePusher does not contain " + p);
  //   errorIf(ParticlePusher::pick(p) == ParticlePusher::INVALID,
  //           "ParticlePusher::pick(" + p + ") == ParticlePusher::INVALID");
  // }
  // errorIf(all_particle_pushers.size() != ParticlePusher::total,
  //         "ParticlePusher::total is incorrect");

  // // check Cooling::
  // for (const auto& c : Cooling::variants) {
  //   auto c_str = std::string(c);
  //   errorIf(not Cooling::contains(c_str), "Cooling does not contain " + c_str);
  // }
  // for (const auto& c : all_coolings) {
  //   errorIf(not Cooling::contains(c), "Cooling does not contain " + c);
  //   errorIf(Cooling::pick(c) == Cooling::INVALID,
  //           "Cooling::pick(" + c + ") == Cooling::INVALID");
  // }
  // errorIf(all_coolings.size() != Cooling::total, "Cooling::total is incorrect");

  return 0;
}