#ifndef PIC_PICSIM_H
#define PIC_PICSIM_H

#include "global.h"
#include "timer.h"
#include "sim.h"
#include "fields.h"
#include "particles.h"

#include <vector>

namespace ntt {

class PICSimulation : public Simulation {
protected:
  std::vector<particles::ParticleSpecies> m_species;

  // dedicated timers
  timer::Timer timer_em{"FIELD SOLVER"};
  timer::Timer timer_deposit{"DEPOSIT"};
  timer::Timer timer_pusher{"PUSHER"};
public:
  PICSimulation(Dimension dim, CoordinateSystem coord_sys) : Simulation{dim, coord_sys, PIC_SIM} {};
  ~PICSimulation() = default;

  void parseInput(int argc, char *argv[]) override;

  void initialize() override;
  void verify() override;
  void mainloop() override;

  void stepForward(const real_t &time);
  virtual void faradayHalfsubstep(const real_t &) {};
  virtual void ampereSubstep(const real_t &) {};
  virtual void addCurrentsSubstep(const real_t &) {};
  virtual void depositSubstep(const real_t &) {};
  void particlePushSubstep(const real_t &);

  [[nodiscard]] auto getSizeInBytes() -> std::size_t;

  void printDetails(std::ostream &os) override;
};

class PICSimulation1D : public PICSimulation {
protected:
  fields::OneDField<real_t> ex1, ex2, ex3;
  fields::OneDField<real_t> bx1, bx2, bx3;
  fields::OneDField<real_t> jx1, jx2, jx3;

public:
  PICSimulation1D() : PICSimulation{ONE_D, CARTESIAN_COORD} {};
  ~PICSimulation1D() = default;
  void initialize() override;
  void verify() override;

  void faradayHalfsubstep(const real_t &time) override;
  void depositSubstep(const real_t &time) override;
  void ampereSubstep(const real_t &time) override;
  void addCurrentsSubstep(const real_t &time) override;

  [[nodiscard]] auto getSizeInBytes() -> std::size_t;
};

class PICSimulation2D : public PICSimulation {
protected:
  fields::TwoDField<real_t> ex1, ex2, ex3;
  fields::TwoDField<real_t> bx1, bx2, bx3;
  fields::TwoDField<real_t> jx1, jx2, jx3;

public:
  PICSimulation2D(CoordinateSystem coord_sys) : PICSimulation{TWO_D, coord_sys} {};
  ~PICSimulation2D() = default;
  void initialize() override;
  void verify() override;

  void faradayHalfsubstep(const real_t &time) override;
  void depositSubstep(const real_t &time) override;
  void ampereSubstep(const real_t &time) override;
  void addCurrentsSubstep(const real_t &time) override;

  [[nodiscard]] auto getSizeInBytes() -> std::size_t;
};

class PICSimulation3D : public PICSimulation {
protected:
  fields::ThreeDField<real_t> ex1, ex2, ex3;
  fields::ThreeDField<real_t> bx1, bx2, bx3;
  fields::ThreeDField<real_t> jx1, jx2, jx3;

public:
  PICSimulation3D(CoordinateSystem coord_sys) : PICSimulation{THREE_D, coord_sys} {};
  ~PICSimulation3D() = default;
  void initialize() override;
  void verify() override;

  void faradayHalfsubstep(const real_t &time) override;
  void depositSubstep(const real_t &time) override;
  void ampereSubstep(const real_t &time) override;
  void addCurrentsSubstep(const real_t &time) override;

  [[nodiscard]] auto getSizeInBytes() -> std::size_t;
};

} // namespace ntt

#endif
