template <>
ProblemGenerator<Dim1, TypePIC>::ProblemGenerator(const SimulationParams&) {}

template <>
void ProblemGenerator<Dim1, TypePIC>::UserInitFields(const SimulationParams&,
                                                     Meshblock<Dim1, TypePIC>&) {}

template <>
void ProblemGenerator<Dim1, TypePIC>::UserInitParticles(const SimulationParams&,
                                                        Meshblock<Dim1, TypePIC>&) {}

template <>
void ProblemGenerator<Dim1, TypePIC>::UserBCFields(const real_t&,
                                                   const SimulationParams&,
                                                   Meshblock<Dim1, TypePIC>&) {}

template <>
Inline auto
ProblemGenerator<Dim1, TypePIC>::UserTargetField_br_hat(const Meshblock<Dim1, TypePIC>&,
                                                        const coord_t<Dim1>&) const -> real_t {
  return ZERO;
}

template <>
void ProblemGenerator<Dim1, TypePIC>::UserDriveParticles(const real_t&,
                                                         const SimulationParams&,
                                                         Meshblock<Dim1, TypePIC>&) {}
