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
void ProblemGenerator<Dim1, TypePIC>::UserDriveParticles(const real_t&,
                                                         const SimulationParams&,
                                                         Meshblock<Dim1, TypePIC>&) {}
