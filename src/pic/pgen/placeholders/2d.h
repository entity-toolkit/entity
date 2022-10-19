template <>
ProblemGenerator<Dim2, TypePIC>::ProblemGenerator(const SimulationParams&) {}

template <>
void ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&,
                                                     Meshblock<Dim2, TypePIC>&) {}

template <>
void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&,
                                                        Meshblock<Dim2, TypePIC>&) {}

template <>
void ProblemGenerator<Dim2, TypePIC>::UserBCFields(const real_t&,
                                                   const SimulationParams&,
                                                   Meshblock<Dim2, TypePIC>&) {}

template <>
void ProblemGenerator<Dim2, TypePIC>::UserDriveParticles(const real_t&,
                                                         const SimulationParams&,
                                                         Meshblock<Dim2, TypePIC>&) {}
