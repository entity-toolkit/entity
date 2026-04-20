#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "archetypes/utils.h"

#include "archetypes/particle_injector.h"
#include "archetypes/traits.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

namespace user {
    using namespace ntt;

    //let us define a magnetic field
    template <Dimension D>
    // we define a dipole
    struct DipoleField{
        // this is a function for the radius from origin
        Inline auto radius(const coord_t<D>& x ) const -> real_t{
            if constexpr (D==Dim::_3D){
                return math::sqrt(SQR(x[0])+SQR(x[1])+SQR(x[2]));
            } else {
                return math::sqrt(SQR(x[0])+SQR(x[1]));
            }
            
        }
        Inline auto bx1(const coord_t<D>& x) const -> real_t{
            return THREE*x[0]*x[1]/math::pow(radius(x), 5);
        }
        Inline auto bx2(const coord_t<D>& x) const -> real_t{
            const auto r = radius(x);
            return (THREE*x[1]*x[1]-SQR(r))/math::pow(r, 5);
        }
        Inline auto bx3(const coord_t<D>& x) const -> real_t{
            if constexpr (D == Dim::_3D) {
                return THREE*x[2]*x[1]/math::pow(radius(x), 5);
            } else {
                return ZERO;
            }
            
        }
    };

    template <SimEngine::type S, class M>
    struct PGen : public arch::ProblemGenerator<S, M> {
        static constexpr auto engines {
            arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value
        };
        static constexpr auto metrics {
            arch::traits::pgen::compatible_with<Metric::Minkowski>::value
        };
        static constexpr auto dimensions {
            arch::traits::pgen::compatible_with<Dim::_2D, Dim::_3D>::value
        };
        // let us now define an init_flds
        //DipoleField<M::Dim> init_flds;

        //Now let us define this dipole as a sort of external field
        inline auto ExternalFields(simtime_t, spidx_t, const Domain<S,M>&) const 
            -> std::pair<bool, DipoleField<M::Dim>> {
            return  {true, DipoleField<M::Dim> {}};  // first part of this is whether external fields need to be applied
            // second return is the field structure itself
            // in theory this can be a function of time, or species index
        }

        const Metadomain<S, M>& metadomain;

        inline PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
            : arch::ProblemGenerator<S, M> {p} 
            , metadomain {metadomain}{}

        inline void InitPrtls(Domain<S, M>& domain) {
            arch::InjectUniformMaxwellian(this->params,
                                            domain, 
                                            ONE,  // number density in ppc
                                            1e-3, // this is the temperature
                                            {1u, 2u},  // initialize two particles
                                            { {ONE, ZERO, ZERO}, {ONE, ZERO, ZERO}}, // velocity
                                            false, // do not use particle weights
                                            {{-2.0, -1.5}, Range::All, Range::All}); // adding this last ine designates the strip where there will be plasma
                                            // if this last line were not there, I would inject plasma in the whole domain 
            //std::map<std::string, std::vector<real_t>> particles {
            //    {"x1", {} },
            //    {"x2", {} },
            //    {"x3", {} }, 
            //    {"ux1", {} },
            //    {"ux2", {} },
            //    {"ux3", {} }
            //};
            //const auto npart = 1000u;
            //const auto ymin  = metadomain.mesh().extent(in::x2).first; // pick min y
            //const auto ymax  = metadomain.mesh().extent(in::x2).second; // pick max y
            //const auto dy    = (ymax - ymin) / static_cast<real_t>(npart - 1u); // define y spacing
            //for (auto p { 0u }; p < npart; ++p) {
            //    particles["x1"].push_back(-1.5); // we stick them at x1 = -1.5
            //    particles["x2"].push_back(ymin + p * dy); // here we are evenly spacing the particles in y
            //    particles["x3"].push_back(0.0);
            //    particles["ux1"].push_back(1.0); // define them as moving right
            //    particles["ux2"].push_back(0.0);
            //    particles["ux3"].push_back(0.0);
            //}   
            //arch::InjectGlobally<S, M>(metadomain, domain, 1u, particles);
            // third entry in this function is the species
        }
    };
}

#endif