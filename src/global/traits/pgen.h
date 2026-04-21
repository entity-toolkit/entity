#ifndef TRAITS_PGEN_H
#define TRAITS_PGEN_H

#include "global.h"

#include "arch/kokkos_aliases.h"

namespace traits::pgen {

  template <auto... Is>
  struct compatible_with {};

  template <class PG>
  concept HasD = requires {
    { PG::D } -> std::convertible_to<Dimension>;
  };

  template <class PG>
  concept HasInitFlds = requires(const PG& pgen) { pgen.init_flds; };

  template <class PG, class D>
  concept HasEmissionPolicy = requires(const PG& pgen,
                                       simtime_t time,
                                       spidx_t   sp,
                                       D&        domain) {
    pgen.EmissionPolicy(time, sp, domain);
  };

  template <class PG, class D>
  concept HasCustomPrtlUpdate = requires(const PG& pgen,
                                         simtime_t time,
                                         spidx_t   sp,
                                         D&        domain) {
    pgen.CustomParticleUpdate(time, sp, domain);
  };

  template <class PG, class D>
  concept HasExternalFields = requires(const PG& pgen,
                                       simtime_t time,
                                       spidx_t   sp,
                                       D&        domain) {
    requires std::same_as<bool, decltype(pgen.ExternalFields(time, sp, domain).first)>;
    pgen.ExternalFields(time, sp, domain).second;
  };

  template <class PG, class D>
  concept HasInitPrtls = requires(PG& pgen, D& domain) {
    { pgen.InitPrtls(domain) } -> std::same_as<void>;
  };

  template <class PG>
  concept HasExtCurrent = requires(const PG& pgen) { pgen.ext_current; };

  template <class PG>
  concept HasAtmFields = requires(const PG& pgen, simtime_t time) {
    pgen.AtmFields(time);
  };

  template <class PG>
  concept HasMatchFields = requires(const PG& pgen, simtime_t time) {
    pgen.MatchFields(time);
  };

  template <class PG>
  concept HasMatchFieldsInX1 = requires(const PG& pgen, simtime_t time) {
    pgen.MatchFieldsInX1(time);
  };

  template <class PG>
  concept HasMatchFieldsInX2 = requires(const PG& pgen, simtime_t time) {
    pgen.MatchFieldsInX2(time);
  };

  template <class PG>
  concept HasMatchFieldsInX3 = requires(const PG& pgen, simtime_t time) {
    pgen.MatchFieldsInX3(time);
  };

  template <class PG>
  concept HasFixFieldsConst = requires(const PG&    pgen,
                                       simtime_t    time,
                                       const bc_in& bc,
                                       ntt::em      comp) {
    {
      pgen.FixFieldsConst(time, bc, comp)
    } -> std::convertible_to<std::pair<real_t, bool>>;
  };

  template <class PG, class D>
  concept HasCustomPostStep = requires(PG& pgen, timestep_t s, simtime_t t, D& domain) {
    { pgen.CustomPostStep(s, t, domain) } -> std::same_as<void>;
  };

  template <class PG, class D>
  concept HasCustomFieldOutput = requires(PG&                  pgen,
                                          const std::string&   name,
                                          ndfield_t<PG::D, 6>& buff,
                                          index_t              idx,
                                          timestep_t           step,
                                          simtime_t            time,
                                          const D&             dom) {
    {
      pgen.CustomFieldOutput(name, buff, idx, step, time, dom)
    } -> std::same_as<void>;
  };

  template <class PG, class D>
  concept HasCustomStatOutput = requires(PG&                pgen,
                                         const std::string& name,
                                         timestep_t         s,
                                         simtime_t          t,
                                         const D&           dom) {
    { pgen.CustomStat(name, s, t, dom) } -> std::convertible_to<real_t>;
  };

} // namespace traits::pgen

#endif // TRAITS_PGEN_H