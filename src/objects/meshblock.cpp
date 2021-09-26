#include "global.h"
#include "meshblock.h"

namespace ntt {

template <>
Meshblock<One_D>::Meshblock(std::vector<std::size_t> res) :
  ex1{"Ex1", res[0]},
  ex2{"Ex2", res[0]},
  ex3{"Ex3", res[0]},
  bx1{"Bx1", res[0]},
  bx2{"Bx2", res[0]},
  bx3{"Bx3", res[0]},
  jx1{"Jx1", res[0]},
  jx2{"Jx2", res[0]},
  jx3{"Jx3", res[0]} {}

template <>
Meshblock<Two_D>::Meshblock(std::vector<std::size_t> res) :
  ex1{"Ex1", res[0], res[1]},
  ex2{"Ex2", res[0], res[1]},
  ex3{"Ex3", res[0], res[1]},
  bx1{"Bx1", res[0], res[1]},
  bx2{"Bx2", res[0], res[1]},
  bx3{"Bx3", res[0], res[1]},
  jx1{"Jx1", res[0], res[1]},
  jx2{"Jx2", res[0], res[1]},
  jx3{"Jx3", res[0], res[1]} {}

template <>
Meshblock<Three_D>::Meshblock(std::vector<std::size_t> res) :
  ex1{"Ex1", res[0], res[1], res[2]},
  ex2{"Ex2", res[0], res[1], res[2]},
  ex3{"Ex3", res[0], res[1], res[2]},
  bx1{"Bx1", res[0], res[1], res[2]},
  bx2{"Bx2", res[0], res[1], res[2]},
  bx3{"Bx3", res[0], res[1], res[2]},
  jx1{"Jx1", res[0], res[1], res[2]},
  jx2{"Jx2", res[0], res[1], res[2]},
  jx3{"Jx3", res[0], res[1], res[2]} {}
}

template class ntt::Meshblock<ntt::One_D>;
template class ntt::Meshblock<ntt::Two_D>;
template class ntt::Meshblock<ntt::Three_D>;
