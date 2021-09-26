#ifndef OBJECTS_MESHBLOCK_H
#define OBJECTS_MESHBLOCK_H

#include "global.h"

namespace ntt {

template<template<typename T> class D>
class Meshblock {
  NTTArray<typename D<real_t>::ndtype_t> ex1;
  NTTArray<typename D<real_t>::ndtype_t> ex2;
  NTTArray<typename D<real_t>::ndtype_t> ex3;
  NTTArray<typename D<real_t>::ndtype_t> bx1;
  NTTArray<typename D<real_t>::ndtype_t> bx2;
  NTTArray<typename D<real_t>::ndtype_t> bx3;
  NTTArray<typename D<real_t>::ndtype_t> jx1;
  NTTArray<typename D<real_t>::ndtype_t> jx2;
  NTTArray<typename D<real_t>::ndtype_t> jx3;

public:
  Meshblock(std::vector<std::size_t> res);
  ~Meshblock() = default;

  template<template<typename T1> class D1>
  friend class Simulation;

  friend class ProblemGenerator;
};

}

#endif
