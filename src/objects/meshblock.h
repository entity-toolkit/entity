#ifndef OBJECTS_MESHBLOCK_H
#define OBJECTS_MESHBLOCK_H

#include "global.h"

namespace ntt {

template<template<typename T> class D>
class Meshblock {
public:
  NTTArray<typename D<double>::ndtype_t> ex1;
  NTTArray<typename D<double>::ndtype_t> ex2;
  NTTArray<typename D<double>::ndtype_t> ex3;
  // NTTArray<typename D<double>::ndtype_t> bx1;
  // NTTArray<typename D<double>::ndtype_t> bx2;
  // NTTArray<typename D<double>::ndtype_t> bx3;

  Meshblock(const int N);
  ~Meshblock() {};
};

}

#endif
