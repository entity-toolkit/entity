#include "global.h"
#include "meshblock.h"

namespace ntt {

template<template<typename T> class D>
Meshblock<D>::Meshblock(const int N) : ex1("Ex1", N), ex2("Ex2", N), ex3("Ex3", N) {

}

template class Meshblock<One_D>;
template class Meshblock<Two_D>;
template class Meshblock<Three_D>;

}
