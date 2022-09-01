#ifndef FRAMEWORK_METRIC_H
#define FRAMEWORK_METRIC_H

#include "global.h"

#ifdef MINKOWSKI_METRIC
#  include "minkowski.h"
#elif defined(SPHERICAL_METRIC)
#  include "spherical.h"
#elif defined(QSPHERICAL_METRIC)
#  include "qspherical.h"
#elif defined(KERR_SCHILD_METRIC)
#  include "kerr_schild.h"
#elif defined(QKERR_SCHILD_METRIC)
#  include "qkerr_schild.h"
#endif

#endif
