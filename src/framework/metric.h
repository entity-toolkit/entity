#ifndef FRAMEWORK_METRIC_H
#define FRAMEWORK_METRIC_H

#include "global.h"

#if (METRIC == MINKOWSKI_METRIC)
#  include "minkowski.h"
#elif (METRIC == SPHERICAL_METRIC)
#  include "spherical.h"
#elif (METRIC == QSPHERICAL_METRIC)
#  include "qspherical.h"
#elif (METRIC == KERR_SCHILD_METRIC)
#  include "kerr_schild.h"
#elif (METRIC == QKERR_SCHILD_METRIC)
#  include "qkerr_schild.h"
#endif

#endif
