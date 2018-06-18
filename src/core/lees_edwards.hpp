#ifndef LEES_EDWARDS_H
#define LEES_EDWARDS_H

#include "config.hpp"

enum LeesEdwardsProtocolType {
    LEES_EDWARDS_PROTOCOL_OFF,
    LEES_EDWARDS_PROTOCOL_STEP,
    LEES_EDWARDS_PROTOCOL_STEADY_SHEAR,
    LEES_EDWARDS_PROTOCOL_OSC_SHEAR,
};

typedef struct {
  LeesEdwardsProtocolType type;
  double time0;
  double offset;
  double velocity;
  double amplitude;
  double frequency;
} lees_edwards_protocol_struct;

extern lees_edwards_protocol_struct lees_edwards_protocol;

typedef struct {
  double offset;
  double velocity;
} lees_edwards_state;

extern lees_edwards_state lees_edwards_data;

#endif
