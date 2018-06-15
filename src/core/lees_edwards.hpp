#ifdef LEES_EDWARDS

#include "config.hpp"

extern double lees_edwards_offset;
extern double lees_edwards_velocity;

enum LeesEdwardsProtocolType {
    LEES_EDWARDS_PROTOCOL_OFF,
    LEES_EDWARDS_PROTOCOL_STEP,
    LEES_EDWARDS_PROTOCOL_CST_SHEAR,
    LEES_EDWARDS_PROTOCOL_OSC_SHEAR,
};

//typedef struct lees_edwards_protocol {
//  char lees_edwards_funtion[];
//  extern double lees_edwards_offset;
//  extern double lees_edwards_velocity;
//  double lees_edwards_amplitude;
//  double lees_edwards_frequency;
//  };

#endif
