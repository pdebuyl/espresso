/*
  Copyright (C) 2010-2018 The ESPResSo project
  Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
    Max-Planck-Institute for Polymer Research, Theory Group

  This file is part of ESPResSo.

  ESPResSo is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ESPResSo is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef SUBT_LJ_H
#define SUBT_LJ_H
/** \file
 *  Routines to subtract the LENNARD-JONES Energy and/or the LENNARD-JONES force
 *  for a particle pair.
 *  \ref forces.cpp
 */

#include "config.hpp"

#ifdef LENNARD_JONES

#include "bonded_interaction_data.hpp"
#include "debug.hpp"
#include "nonbonded_interactions/lj.hpp"

/** set the parameters for the subtract LJ potential
 *
 *  @retval ES_OK on success
 *  @retval ES_ERROR on error
 */
int subt_lj_set_params(int bond_type);

/** Computes the negative of the LENNARD-JONES pair forces
    and adds this force to the particle forces.
    @param p1        Pointer to first particle.
    @param p2        Pointer to second/middle particle.
    @param dx        change in position
    @param force     force on particles
    @return true if bond is broken
*/
inline int calc_subt_lj_pair_force(Particle *p1, Particle *p2,
                                   Bonded_ia_parameters *,
                                   Utils::Vector3d const &dx, double *force) {
  auto ia_params = get_ia_param(p1->p.type, p2->p.type);

  auto const neg_dir = -dx;

  add_lj_pair_force(ia_params, neg_dir.data(), neg_dir.norm(), force);

  return 0;
}

inline int subt_lj_pair_energy(const Particle *p1, const Particle *p2,
                               Bonded_ia_parameters *,
                               Utils::Vector3d const &dx, double *_energy) {
  auto ia_params = get_ia_param(p1->p.type, p2->p.type);

  *_energy = -lj_pair_energy(ia_params, dx.norm());
  return 0;
}

#endif

#endif
