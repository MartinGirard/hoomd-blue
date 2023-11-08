//
// Created by martin on 07.11.23.
//

#ifndef HOOMD_GPUPAIRITERATOR_CUH
#define HOOMD_GPUPAIRITERATOR_CUH

// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"

#include "hoomd/GPUPartition.cuh"

#ifdef __HIPCC__
#include "hoomd/WarpTools.cuh"
#endif // __HIPCC__

#include <assert.h>
#include <type_traits>
namespace hoomd::md {
    struct NeighborData {
        const unsigned int *d_n_neigh;
        const unsigned int *d_nlist;
        const size_t *d_head_list;
    };

    struct PairIterator {
        PairIterator(const NeighborData, unsigned tid, unsigned int idx, unsigned _tpp) : tpp(_tpp)  {
            n_neigh = data.d_n_neigh[idx];
            my_head = data.d_head_list[idx];
            cur_j = tid % tpp < n_neigh ? __ldg(data.d_nlist + my_head + tid % tpp) : 0;
        }

        PairIterator &operator++() {

            if (neigh_idx + tpp < n_neigh) {
                cur_j = __ldg(data.d_nlist + my_head + neigh_idx + tpp);
            }
            neigh_idx++;
            return *this;
        }

        unsigned int &operator->() {
            return cur_j;
        }

        unsigned int operator*() const {
            return cur_j;
        }

        unsigned int begin() {
            return 0;
        }

        unsigned int end() {
            return n_neigh;
        }

        bool operator==(const PairIterator<tpp> &rhs) {
            return neigh_idx == rhs.neigh_idx;
        }

        bool operator!=(const PairIterator<tpp> &rhs) {
            return neigh_idx != rhs.neigh_idx;
        }


    protected:
        unsigned n_neigh, cur_j = 0, neigh_idx = 0;
        const unsigned tpp;
        size_t my_head;
        NeighborData data;
    };

    struct PairParticleData {
        const Scalar4 *d_pos;
        const Scalar *d_charge;
        const BoxDim box;
        const Scalar *rcutsq;
        const Scalar *ron;
    };

    struct BaseInteraction{
        BaseInteraction(PairParticleData pdada) : m_pdata(pdada){}
    protected:
        PairParticleData m_pdata;
    };

    struct DefaultPairInteraction : public BaseInteraction{

        DefaultPairInteraction(PairParticleData pdata) : BaseInteraction(pdata){}

        template<class evaluator, unsigned shift_mode, bool compute_virial>
        Scalar4 operator()(PairIterator &it,
                           Indexer2D &typpair_idx,
                           unsigned idx,
                           typename evaluator::param_type* forcefield) {
            Scalar4 force = {0., 0., 0., 0.};
            // read in the position of our particle.
            Scalar4 postypei = __ldg(m_pdata.d_pos + idx);
            Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

            Scalar qi = Scalar(0);
            if (evaluator::needsCharge())
                qi = __ldg(m_pdata.d_charge + idx);

            for (auto &cur_j: it) {
                // get the neighbor's position
                Scalar4 postypej = __ldg(m_pdata.d_pos + cur_j);
                Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

                Scalar qj = Scalar(0.0);
                if (evaluator::needsCharge())
                    qj = __ldg(m_pdata.d_charge + cur_j);

                // calculate dr (with periodic boundary conditions)
                Scalar3 dx = posi - posj;

                // apply periodic boundary conditions
                dx = m_pdata.box.minImage(dx);

                // calculate r squared
                Scalar rsq = dot(dx, dx);

                // access the per type pair parameters
                unsigned int typpair = typpair_idx(__scalar_as_int(postypei.w),
                                                   __scalar_as_int(postypej.w));
                Scalar ronsq = *(m_pdata.ron+typpair);
                Scalar rcutsq = *(m_pdata.rcutsq + typpair);
                // design specifies that energies are shifted if
                // 1) shift mode is set to shift
                // or 2) shift mode is explor and ron > rcut
                bool energy_shift = false;
                if (shift_mode == 1)
                    energy_shift = true;
                else if (shift_mode == 2) {
                    if (ronsq > rcutsq)
                        energy_shift = true;
                }

                // evaluate the potential
                Scalar force_divr = Scalar(0.0);
                Scalar pair_eng = Scalar(0.0);

                evaluator eval(rsq, rcutsq, *(forcefield + typpair));
                if (evaluator::needsCharge())
                    eval.setCharge(qi, qj);

                eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

                if (shift_mode == 2) {
                    if (rsq >= ronsq && rsq < rcutsq) {
                        // Implement XPLOR smoothing
                        Scalar old_pair_eng = pair_eng;
                        Scalar old_force_divr = force_divr;

                        // calculate 1.0 / (xplor denominator)
                        Scalar xplor_denom_inv
                                = Scalar(1.0)
                                  / ((rcutsq - ronsq) * (rcutsq - ronsq) *
                                     (rcutsq - ronsq));

                        Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                        Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq
                                   * (rcutsq + Scalar(2.0) * rsq -
                                      Scalar(3.0) * ronsq)
                                   * xplor_denom_inv;
                        Scalar ds_dr_divr
                                = Scalar(12.0) * (rsq - ronsq) *
                                  rsq_minus_r_cut_sq * xplor_denom_inv;

                        // make modifications to the old pair energy and force
                        pair_eng = old_pair_eng * s;
                        force_divr =
                                s * old_force_divr - ds_dr_divr * old_pair_eng;
                    }
                }
                // calculate the virial
                if (compute_virial) {
                    Scalar force_div2r = Scalar(0.5) * force_divr;
                    virialxx += dx.x * dx.x * force_div2r;
                    virialxy += dx.x * dx.y * force_div2r;
                    virialxz += dx.x * dx.z * force_div2r;
                    virialyy += dx.y * dx.y * force_div2r;
                    virialyz += dx.y * dx.z * force_div2r;
                    virialzz += dx.z * dx.z * force_div2r;
                }

                // add up the force vector components
                force.x += dx.x * force_divr;
                force.y += dx.y * force_divr;
                force.z += dx.z * force_divr;

                force.w += pair_eng;
            }
            // potential energy per particle must be halved
            force.w *= Scalar(0.5);
            return force;
        }

    };
}
#endif // HOOMD_GPUPAIRITERATOR_CUH
