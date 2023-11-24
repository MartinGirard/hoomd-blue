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

#ifndef HOSTDEVICE
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

#ifdef __HIPCC__
#define INLINE __forceinline__
#else
#define INLINE inline
#endif

namespace hoomd::md {

    template<typename T>
    T HOSTDEVICE inline load(T* a){
#ifdef __HIPCC__
        return __ldg(a);
#else
        return *a;
#endif
    };

    struct NeighborData {
        const unsigned int n_neigh;
        const unsigned int* d_nlist;
        const size_t my_head;
        const Scalar4 *d_pos;
        const Scalar *d_charge;
    };

    struct PairIterator {
        HOSTDEVICE PairIterator(const NeighborData _data, unsigned tid, unsigned _tpp) :
        data(_data), tpp(_tpp) {
            if(tid % tpp < data.n_neigh) {
                cur_j = load(data.d_nlist + data.my_head + tid % tpp);
            }
            neigh_idx = tid % tpp;
            valid = neigh_idx < data.n_neigh;
        }

        HOSTDEVICE INLINE PairIterator& operator++() {
            if (neigh_idx + tpp < data.n_neigh) {
                cur_j = load(data.d_nlist + data.my_head + neigh_idx + tpp);
            }
            neigh_idx += tpp;
            valid = neigh_idx < data.n_neigh;
            return *this;
        }

        HOSTDEVICE unsigned int operator*() const {
            return cur_j;
        }

        HOSTDEVICE auto position() const{return load(data.d_pos + cur_j);};
        HOSTDEVICE Scalar charge() const{return load(data.d_charge + cur_j);};

        unsigned cur_j = 0, neigh_idx = 0;

        const NeighborData data;
        const unsigned tpp;
        bool valid;
    };

    struct PairParticleData {
        const BoxDim box;
        const Scalar *rcutsq;
        const Scalar *ron;
    };

    struct iParticleData{
        const Scalar4 postype;
        const Scalar qi;
    };

    struct Virial{
        Scalar xx, yy, zz, xy, xz, yz;
    };

    struct ThirdLaw{
        ThirdLaw(Scalar4* _force, Scalar* _virial, size_t _pitch, size_t _local) : force(_force), virial(_virial), pitch(_pitch), local(_local){}

        void operator()(const Scalar3& dx,
                        const Scalar& force_divr,
                        const Scalar& force_div2r,
                        const Scalar& pair_eng,
                        unsigned int j,
                        bool compute_virial
        ) const{
            if(j < local) {
                unsigned int mem_idx = j;
                force[mem_idx].x -= dx.x * force_divr;
                force[mem_idx].y -= dx.y * force_divr;
                force[mem_idx].z -= dx.z * force_divr;
                force[mem_idx].w += pair_eng * Scalar(0.5);
                if (compute_virial) {
                    virial[0 * pitch + mem_idx] += force_div2r * dx.x * dx.x;
                    virial[1 * pitch + mem_idx] += force_div2r * dx.x * dx.y;
                    virial[2 * pitch + mem_idx] += force_div2r * dx.x * dx.z;
                    virial[3 * pitch + mem_idx] += force_div2r * dx.y * dx.y;
                    virial[4 * pitch + mem_idx] += force_div2r * dx.y * dx.z;
                    virial[5 * pitch + mem_idx] += force_div2r * dx.z * dx.z;
                }
            }
        };

        Scalar4* force;
        Scalar* virial;
        size_t pitch;
        size_t local;
    };

    struct BaseInteraction{
        HOSTDEVICE BaseInteraction(PairParticleData pdada) : m_pdata(pdada){}
    protected:
        const PairParticleData m_pdata;
    };


    struct DefaultPairInteraction : public BaseInteraction{

        HOSTDEVICE DefaultPairInteraction(PairParticleData pdata, Scalar4& f, Virial& _v) : BaseInteraction(pdata), force(f), v(_v){}

        HOSTDEVICE constexpr static bool reduce_and_write() {
            return true;
        }

        template<class evaluator, unsigned shift_mode, bool compute_virial, bool third_law = false>
        auto HOSTDEVICE INLINE operator()(PairIterator &it, // iterator over neighbors
                           const Index2D& typpair_idx, // Index2D to access pair values
                           const iParticleData& idata, // particle index we are computing
                           const typename evaluator::param_type* forcefield, // force-field parameters
                           const void* extra,// required extra stuff
                           const ThirdLaw* TL = nullptr
                           ) // set the hook to nullptr to avoid usage unless specified
                           {
            //Scalar4 force = {0., 0., 0., 0.};
            //Virial v{};
            // read in the position of our particle.
            Scalar4 postypei = idata.postype;
            Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

            while (it.valid) {
                {
                    auto cur_j = *it;
                    Scalar4 postypej = it.position();
                    Scalar qj = evaluator::needsCharge()? it.charge() : Scalar(0.0); // there's no load if charge is not needed
                    ++it;

                    // get the neighbor's position
                    Scalar3 posj = make_scalar3(postypej.x, postypej.y,
                                                postypej.z);

                    // calculate dr (with periodic boundary conditions)
                    Scalar3 dx = posi - posj;

                    // apply periodic boundary conditions
                    dx = m_pdata.box.minImage(dx);

                    // calculate r squared
                    Scalar rsq = dot(dx, dx);

                    // access the per type pair parameters
                    unsigned int typpair = typpair_idx(
                            __scalar_as_int(postypei.w),
                            __scalar_as_int(postypej.w));

                    Scalar ronsq =
                            shift_mode == 2 ? *(m_pdata.ron + typpair) : Scalar(
                                    0.0);
                    Scalar rcutsq = *(m_pdata.rcutsq + typpair);
                    // design specifies that energies are shifted if
                    // 1) shift mode is set to shift
                    // or 2) shift mode is explor and ron > rcut
                    bool energy_shift = false;
                    if  (shift_mode == 1)
                        energy_shift = true;
                    else if  (shift_mode == 2) {
                        if (ronsq > rcutsq)
                            energy_shift = true;
                    }

                    // evaluate the potential
                    Scalar force_divr = Scalar(0.0);
                    Scalar pair_eng = Scalar(0.0);

                    evaluator eval(rsq, rcutsq, *(forcefield + typpair));
                    if constexpr (evaluator::needsCharge())
                        eval.setCharge(idata.qi, qj);

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
                                    s * old_force_divr -
                                    ds_dr_divr * old_pair_eng;
                        }
                    }
                    // calculate the virial
                    if (compute_virial) {
                        Scalar force_div2r = Scalar(0.5) * force_divr;
                        v.xx += dx.x * dx.x * force_div2r;
                        v.xy += dx.x * dx.y * force_div2r;
                        v.xz += dx.x * dx.z * force_div2r;
                        v.yy += dx.y * dx.y * force_div2r;
                        v.yz += dx.y * dx.z * force_div2r;
                        v.zz += dx.z * dx.z * force_div2r;
                    }

                    // add up the force vector components
                    force.x += dx.x * force_divr;
                    force.y += dx.y * force_divr;
                    force.z += dx.z * force_divr;

                    force.w += pair_eng;
                    // enclose the 3rd law use on CPU within a template check so it doesnt get compiled at all on gpu
                    if(third_law) {
                        auto third_law_compute = *((ThirdLaw *) TL);
                        third_law_compute(dx, force_divr,
                                          Scalar(0.5) * force_divr, pair_eng,
                                          cur_j, compute_virial);
                    }
                }
            }
            // potential energy per particle must be halved
            force.w *= Scalar(0.5);
            return;
            //return std::make_pair(force, v);
        }

        Scalar4& force;
        Virial& v;
    };

    template<class Interaction, class evaluator, unsigned shift_mode, bool compute_virial, bool third_law>
    struct InteractionDispatchHelper{
        template<class... Args>
                auto Call(Interaction& I, Args... args){
                    return I.template operator()<evaluator, shift_mode, compute_virial, third_law>(args...);
                }
    };



    template<class Interaction, class evaluator, unsigned shift_mode, bool third_law, class... Args>
    auto resolveVirial(Interaction& i, bool has_virial, Args... args){
        if (has_virial) {
            auto I = InteractionDispatchHelper<Interaction, evaluator, shift_mode, true, third_law>{};
            return I.Call(i, args...);
        }
        else {
            auto I = InteractionDispatchHelper<Interaction, evaluator, shift_mode, false, third_law>{};
            return I.Call(i, args...);
        }
    }

    template<class Interaction, class evaluator, bool third_law, class... Args>
    auto resolveShift(Interaction& i, unsigned shift, bool has_virial, Args... args){
        switch (shift) {
            case 0:
                return resolveVirial<Interaction, evaluator, 0, third_law>(i, has_virial, args...);
            case 1:
                return resolveVirial<Interaction, evaluator, 1, third_law>(i, has_virial, args...);
            case 2:
                return resolveVirial<Interaction, evaluator, 2, third_law>(i, has_virial, args...);
            default:
                throw std::runtime_error("Bad shift value");
        }
    }

    template<class Interaction, class evaluator, class... Args>
    auto dispatch(Interaction& i, unsigned shift, bool has_virial, bool third_law, Args... args){
        if(third_law)
            return resolveShift<Interaction, evaluator, true>(i, shift, has_virial, args...);
        else
            return resolveShift<Interaction, evaluator, false>(i, shift, has_virial, args...);
    }

}
#endif // HOOMD_GPUPAIRITERATOR_CUH
