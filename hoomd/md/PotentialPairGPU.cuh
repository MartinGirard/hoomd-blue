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
#include "GPUPairIterator.cuh"
/*! \file PotentialPairGPU.cuh
    \brief Defines templated GPU kernel code for calculating the pair forces.
*/

#ifndef __POTENTIAL_PAIR_GPU_CUH__
#define __POTENTIAL_PAIR_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Maximum number of threads (width of a warp)
// currently this is hardcoded, we should set it to the max of platforms
#if defined(__HIP_PLATFORM_NVCC__)
const int gpu_pair_force_max_tpp = 32;
#elif defined(__HIP_PLATFORM_HCC__)
const int gpu_pair_force_max_tpp = 64;
#endif

//! Wraps arguments to gpu_cgpf
struct pair_args_t
    {
    //! Construct a pair_args_t
    pair_args_t(Scalar4* _d_force,
                Scalar* _d_virial,
                const size_t _virial_pitch,
                const unsigned int _N,
                const unsigned int _n_max,
                const Scalar4* _d_pos,
                const Scalar* _d_charge,
                const BoxDim& _box,
                const unsigned int* _d_n_neigh,
                const unsigned int* _d_nlist,
                const size_t* _d_head_list,
                const Scalar* _d_rcutsq,
                const Scalar* _d_ronsq,
                const size_t _size_neigh_list,
                const unsigned int _ntypes,
                const unsigned int _block_size,
                const unsigned int _shift_mode,
                const unsigned int _compute_virial,
                const unsigned int _threads_per_particle,
                const GPUPartition& _gpu_partition,
                const hipDeviceProp_t& _devprop)
        : d_force(_d_force), d_virial(_d_virial), virial_pitch(_virial_pitch), N(_N), n_max(_n_max),
          d_pos(_d_pos), d_charge(_d_charge), box(_box), d_n_neigh(_d_n_neigh), d_nlist(_d_nlist),
          d_head_list(_d_head_list), d_rcutsq(_d_rcutsq), d_ronsq(_d_ronsq),
          size_neigh_list(_size_neigh_list), ntypes(_ntypes), block_size(_block_size),
          shift_mode(_shift_mode), compute_virial(_compute_virial),
          threads_per_particle(_threads_per_particle), gpu_partition(_gpu_partition),
          devprop(_devprop) {};

    Scalar4* d_force;          //!< Force to write out
    Scalar* d_virial;          //!< Virial to write out
    const size_t virial_pitch; //!< The pitch of the 2D array of virial matrix elements
    const unsigned int N;      //!< number of particles
    const unsigned int n_max;  //!< Max size of pdata arrays
    const Scalar4* d_pos;      //!< particle positions
    const Scalar* d_charge;    //!< particle charges
    const BoxDim box;          //!< Simulation box in GPU format
    const unsigned int*
        d_n_neigh;                //!< Device array listing the number of neighbors on each particle
    const unsigned int* d_nlist;  //!< Device array listing the neighbors of each particle
    const size_t* d_head_list;    //!< Head list indexes for accessing d_nlist
    const Scalar* d_rcutsq;       //!< Device array listing r_cut squared per particle type pair
    const Scalar* d_ronsq;        //!< Device array listing r_on squared per particle type pair
    const size_t size_neigh_list; //!< Size of the neighbor list for texture binding
    const unsigned int ntypes;    //!< Number of particle types in the simulation
    const unsigned int block_size;           //!< Block size to execute
    const unsigned int shift_mode;           //!< The potential energy shift mode
    const unsigned int compute_virial;       //!< Flag to indicate if virials should be computed
    const unsigned int threads_per_particle; //!< Number of threads per particle (maximum: 1 warp)
    const GPUPartition& gpu_partition; //!< The load balancing partition of particles between GPUs
    const hipDeviceProp_t& devprop;    //!< CUDA device properties
    };

#ifdef __HIPCC__

//! Kernel for calculating pair forces
/*! This kernel is called to calculate the pair forces on all N particles. Actual evaluation of the
   potentials and forces for each pair is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles in system
    \param d_pos particle positions
    \param d_charge particle charges
    \param box Box dimensions used to implement periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param d_head_list Indexes for reading \a d_nlist
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param d_ronsq ron squared, stored per type pair
    \param ntypes Number of types in the simulation
    \param offset Offset of first particle

    \a d_params, \a d_rcutsq, and \a d_ronsq must be indexed with an Index2DUpperTriangular(typei,
   typej) to access the unique value for that type pair. These values are all cached into shared
   memory for quick access, so a dynamic amount of shared memory must be allocated for this kernel
   launch. The amount is (2*sizeof(Scalar) + sizeof(typename evaluator::param_type)) *
   typpair_idx.getNumElements()

    Certain options are controlled via template parameters to avoid the performance hit when they
   are not enabled. \tparam evaluator EvaluatorPair class to evaluate V(r) and -delta V(r)/r \tparam
   shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR switching
   is enabled (See PotentialPair for a discussion on what that entails) \tparam compute_virial When
   non-zero, the virial tensor is computed. When zero, the virial tensor is not computed. \tparam
   tpp Number of threads to use per particle, must be power of 2 and smaller than warp size

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each group of \a tpp threads will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template<class evaluator,
         unsigned int shift_mode,
         unsigned int compute_virial,
         int tpp,
         bool enable_shared_cache,
         class Interaction = DefaultPairInteraction>
__global__ void
gpu_compute_pair_forces_shared_kernel(Scalar4* d_force,
                                      Scalar* d_virial,
                                      const size_t virial_pitch,
                                      const unsigned int N,
                                      const Scalar4* d_pos,
                                      const Scalar* d_charge,
                                      const BoxDim box,
                                      const unsigned int* d_n_neigh,
                                      const unsigned int* d_nlist,
                                      const size_t* d_head_list,
                                      const typename evaluator::param_type* d_params,
                                      const Scalar* d_rcutsq,
                                      const Scalar* d_ronsq,
                                      const unsigned int ntypes,
                                      const unsigned int offset,
                                      unsigned int max_extra_bytes)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    HIP_DYNAMIC_SHARED(char, s_data)
    typename evaluator::param_type* s_params = (typename evaluator::param_type*)(&s_data[0]);
    Scalar* s_rcutsq
        = (Scalar*)(&s_data[num_typ_parameters * sizeof(typename evaluator::param_type)]);
    Scalar* s_ronsq
        = (Scalar*)(&s_data[num_typ_parameters
                            * (sizeof(typename evaluator::param_type) + sizeof(Scalar))]);
    auto s_extra = reinterpret_cast<char*>(s_ronsq + num_typ_parameters);

    if (enable_shared_cache)
        {
        // load in the per type pair parameters
        for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < num_typ_parameters)
                {
                s_rcutsq[cur_offset + threadIdx.x] = d_rcutsq[cur_offset + threadIdx.x];
                if (shift_mode == 2)
                    s_ronsq[cur_offset + threadIdx.x] = d_ronsq[cur_offset + threadIdx.x];
                }
            }

        unsigned int param_size
            = num_typ_parameters * sizeof(typename evaluator::param_type) / sizeof(int);
        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < param_size)
                {
                ((int*)s_params)[cur_offset + threadIdx.x]
                    = ((int*)d_params)[cur_offset + threadIdx.x];
                }
            }

        __syncthreads();

        // initialize extra shared mem
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int cur_pair = 0; cur_pair < num_typ_parameters; ++cur_pair)
            s_params[cur_pair].load_shared(s_extra, available_bytes);

        __syncthreads();
        }

    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    bool active = true;
    if (idx >= N)
        {
        // need to mask this thread, but still participate in warp-level reduction
        active = false;
        }

    // add offset to get actual particle index
    idx += offset;

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Virial V{};
    if (active)
        {

        const iParticleData idata{
            .postype = load(d_pos + idx),
            .qi = evaluator::needsCharge() ? load(d_charge + idx) : Scalar(0)
        };

        const NeighborData n{
            .d_nlist = d_nlist,
            .my_head = d_head_list[idx],
            .d_pos = d_pos,
            .d_charge = d_charge,
            .n_neigh  = d_n_neigh[idx],
        };
        PairIterator iterator(n, threadIdx.x, tpp);

        const PairParticleData pdata{
            .box = box,
            .rcutsq = enable_shared_cache ? s_rcutsq : d_rcutsq,
            .ron = shift_mode == 2 ? (enable_shared_cache ? s_ronsq : d_ronsq) : nullptr
        };
        const auto FEval = Interaction(pdata, force, V);
        const auto params = enable_shared_cache ? s_params : d_params;
        FEval.template operator()<evaluator, shift_mode, compute_virial>(iterator, typpair_idx, idata, params, nullptr);
        }
    // either the Interaction type takes care of outputing forces (eg triplet potentials) or we do it here (standard)
    if constexpr (Interaction::reduce_and_write()) {
        // reduce force over threads in cta
        hoomd::detail::WarpReduce<Scalar, tpp> reducer;
        force.x = reducer.Sum(force.x);
        force.y = reducer.Sum(force.y);
        force.z = reducer.Sum(force.z);
        force.w = reducer.Sum(force.w);

        // now that the force calculation is complete, write out the result
        if (active && threadIdx.x % tpp == 0)
            d_force[idx] = force;

        if (compute_virial) {
            V.xx = reducer.Sum(V.xx);
            V.xy = reducer.Sum(V.xy);
            V.xz = reducer.Sum(V.xz);
            V.yy = reducer.Sum(V.yy);
            V.yz = reducer.Sum(V.yz);
            V.zz = reducer.Sum(V.zz);

            // if we are the first thread in the cta, write out virial to global mem
            if (active && threadIdx.x % tpp == 0) {
                d_virial[0 * virial_pitch + idx] = V.xx;
                d_virial[1 * virial_pitch + idx] = V.xy;
                d_virial[2 * virial_pitch + idx] = V.xz;
                d_virial[3 * virial_pitch + idx] = V.yy;
                d_virial[4 * virial_pitch + idx] = V.yz;
                d_virial[5 * virial_pitch + idx] = V.zz;
            }
        }
    }
    }

template<typename T> int get_max_block_size(T func)
    {
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % gpu_pair_force_max_tpp;
    return max_threads;
    }

//! Pair force compute kernel launcher
/*!
 * \tparam evaluator EvaluatorPair class to evaluate V(r) and -delta V(r)/r
 * \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR
 * switching is enabled (See PotentialPair for a discussion on what that entails) \tparam
 * compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not
 * computed. \tparam tpp Number of threads to use per particle, must be power of 2 and smaller than
 * warp size
 *
 * Partial function template specialization is not allowed in C++, so instead we have to wrap this
 * with a struct that we are allowed to partially specialize.
 */
template<class evaluator, unsigned int shift_mode, unsigned int compute_virial, int tpp>
struct PairForceComputeKernel
    {
    //! Launcher for the pair force kernel
    /*!
     * \param pair_args Other arguments to pass onto the kernel
     * \param range Range of particle indices this GPU operates on
     * \param d_params Parameters for the potential, stored per type pair
     */

    static void launch(const pair_args_t& pair_args,
                       std::pair<unsigned int, unsigned int> range,
                       const typename evaluator::param_type* d_params)
        {
        unsigned int N = range.second - range.first;
        unsigned int offset = range.first;

        if (tpp == pair_args.threads_per_particle)
            {
            unsigned int block_size = pair_args.block_size;
            bool enable_shared_cache = true;

            Index2D typpair_idx(pair_args.ntypes);
            size_t param_shared_bytes
                = (2 * sizeof(Scalar) + sizeof(typename evaluator::param_type))
                  * typpair_idx.getNumElements();

            unsigned int max_block_size;
            max_block_size
                = get_max_block_size(gpu_compute_pair_forces_shared_kernel<evaluator,
                                                                           shift_mode,
                                                                           compute_virial,
                                                                           tpp,
                                                                           true>);

            hipFuncAttributes attr;
            hipFuncGetAttributes(
                &attr,
                reinterpret_cast<const void*>(&gpu_compute_pair_forces_shared_kernel<evaluator,
                                                                                     shift_mode,
                                                                                     compute_virial,
                                                                                     tpp,
                                                                                     true>));

            if (param_shared_bytes + attr.sharedSizeBytes > pair_args.devprop.sharedMemPerBlock)
                {
                param_shared_bytes = 0;
                enable_shared_cache = false;
                }

            unsigned int max_extra_bytes = static_cast<unsigned int>(
                pair_args.devprop.sharedMemPerBlock - param_shared_bytes - attr.sharedSizeBytes);

            // determine dynamically requested shared memory in nested managed arrays
            char* ptr = nullptr;
            unsigned int available_bytes = max_extra_bytes;
            for (unsigned int i = 0; i < typpair_idx.getNumElements(); ++i)
                {
                d_params[i].allocate_shared(ptr, available_bytes);
                }

            unsigned int extra_shared_bytes = max_extra_bytes - available_bytes;

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(N / (block_size / tpp) + 1, 1, 1);

            if (enable_shared_cache)
                {
                hipLaunchKernelGGL((gpu_compute_pair_forces_shared_kernel<evaluator,
                                                                          shift_mode,
                                                                          compute_virial,
                                                                          tpp,
                                                                          true>),
                                   dim3(grid),
                                   dim3(block_size),
                                   param_shared_bytes + extra_shared_bytes,
                                   0,
                                   pair_args.d_force,
                                   pair_args.d_virial,
                                   pair_args.virial_pitch,
                                   N,
                                   pair_args.d_pos,
                                   pair_args.d_charge,
                                   pair_args.box,
                                   pair_args.d_n_neigh,
                                   pair_args.d_nlist,
                                   pair_args.d_head_list,
                                   d_params,
                                   pair_args.d_rcutsq,
                                   pair_args.d_ronsq,
                                   pair_args.ntypes,
                                   offset,
                                   max_extra_bytes);
                }
            else
                {
                hipLaunchKernelGGL((gpu_compute_pair_forces_shared_kernel<evaluator,
                                                                          shift_mode,
                                                                          compute_virial,
                                                                          tpp,
                                                                          false>),
                                   dim3(grid),
                                   dim3(block_size),
                                   param_shared_bytes + extra_shared_bytes,
                                   0,
                                   pair_args.d_force,
                                   pair_args.d_virial,
                                   pair_args.virial_pitch,
                                   N,
                                   pair_args.d_pos,
                                   pair_args.d_charge,
                                   pair_args.box,
                                   pair_args.d_n_neigh,
                                   pair_args.d_nlist,
                                   pair_args.d_head_list,
                                   d_params,
                                   pair_args.d_rcutsq,
                                   pair_args.d_ronsq,
                                   pair_args.ntypes,
                                   offset,
                                   max_extra_bytes);
                }
            }
        else
            {
            PairForceComputeKernel<evaluator, shift_mode, compute_virial, tpp / 2>::launch(
                pair_args,
                range,
                d_params);
            }
        }
    };

//! Template specialization to do nothing for the tpp = 0 case
template<class evaluator, unsigned int shift_mode, unsigned int compute_virial>
struct PairForceComputeKernel<evaluator, shift_mode, compute_virial, 0>
    {
    static void launch(const pair_args_t& pair_args,
                       std::pair<unsigned int, unsigned int> range,
                       const typename evaluator::param_type* d_params)
        {
        // do nothing
        }
    };

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param pair_args Other arguments to pass onto the kernel
    \param d_params Parameters for the potential, stored per type pair

    This is just a driver function for gpu_compute_pair_forces_shared_kernel(), see it for details.
*/
template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_pair_forces(const pair_args_t& pair_args,
                        const typename evaluator::param_type* d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.d_ronsq);
    assert(pair_args.ntypes > 0);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = pair_args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = pair_args.gpu_partition.getRangeAndSetGPU(idev);

        // Launch kernel
        if (pair_args.compute_virial)
            {
            switch (pair_args.shift_mode)
                {
            case 0:
                {
                PairForceComputeKernel<evaluator, 0, 1, gpu_pair_force_max_tpp>::launch(pair_args,
                                                                                        range,
                                                                                        d_params);
                break;
                }
            case 1:
                {
                PairForceComputeKernel<evaluator, 1, 1, gpu_pair_force_max_tpp>::launch(pair_args,
                                                                                        range,
                                                                                        d_params);
                break;
                }
            case 2:
                {
                PairForceComputeKernel<evaluator, 2, 1, gpu_pair_force_max_tpp>::launch(pair_args,
                                                                                        range,
                                                                                        d_params);
                break;
                }
            default:
                break;
                }
            }
        else
            {
            switch (pair_args.shift_mode)
                {
            case 0:
                {
                PairForceComputeKernel<evaluator, 0, 0, gpu_pair_force_max_tpp>::launch(pair_args,
                                                                                        range,
                                                                                        d_params);
                break;
                }
            case 1:
                {
                PairForceComputeKernel<evaluator, 1, 0, gpu_pair_force_max_tpp>::launch(pair_args,
                                                                                        range,
                                                                                        d_params);
                break;
                }
            case 2:
                {
                PairForceComputeKernel<evaluator, 2, 0, gpu_pair_force_max_tpp>::launch(pair_args,
                                                                                        range,
                                                                                        d_params);
                break;
                }
            default:
                break;
                }
            }
        }

    return hipSuccess;
    }
#else
template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_pair_forces(const pair_args_t& pair_args,
                        const typename evaluator::param_type* d_params);
#endif

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // __POTENTIAL_PAIR_GPU_CUH__
