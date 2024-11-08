// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef ENABLE_HIP

#include "hoomd/hpmc/IntegratorHPMCMono.h"
#include "hoomd/hpmc/IntegratorHPMCMonoGPUTypes.cuh"

#include "hoomd/Autotuner.h"
#include "hoomd/GPUVector.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include "hoomd/GPUPartition.cuh"

#include <hip/hip_runtime.h>

#ifdef ENABLE_MPI
#include "hoomd/MPIConfiguration.h"
#include <mpi.h>
#endif

/*! \file IntegratorHPMCMonoGPU.h
    \brief Defines the template class for HPMC on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
//! Helper class to manage shuffled update orders in a GlobalVector
/*! Stores an update order from 0 to N-1, inclusive, and can be resized. shuffle() shuffles the
   order of elements to a new random permutation. operator [i] gets the index of the item at order i
   in the current shuffled sequence.

    NOTE: this should supersede UpdateOrder

    \note we use GPUArrays instead of GlobalArrays currently to allow host access to the shuffled
   order without an unnecessary hipDeviceSynchronize()

    \ingroup hpmc_data_structs
*/
class UpdateOrderGPU
    {
    public:
    //! Constructor
    /*! \param seed Random number seed
        \param N number of integers to shuffle
    */
    UpdateOrderGPU(std::shared_ptr<const ExecutionConfiguration> exec_conf, unsigned int N = 0)
        : m_is_reversed(false), m_update_order(exec_conf), m_reverse_update_order(exec_conf)
        {
        resize(N);
        }

    //! Resize the order
    /*! \param N new size
        \post The order is 0, 1, 2, ... N-1
    */
    void resize(unsigned int N)
        {
        if (!N || N == m_update_order.size())
            return;

        // initialize the update order
        m_update_order.resize(N);
        m_reverse_update_order.resize(N);

        ArrayHandle<unsigned int> h_update_order(m_update_order,
                                                 access_location::host,
                                                 access_mode::overwrite);
        ArrayHandle<unsigned int> h_reverse_update_order(m_reverse_update_order,
                                                         access_location::host,
                                                         access_mode::overwrite);

        for (unsigned int i = 0; i < N; i++)
            {
            h_update_order.data[i] = i;
            h_reverse_update_order.data[i] = N - i - 1;
            }
        m_is_reversed = false;
        }

    //! Shuffle the order
    /*! \param timestep Current timestep of the simulation
        \note \a timestep is used to seed the RNG, thus assuming that the order is shuffled only
       once per timestep.
    */
    void shuffle(uint64_t timestep, uint16_t seed, unsigned int rank, unsigned int select = 0)
        {
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoShuffle, timestep, seed),
            hoomd::Counter(rank, select));

        // reverse the order with 1/2 probability
        m_is_reversed = hoomd::UniformIntDistribution(1)(rng);
        }

    //! Access element of the shuffled order
    unsigned int operator[](unsigned int i)
        {
        const GlobalVector<unsigned int>& update_order
            = m_is_reversed ? m_reverse_update_order : m_update_order;
        ArrayHandle<unsigned int> h_update_order(update_order,
                                                 access_location::host,
                                                 access_mode::read);
        return h_update_order.data[i];
        }

    //! Access the underlying GlobalVector
    const GlobalVector<unsigned int>& get() const
        {
        if (m_is_reversed)
            return m_reverse_update_order;
        else
            return m_update_order;
        }

    private:
    bool m_is_reversed;                                //!< True if order is reversed
    GlobalVector<unsigned int> m_update_order;         //!< Update order
    GlobalVector<unsigned int> m_reverse_update_order; //!< Inverse permutation
    };

    } // end namespace detail

//! Template class for HPMC update on the GPU
/*!
    \ingroup hpmc_integrators
*/
template<class Shape> class IntegratorHPMCMonoGPU : public IntegratorHPMCMono<Shape>
    {
    public:
    //! Construct the integrator
    IntegratorHPMCMonoGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<CellList> cl);
    //! Destructor
    virtual ~IntegratorHPMCMonoGPU();

    /// Start autotuning kernel launch parameters
    virtual void startAutotuning()
        {
        IntegratorHPMCMono<Shape>::startAutotuning();

        // Tune patch kernels and cell list in addition to those in `m_autotuners`.
        m_cl->startAutotuning();
        }

    //! Take one timestep forward
    virtual void update(uint64_t timestep);

#ifdef ENABLE_MPI
    void setNtrialCommunicator(std::shared_ptr<MPIConfiguration> mpi_conf)
        {
        m_ntrial_comm = mpi_conf;
        }
#endif

#ifdef ENABLE_MPI
    void setParticleCommunicator(std::shared_ptr<MPIConfiguration> mpi_conf)
        {
        m_particle_comm = mpi_conf;
        }
#endif

    protected:
    std::shared_ptr<CellList> m_cl; //!< Cell list
    uint3 m_last_dim;               //!< Dimensions of the cell list on the last call to update
    unsigned int m_last_nmax;       //!< Last cell list NMax value allocated in excell

    GlobalArray<unsigned int> m_excell_idx;  //!< Particle indices in expanded cells
    GlobalArray<unsigned int> m_excell_size; //!< Number of particles in each expanded cell
    Index2D m_excell_list_indexer;           //!< Indexer to access elements of the excell_idx list

    /// Autotuner for proposing moves.
    std::shared_ptr<Autotuner<1>> m_tuner_moves;

    /// Autotuner for the narrow phase.
    std::shared_ptr<Autotuner<3>> m_tuner_narrow;

    /// Autotuner for the update step group and block sizes.
    std::shared_ptr<Autotuner<1>> m_tuner_update_pdata;

    /// Autotuner for excell block_size.
    std::shared_ptr<Autotuner<1>> m_tuner_excell_block_size;

    /// Autotuner for convergence check.
    std::shared_ptr<Autotuner<1>> m_tuner_convergence;

    GlobalArray<Scalar4> m_trial_postype;           //!< New positions (and type) of particles
    GlobalArray<Scalar4> m_trial_orientation;       //!< New orientations
    GlobalArray<Scalar4> m_trial_vel;               //!< New velocities (auxilliary variables)
    GlobalArray<unsigned int> m_trial_move_type;    //!< Flags to indicate which type of move
    GlobalArray<unsigned int> m_reject_out_of_cell; //!< Flags to reject particle moves if they are
                                                    //!< out of the cell, per particle
    GlobalArray<unsigned int> m_reject; //!< Flags to reject particle moves, per particle
    GlobalArray<unsigned int>
        m_reject_out; //!< Flags to reject particle moves, per particle (temporary)

    detail::UpdateOrderGPU m_update_order; //!< Particle update order
    GlobalArray<unsigned int> m_condition; //!< Condition of convergence check

    //! For energy evaluation
    GlobalArray<Scalar> m_additive_cutoff; //!< Per-type additive cutoffs from patch potential

    GlobalArray<hpmc_counters_t> m_counters; //!< Per-device counters

    std::vector<hipStream_t> m_narrow_phase_streams; //!< Stream for narrow phase kernel, per device

    // the phase1 and phase2 kernels are for ntrial > 0
    std::vector<std::vector<hipEvent_t>>
        m_sync; //!< Synchronization event for every stream and device
    std::vector<std::vector<hipEvent_t>> m_sync_phase1; //!< Synchronization event for phase1 stream
    std::vector<std::vector<hipEvent_t>> m_sync_phase2; //!< Synchronization event for phase2 stream

#ifdef ENABLE_MPI
    std::shared_ptr<MPIConfiguration> m_ntrial_comm; //!< Communicator for MPI parallel ntrial
    std::shared_ptr<MPIConfiguration>
        m_particle_comm; //!< Communicator for MPI particle decomposition
#endif

    //! Set up excell_list
    virtual void initializeExcellMem();

    //! Set the nominal width appropriate for looped moves
    virtual void updateCellWidth();

    //! Update GPU memory hints
    virtual void updateGPUAdvice();
    };

template<class Shape>
IntegratorHPMCMonoGPU<Shape>::IntegratorHPMCMonoGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                    std::shared_ptr<CellList> cl)
    : IntegratorHPMCMono<Shape>(sysdef), m_cl(cl), m_update_order(this->m_exec_conf)
    {
    this->m_cl->setRadius(1);
    this->m_cl->setComputeTypeBody(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // with multiple GPUs, request a cell list per device
    m_cl->setPerDevice(this->m_exec_conf->allConcurrentManagedAccess());

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    m_tuner_moves.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                         this->m_exec_conf,
                                         "hpmc_moves"));

    m_tuner_update_pdata.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "hpmc_update_pdata"));

    m_tuner_excell_block_size.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "hpmc_excell_block_size"));

    // Tuning parameters for narrow phase:
    // 0: block size
    // 1: threads per particle
    // 2: overlap threads

    // Only widen the parallelism if the shape supports it, and limit parallelism to fit within the
    // warp.
    std::function<bool(const std::array<unsigned int, 3>&)> is_narrow_parameter_valid
        = [](const std::array<unsigned int, 3>& parameter) -> bool
    {
        unsigned int block_size = parameter[0];
        unsigned int threads_per_particle = parameter[1];
        unsigned int overlap_threads = parameter[2];
        return (overlap_threads == 1 || Shape::isParallel())
               && (threads_per_particle * overlap_threads <= block_size)
               && (block_size % (threads_per_particle * overlap_threads)) == 0;
    };

    const unsigned int narrow_phase_max_tpp = this->m_exec_conf->dev_prop.maxThreadsDim[2];
    m_tuner_narrow.reset(
        new Autotuner<3>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                          AutotunerBase::getTppListPow2(this->m_exec_conf, narrow_phase_max_tpp),
                          AutotunerBase::getTppListPow2(this->m_exec_conf)},
                         this->m_exec_conf,
                         "hpmc_narrow",
                         3,
                         true,
                         is_narrow_parameter_valid));

    m_tuner_convergence.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "hpmc_convergence"));

    this->m_autotuners.insert(this->m_autotuners.end(),
                              {m_tuner_moves,
                               m_tuner_update_pdata,
                               m_tuner_excell_block_size,
                               m_tuner_convergence,
                               m_tuner_narrow});

    // initialize memory
    GlobalArray<Scalar4>(1, this->m_exec_conf).swap(m_trial_postype);
    TAG_ALLOCATION(m_trial_postype);

    GlobalArray<Scalar4>(1, this->m_exec_conf).swap(m_trial_orientation);
    TAG_ALLOCATION(m_trial_orientation);

    GlobalArray<Scalar4>(1, this->m_exec_conf).swap(m_trial_vel);
    TAG_ALLOCATION(m_trial_vel);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_trial_move_type);
    TAG_ALLOCATION(m_trial_move_type);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_reject_out_of_cell);
    TAG_ALLOCATION(m_reject_out_of_cell);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_reject);
    TAG_ALLOCATION(m_reject);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_reject_out);
    TAG_ALLOCATION(m_reject_out);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_condition);
    TAG_ALLOCATION(m_condition);

#if defined(__HIP_PLATFORM_NVCC__)
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        cudaMemAdvise(m_condition.get(),
                      sizeof(unsigned int),
                      cudaMemAdviseSetPreferredLocation,
                      cudaCpuDeviceId);
        cudaMemPrefetchAsync(m_condition.get(), sizeof(unsigned int), cudaCpuDeviceId);

        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_condition.get(),
                          sizeof(unsigned int),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif

    GlobalArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);
    TAG_ALLOCATION(m_excell_size);

    GlobalArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);
    TAG_ALLOCATION(m_excell_idx);

    //! One counter per GPU, separated by an entire memory page
    unsigned int pitch
        = (unsigned int)((getpagesize() + sizeof(hpmc_counters_t) - 1) / sizeof(hpmc_counters_t));
    GlobalArray<hpmc_counters_t>(pitch, this->m_exec_conf->getNumActiveGPUs(), this->m_exec_conf)
        .swap(m_counters);
    TAG_ALLOCATION(m_counters);

#ifdef __HIP_PLATFORM_NVCC__
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_counters.get() + idev * m_counters.getPitch(),
                          sizeof(hpmc_counters_t) * m_counters.getPitch(),
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(m_counters.get() + idev * m_counters.getPitch(),
                                 sizeof(hpmc_counters_t) * m_counters.getPitch(),
                                 gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif

    m_narrow_phase_streams.resize(this->m_exec_conf->getNumActiveGPUs());
    for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
        hipStreamCreate(&m_narrow_phase_streams[idev]);
        }

    // synchronization events
    unsigned int ntypes = this->m_pdata->getNTypes();
    m_sync.resize(ntypes);
    m_sync_phase1.resize(ntypes);
    m_sync_phase2.resize(ntypes);
    for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
        {
        m_sync[itype].resize(this->m_exec_conf->getNumActiveGPUs());
        m_sync_phase1[itype].resize(this->m_exec_conf->getNumActiveGPUs());
        m_sync_phase2[itype].resize(this->m_exec_conf->getNumActiveGPUs());
        for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
            hipEventCreateWithFlags(&m_sync[itype][idev], hipEventDisableTiming);
            hipEventCreateWithFlags(&m_sync_phase1[itype][idev], hipEventDisableTiming);
            hipEventCreateWithFlags(&m_sync_phase2[itype][idev], hipEventDisableTiming);
            }
        }

#ifdef __HIP_PLATFORM_NVCC__
    // memory hint for overlap matrix
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(this->m_overlaps.get(),
                      sizeof(unsigned int) * this->m_overlaps.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);
        CHECK_CUDA_ERROR();
        }
#endif

    // patch
    GlobalArray<Scalar>(this->m_pdata->getNTypes(), this->m_exec_conf).swap(m_additive_cutoff);
    TAG_ALLOCATION(m_additive_cutoff);
    }

template<class Shape> IntegratorHPMCMonoGPU<Shape>::~IntegratorHPMCMonoGPU()
    {
    for (auto s : m_sync)
        {
        for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
            hipEventDestroy(s[idev]);
            }
        }

    for (auto s : m_sync_phase1)
        {
        for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
            hipEventDestroy(s[idev]);
            }
        }

    for (auto s : m_sync_phase2)
        {
        for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
            hipEventDestroy(s[idev]);
            }
        }

    for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
        hipStreamDestroy(m_narrow_phase_streams[idev]);
        }
    }

template<class Shape> void IntegratorHPMCMonoGPU<Shape>::updateGPUAdvice()
    {
#ifdef __HIP_PLATFORM_NVCC__
    // update memory hints
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            auto range = this->m_pdata->getGPUPartition().getRange(idev);

            unsigned int nelem = range.second - range.first;
            if (nelem == 0)
                continue;

            cudaMemAdvise(m_trial_postype.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(m_trial_postype.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);

            cudaMemAdvise(m_trial_move_type.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(m_trial_move_type.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);

            cudaMemAdvise(m_reject.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(m_reject.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);

            cudaMemAdvise(m_trial_orientation.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(m_trial_orientation.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);

            cudaMemAdvise(m_trial_vel.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(m_trial_vel.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);

            cudaMemAdvise(m_reject_out.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(m_reject_out.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);

            cudaMemAdvise(m_reject_out_of_cell.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(m_reject_out_of_cell.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);
            }
        }
#endif
    }

template<class Shape> void IntegratorHPMCMonoGPU<Shape>::update(uint64_t timestep)
    {
    IntegratorHPMC::update(timestep);

    // rng for shuffle and grid shift
    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoShift, timestep, this->m_sysdef->getSeed()),
        hoomd::Counter());

    if (this->m_pdata->getN() > 0)
        {
        // compute the width of the active region
        Scalar3 npd = this->m_pdata->getBox().getNearestPlaneDistance();
        Scalar3 ghost_fraction = this->m_nominal_width / npd;

        // check if we are below a minimum image convention box size
        // the minimum image convention comes from the global box, not the local one
        BoxDim global_box = this->m_pdata->getGlobalBox();
        Scalar3 nearest_plane_distance = global_box.getNearestPlaneDistance();

        if ((global_box.getPeriodic().x && nearest_plane_distance.x <= this->m_nominal_width * 2)
            || (global_box.getPeriodic().y && nearest_plane_distance.y <= this->m_nominal_width * 2)
            || (this->m_sysdef->getNDimensions() == 3 && global_box.getPeriodic().z
                && nearest_plane_distance.z <= this->m_nominal_width * 2))
            {
            std::ostringstream oss;

            oss << "Simulation box too small for GPU accelerated HPMC execution - increase it so "
                   "the minimum image convention may be applied."
                << std::endl;

            oss << "nominal_width = " << this->m_nominal_width << std::endl;
            if (global_box.getPeriodic().x)
                oss << "nearest_plane_distance.x=" << nearest_plane_distance.x << std::endl;
            if (global_box.getPeriodic().y)
                oss << "nearest_plane_distance.y=" << nearest_plane_distance.y << std::endl;
            if (this->m_sysdef->getNDimensions() == 3 && global_box.getPeriodic().z)
                oss << "nearest_plane_distance.z=" << nearest_plane_distance.z << std::endl;
            throw std::runtime_error(oss.str());
            }

        // update the cell list
        this->m_cl->compute(timestep);

        // if the cell list is a different size than last time, reinitialize the expanded cell list
        uint3 cur_dim = this->m_cl->getDim();
        if (m_last_dim.x != cur_dim.x || m_last_dim.y != cur_dim.y || m_last_dim.z != cur_dim.z
            || m_last_nmax != this->m_cl->getNmax())
            {
            initializeExcellMem();

            m_last_dim = cur_dim;
            m_last_nmax = this->m_cl->getNmax();
            }

        // test if we are in domain decomposition mode
        bool domain_decomposition = false;
#ifdef ENABLE_MPI
        if (this->m_sysdef->isDomainDecomposed())
            domain_decomposition = true;
#endif

        // resize some arrays
        bool resized = m_reject.getNumElements() < this->m_pdata->getMaxN();

        bool update_gpu_advice = false;

        if (resized)
            {
            m_reject.resize(this->m_pdata->getMaxN());
            m_reject_out_of_cell.resize(this->m_pdata->getMaxN());
            m_reject_out.resize(this->m_pdata->getMaxN());
            m_trial_postype.resize(this->m_pdata->getMaxN());
            m_trial_orientation.resize(this->m_pdata->getMaxN());
            m_trial_vel.resize(this->m_pdata->getMaxN());
            m_trial_move_type.resize(this->m_pdata->getMaxN());

            update_gpu_advice = true;
            }

        if (update_gpu_advice)
            updateGPUAdvice();

        m_update_order.resize(this->m_pdata->getN());

        // access the cell list data
        ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(),
                                              access_location::device,
                                              access_mode::read);
        ArrayHandle<unsigned int> d_cell_idx(this->m_cl->getIndexArray(),
                                             access_location::device,
                                             access_mode::read);
        ArrayHandle<unsigned int> d_cell_adj(this->m_cl->getCellAdjArray(),
                                             access_location::device,
                                             access_mode::read);

        // per-device cell list data
        const ArrayHandle<unsigned int>& d_cell_size_per_device
            = m_cl->getPerDevice() ? ArrayHandle<unsigned int>(m_cl->getCellSizeArrayPerDevice(),
                                                               access_location::device,
                                                               access_mode::read)
                                   : ArrayHandle<unsigned int>(GlobalArray<unsigned int>(),
                                                               access_location::device,
                                                               access_mode::read);
        const ArrayHandle<unsigned int>& d_cell_idx_per_device
            = m_cl->getPerDevice() ? ArrayHandle<unsigned int>(m_cl->getIndexArrayPerDevice(),
                                                               access_location::device,
                                                               access_mode::read)
                                   : ArrayHandle<unsigned int>(GlobalArray<unsigned int>(),
                                                               access_location::device,
                                                               access_mode::read);

        unsigned int ngpu = this->m_exec_conf->getNumActiveGPUs();
        if (ngpu > 1)
            {
            // reset per-device counters
            ArrayHandle<hpmc_counters_t> d_counters_per_device(this->m_counters,
                                                               access_location::device,
                                                               access_mode::overwrite);
            hipMemset(d_counters_per_device.data,
                      0,
                      sizeof(hpmc_counters_t) * this->m_counters.getNumElements());
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // access the parameters and interaction matrix
        auto& params = this->getParams();

        ArrayHandle<unsigned int> d_overlaps(this->m_overlaps,
                                             access_location::device,
                                             access_mode::read);

        // access the move sizes by type
        ArrayHandle<Scalar> d_d(this->m_d, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_a(this->m_a, access_location::device, access_mode::read);

        BoxDim box = this->m_pdata->getBox();

        Scalar3 ghost_width = this->m_cl->getGhostWidth();

        // randomize particle update order
        this->m_update_order.shuffle(timestep,
                                     this->m_sysdef->getSeed(),
                                     this->m_exec_conf->getRank());

        // expanded cells & neighbor list
        ArrayHandle<unsigned int> d_excell_idx(m_excell_idx,
                                               access_location::device,
                                               access_mode::overwrite);
        ArrayHandle<unsigned int> d_excell_size(m_excell_size,
                                                access_location::device,
                                                access_mode::overwrite);

        // update the expanded cells
        this->m_tuner_excell_block_size->begin();
        gpu::hpmc_excell(d_excell_idx.data,
                         d_excell_size.data,
                         m_excell_list_indexer,
                         m_cl->getPerDevice() ? d_cell_idx_per_device.data : d_cell_idx.data,
                         m_cl->getPerDevice() ? d_cell_size_per_device.data : d_cell_size.data,
                         d_cell_adj.data,
                         this->m_cl->getCellIndexer(),
                         this->m_cl->getCellListIndexer(),
                         this->m_cl->getCellAdjIndexer(),
                         this->m_exec_conf->getNumActiveGPUs(),
                         this->m_tuner_excell_block_size->getParam()[0]);
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        this->m_tuner_excell_block_size->end();

        for (unsigned int i = 0; i < this->m_nselect; i++)
            {
                { // ArrayHandle scope
                ArrayHandle<unsigned int> d_update_order_by_ptl(m_update_order.get(),
                                                                access_location::device,
                                                                access_mode::read);
                ArrayHandle<unsigned int> d_reject_out_of_cell(m_reject_out_of_cell,
                                                               access_location::device,
                                                               access_mode::overwrite);

                // access data for proposed moves
                ArrayHandle<Scalar4> d_trial_postype(m_trial_postype,
                                                     access_location::device,
                                                     access_mode::overwrite);
                ArrayHandle<Scalar4> d_trial_orientation(m_trial_orientation,
                                                         access_location::device,
                                                         access_mode::overwrite);
                ArrayHandle<Scalar4> d_trial_vel(m_trial_vel,
                                                 access_location::device,
                                                 access_mode::overwrite);
                ArrayHandle<unsigned int> d_trial_move_type(m_trial_move_type,
                                                            access_location::device,
                                                            access_mode::overwrite);

                // access the particle data
                ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(),
                                               access_location::device,
                                               access_mode::read);
                ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                                   access_location::device,
                                                   access_mode::read);
                ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                                           access_location::device,
                                           access_mode::read);

                // MC counters
                ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total,
                                                        access_location::device,
                                                        access_mode::read);
                ArrayHandle<hpmc_counters_t> d_counters_per_device(this->m_counters,
                                                                   access_location::device,
                                                                   access_mode::read);

                // fill the parameter structure for the GPU kernels
                gpu::hpmc_args_t args(d_postype.data,
                                      d_orientation.data,
                                      d_vel.data,
                                      ngpu > 1 ? d_counters_per_device.data : d_counters.data,
                                      (unsigned int)this->m_counters.getPitch(),
                                      this->m_cl->getCellIndexer(),
                                      this->m_cl->getDim(),
                                      ghost_width,
                                      this->m_pdata->getN(),
                                      this->m_pdata->getNTypes(),
                                      this->m_sysdef->getSeed(),
                                      this->m_exec_conf->getRank(),
                                      d_d.data,
                                      d_a.data,
                                      d_overlaps.data,
                                      this->m_overlap_idx,
                                      this->m_translation_move_probability,
                                      timestep,
                                      this->m_sysdef->getNDimensions(),
                                      box,
                                      i,
                                      ghost_fraction,
                                      domain_decomposition,
                                      0, // block size
                                      0, // tpp
                                      0, // overlap_threads
                                      d_reject_out_of_cell.data,
                                      d_trial_postype.data,
                                      d_trial_orientation.data,
                                      d_trial_vel.data,
                                      d_trial_move_type.data,
                                      d_update_order_by_ptl.data,
                                      d_excell_idx.data,
                                      d_excell_size.data,
                                      m_excell_list_indexer,
                                      0, // d_reject_in
                                      0, // d_reject_out
                                      this->m_exec_conf->dev_prop,
                                      this->m_pdata->getGPUPartition(),
                                      0);

                // propose trial moves, \sa gpu::kernel::hpmc_moves

                // reset acceptance results and move types
                m_tuner_moves->begin();
                args.block_size = m_tuner_moves->getParam()[0];
                gpu::hpmc_gen_moves<Shape>(args, params.data());
                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_tuner_moves->end();
                }

            bool converged = false;

                {
                // initialize reject flags
                ArrayHandle<unsigned int> d_reject_out_of_cell(m_reject_out_of_cell,
                                                               access_location::device,
                                                               access_mode::read);
                ArrayHandle<unsigned int> d_reject(m_reject,
                                                   access_location::device,
                                                   access_mode::overwrite);
                ArrayHandle<unsigned int> d_reject_out(m_reject_out,
                                                       access_location::device,
                                                       access_mode::overwrite);

                this->m_exec_conf->beginMultiGPU();
                for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                    {
                    hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);

                    auto range = this->m_pdata->getGPUPartition().getRange(idev);
                    if (range.second - range.first != 0)
                        {
                        hipMemcpyAsync(d_reject.data + range.first,
                                       d_reject_out_of_cell.data + range.first,
                                       sizeof(unsigned int) * (range.second - range.first),
                                       hipMemcpyDeviceToDevice);
                        hipMemsetAsync(d_reject_out.data + range.first,
                                       0,
                                       sizeof(unsigned int) * (range.second - range.first));
                        }
                    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    }
                this->m_exec_conf->endMultiGPU();
                }

            while (!converged)
                {
                    {
                    ArrayHandle<unsigned int> d_condition(m_condition,
                                                          access_location::device,
                                                          access_mode::overwrite);
                    // reset condition flag
                    hipMemsetAsync(d_condition.data, 0, sizeof(unsigned int));
                    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    }

                // ArrayHandle scope
                ArrayHandle<unsigned int> d_update_order_by_ptl(m_update_order.get(),
                                                                access_location::device,
                                                                access_mode::read);
                ArrayHandle<unsigned int> d_reject(m_reject,
                                                   access_location::device,
                                                   access_mode::read);
                ArrayHandle<unsigned int> d_reject_out(m_reject_out,
                                                       access_location::device,
                                                       access_mode::overwrite);
                ArrayHandle<unsigned int> d_reject_out_of_cell(m_reject_out_of_cell,
                                                               access_location::device,
                                                               access_mode::read);

                // access data for proposed moves
                ArrayHandle<Scalar4> d_trial_postype(m_trial_postype,
                                                     access_location::device,
                                                     access_mode::read);
                ArrayHandle<Scalar4> d_trial_orientation(m_trial_orientation,
                                                         access_location::device,
                                                         access_mode::read);
                ArrayHandle<Scalar4> d_trial_vel(m_trial_vel,
                                                 access_location::device,
                                                 access_mode::read);
                ArrayHandle<unsigned int> d_trial_move_type(m_trial_move_type,
                                                            access_location::device,
                                                            access_mode::read);

                // access the particle data
                ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(),
                                               access_location::device,
                                               access_mode::read);
                ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                                   access_location::device,
                                                   access_mode::read);
                ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                                           access_location::device,
                                           access_mode::read);
                ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(),
                                                access_location::device,
                                                access_mode::read);

                // MC counters
                ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total,
                                                        access_location::device,
                                                        access_mode::readwrite);
                ArrayHandle<hpmc_counters_t> d_counters_per_device(this->m_counters,
                                                                   access_location::device,
                                                                   access_mode::readwrite);

                // fill the parameter structure for the GPU kernels
                gpu::hpmc_args_t args(d_postype.data,
                                      d_orientation.data,
                                      d_vel.data,
                                      ngpu > 1 ? d_counters_per_device.data : d_counters.data,
                                      (unsigned int)this->m_counters.getPitch(),
                                      this->m_cl->getCellIndexer(),
                                      this->m_cl->getDim(),
                                      ghost_width,
                                      this->m_pdata->getN(),
                                      this->m_pdata->getNTypes(),
                                      this->m_sysdef->getSeed(),
                                      this->m_exec_conf->getRank(),
                                      d_d.data,
                                      d_a.data,
                                      d_overlaps.data,
                                      this->m_overlap_idx,
                                      this->m_translation_move_probability,
                                      timestep,
                                      this->m_sysdef->getNDimensions(),
                                      box,
                                      i,
                                      ghost_fraction,
                                      domain_decomposition,
                                      0, // block size
                                      0, // tpp
                                      0, // overlap threads
                                      d_reject_out_of_cell.data,
                                      d_trial_postype.data,
                                      d_trial_orientation.data,
                                      d_trial_vel.data,
                                      d_trial_move_type.data,
                                      d_update_order_by_ptl.data,
                                      d_excell_idx.data,
                                      d_excell_size.data,
                                      m_excell_list_indexer,
                                      d_reject.data,
                                      d_reject_out.data,
                                      this->m_exec_conf->dev_prop,
                                      this->m_pdata->getGPUPartition(),
                                      &m_narrow_phase_streams.front());

                /*
                 *  check overlaps, new configuration simultaneously against the old and the new
                 * configuration
                 */

                this->m_exec_conf->beginMultiGPU();

                m_tuner_narrow->begin();
                auto param = m_tuner_narrow->getParam();
                args.block_size = param[0];
                args.tpp = param[1];
                args.overlap_threads = param[2];
                gpu::hpmc_narrow_phase<Shape>(args, params.data());
                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_tuner_narrow->end();

                    {
                    ArrayHandle<unsigned int> d_reject_out_of_cell(m_reject_out_of_cell,
                                                                   access_location::device,
                                                                   access_mode::read);
                    ArrayHandle<unsigned int> d_reject(m_reject,
                                                       access_location::device,
                                                       access_mode::readwrite);
                    ArrayHandle<unsigned int> d_reject_out(m_reject_out,
                                                           access_location::device,
                                                           access_mode::readwrite);
                    ArrayHandle<unsigned int> d_condition(m_condition,
                                                          access_location::device,
                                                          access_mode::readwrite);
                    ArrayHandle<unsigned int> d_trial_move_type(m_trial_move_type,
                                                                access_location::device,
                                                                access_mode::read);

                    this->m_exec_conf->beginMultiGPU();
                    m_tuner_convergence->begin();
                    gpu::hpmc_check_convergence(d_trial_move_type.data,
                                                d_reject_out_of_cell.data,
                                                d_reject.data,
                                                d_reject_out.data,
                                                d_condition.data,
                                                this->m_pdata->getGPUPartition(),
                                                m_tuner_convergence->getParam()[0]);
                    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    m_tuner_convergence->end();
                    this->m_exec_conf->endMultiGPU();
                    }

                // flip reject flags
                std::swap(m_reject, m_reject_out);

                    {
                    ArrayHandle<unsigned int> h_condition(m_condition,
                                                          access_location::host,
                                                          access_mode::read);
                    if (*h_condition.data == 0)
                        converged = true;
                    }
                } // end while (!converged)

                {
                // access data for proposed moves
                ArrayHandle<Scalar4> d_trial_postype(m_trial_postype,
                                                     access_location::device,
                                                     access_mode::read);
                ArrayHandle<Scalar4> d_trial_orientation(m_trial_orientation,
                                                         access_location::device,
                                                         access_mode::read);
                ArrayHandle<Scalar4> d_trial_vel(m_trial_vel,
                                                 access_location::device,
                                                 access_mode::read);
                ArrayHandle<unsigned int> d_trial_move_type(m_trial_move_type,
                                                            access_location::device,
                                                            access_mode::read);

                // access the particle data
                ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(),
                                               access_location::device,
                                               access_mode::readwrite);
                ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                                   access_location::device,
                                                   access_mode::readwrite);
                ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                                           access_location::device,
                                           access_mode::readwrite);

                // MC counters
                ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total,
                                                        access_location::device,
                                                        access_mode::readwrite);
                ArrayHandle<hpmc_counters_t> d_counters_per_device(this->m_counters,
                                                                   access_location::device,
                                                                   access_mode::readwrite);

                // flags
                ArrayHandle<unsigned int> d_reject(m_reject,
                                                   access_location::device,
                                                   access_mode::read);

                // Update the particle data and statistics
                this->m_exec_conf->beginMultiGPU();
                m_tuner_update_pdata->begin();
                gpu::hpmc_update_args_t args(d_postype.data,
                                             d_orientation.data,
                                             d_vel.data,
                                             ngpu > 1 ? d_counters_per_device.data
                                                      : d_counters.data,
                                             (unsigned int)this->m_counters.getPitch(),
                                             this->m_pdata->getGPUPartition(),
                                             d_trial_postype.data,
                                             d_trial_orientation.data,
                                             d_trial_vel.data,
                                             d_trial_move_type.data,
                                             d_reject.data,
                                             m_tuner_update_pdata->getParam()[0]);
                gpu::hpmc_update_pdata<Shape>(args, params.data());
                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_tuner_update_pdata->end();
                this->m_exec_conf->endMultiGPU();
                }
            } // end loop over nselect
        }

    // shift particles
    Scalar3 shift = make_scalar3(0, 0, 0);
    hoomd::UniformDistribution<Scalar> uniform(-this->m_nominal_width / Scalar(2.0),
                                               this->m_nominal_width / Scalar(2.0));
    shift.x = uniform(rng);
    shift.y = uniform(rng);
    if (this->m_sysdef->getNDimensions() == 3)
        {
        shift.z = uniform(rng);
        }

    if (this->m_pdata->getN() > 0)
        {
        BoxDim box = this->m_pdata->getBox();

        // access the particle data
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::readwrite);
        ArrayHandle<int3> d_image(this->m_pdata->getImages(),
                                  access_location::device,
                                  access_mode::readwrite);

        gpu::hpmc_shift(d_postype.data, d_image.data, this->m_pdata->getN(), box, shift, 128);
        }
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // update the particle data origin
    this->m_pdata->translateOrigin(shift);

    this->communicate(true);
    // all particle have been moved, the aabb tree is now invalid
    this->m_aabb_tree_invalid = true;

    // set current MPS value
    hpmc_counters_t run_counters = this->getCounters(1);
    double cur_time = double(this->m_clock.getTime()) / Scalar(1e9);
    this->m_mps = double(run_counters.getNMoves()) / cur_time;
    }

template<class Shape> void IntegratorHPMCMonoGPU<Shape>::initializeExcellMem()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc resizing expanded cells" << std::endl;

    // get the current cell dimensions
    unsigned int num_cells = this->m_cl->getCellIndexer().getNumElements();
    unsigned int num_adj = this->m_cl->getCellAdjIndexer().getW();
    unsigned int n_cell_list
        = this->m_cl->getPerDevice() ? this->m_exec_conf->getNumActiveGPUs() : 1;
    unsigned int num_max = this->m_cl->getNmax() * n_cell_list;

    // make the excell dimensions the same, but with room for Nmax*Nadj in each cell
    m_excell_list_indexer = Index2D(num_max * num_adj, num_cells);

    // reallocate memory
    m_excell_idx.resize(m_excell_list_indexer.getNumElements());
    m_excell_size.resize(num_cells);

#if defined(__HIP_PLATFORM_NVCC__) \
    && 0 // excell is currently not multi-GPU optimized, let the CUDA driver figure this out
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_excell_idx.get(),
                          sizeof(unsigned int) * m_excell_idx.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_excell_size.get(),
                          sizeof(unsigned int) * m_excell_size.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            CHECK_CUDA_ERROR();
            }
        }
#endif
    }

template<class Shape> void IntegratorHPMCMonoGPU<Shape>::updateCellWidth()
    {
    // call base class method
    IntegratorHPMCMono<Shape>::updateCellWidth();

    // update the cell list
    this->m_cl->setNominalWidth(this->m_nominal_width);

#ifdef __HIP_PLATFORM_NVCC__
    // set memory hints
    cudaMemAdvise(this->m_params.data(),
                  this->m_params.size() * sizeof(typename Shape::param_type),
                  cudaMemAdviseSetReadMostly,
                  0);
    CHECK_CUDA_ERROR();
#endif

    // sync up so we can access the parameters
    hipDeviceSynchronize();

    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        // attach nested memory regions
        this->m_params[i].set_memory_hint();
        CHECK_CUDA_ERROR();
        }
    }

namespace detail
    {
//! Export this hpmc integrator to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template<class Shape>
void export_IntegratorHPMCMonoGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<IntegratorHPMCMonoGPU<Shape>,
                     IntegratorHPMCMono<Shape>,
                     std::shared_ptr<IntegratorHPMCMonoGPU<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<CellList>>())
#ifdef ENABLE_MPI
        .def("setParticleCommunicator", &IntegratorHPMCMonoGPU<Shape>::setParticleCommunicator)
#endif
        ;
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd

#endif // ENABLE_HIP
