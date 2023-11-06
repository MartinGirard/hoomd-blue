// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_PAIRFEP_H__
#define __POTENTIAL_PAIRFEP_H__

#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "NeighborList.h"
#include "hoomd/ForceCompute.h"
//#include "hoomd/GSDShapeSpecWriter.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/managed_allocator.h"
#include "hoomd/md/EvaluatorPairLJ.h"
#include "PotentialPair.h"

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

/*! \file PotentialPair.h
    \brief Defines the template class for standard pair potentials
    \details The heart of the code that computes pair potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {

template<class evaluator> class PotentialPairFEP : public PotentialPair<evaluator>
    {
    public:

    //! Construct the pair potential
    PotentialPairFEP(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<NeighborList> nlist,
                     unsigned int type_override,
                     Scalar charge_override);
    //! Destructor
    virtual ~PotentialPairFEP();

    std::string get_type_override() const{
        if(m_type_override != -1)
            return this->m_pdata->getNameByType(m_type_override);
        else
            return std::string("");
    }

    void set_type_override(std::string type){
        if(type == ""){
            m_type_override = -1;
        }else{
            m_type_override = this->m_pdata->getTypeByName(type);
        }
    }

    float get_charge_override() const{
        return m_charge_override;
    }

    void set_charge_override(Scalar charge){
        m_charge_override = charge;
    }

    protected:
    unsigned int m_type_override;
    Scalar m_charge_override;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! Compute the long-range corrections to energy and pressure to account for truncating the pair
    //! potentials
    virtual void computeTailCorrection()
        {} // end void computeTailCorrection()

    }; // end class PotentialPair

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/
template<class evaluator>
PotentialPairFEP<evaluator>::PotentialPairFEP(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<NeighborList> nlist,
                                        unsigned int type_override,
                                        Scalar charge_override)
    : PotentialPair<evaluator>(sysdef, nlist), m_type_override(type_override), m_charge_override(charge_override)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing PotentialPairFEP<" << evaluator::getName() << ">"
                                << std::endl;
    }

template<class evaluator> PotentialPairFEP<evaluator>::~PotentialPairFEP()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying PotentialPairFEP<" << evaluator::getName() << ">"
                                << std::endl;
    }

/*! \post The FEP is computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template<class evaluator> void PotentialPairFEP<evaluator>::computeForces(uint64_t timestep) {
        // start by updating the neighborlist
        this->m_nlist->compute(timestep);

        // depending on the neighborlist settings, we can take advantage of newton's third law
        // to reduce computations at the cost of memory access complexity: set that flag now
        bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
        if (third_law) {
            throw std::runtime_error(
                    "FEP implementation does not support half neighborlists");
        }

        // access the neighbor list, particle data, and system box
        ArrayHandle<unsigned int> h_n_neigh(this->m_nlist->getNNeighArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_nlist(this->m_nlist->getNListArray(),
                                          access_location::host,
                                          access_mode::read);
        //     Index2D nli = m_nlist->getNListIndexer();
        ArrayHandle<size_t> h_head_list(this->m_nlist->getHeadList(),
                                        access_location::host,
                                        access_mode::read);

        ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(),
                                   access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(),
                                     access_location::host, access_mode::read);

        // force arrays
        ArrayHandle<Scalar4> h_force(this->m_force, access_location::host,
                                     access_mode::overwrite);

        const BoxDim box = this->m_pdata->getGlobalBox();
        ArrayHandle<Scalar> h_ronsq(this->m_ronsq, access_location::host,
                                    access_mode::read);
        ArrayHandle<Scalar> h_rcutsq(this->m_rcutsq, access_location::host,
                                     access_mode::read);

        // need to start from a zero force, energy and virial
        memset((void *) h_force.data, 0,
               sizeof(Scalar4) * this->m_force.getNumElements());

        // for each particle
        for (int i = 0; i < (int) this->m_pdata->getN(); i++) {
            // access the particle's position and type (MEM TRANSFER: 4 scalars)
            Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y,
                                      h_pos.data[i].z);
            if(m_type_override != -1) {
                unsigned int typei = m_type_override;//__scalar_as_int(h_pos.data[i].w);
            }
            else{
                unsigned int typei = __scalar_as_int(h_pos.data[i].w);
            }

            // sanity check
            assert(typei < this->m_pdata->getNTypes());

            // access charge (if needed)
            Scalar qi = Scalar(0.0);
            if (evaluator::needsCharge())
                qi = m_charge_override == INFINITY? h_charge.data[i] : m_charge_override;

            // initialize current particle force, potential energy, and virial to 0
            Scalar3 fi = make_scalar3(0, 0, 0);
            Scalar pei = 0.0;


            // loop over all of the neighbors of this particle
            const size_t myHead = h_head_list.data[i];
            const unsigned int size = (unsigned int) h_n_neigh.data[i];
            for (unsigned int k = 0; k < size; k++) {
                // access the index of this neighbor (MEM TRANSFER: 1 scalar)
                unsigned int j = h_nlist.data[myHead + k];
                assert(j < this->m_pdata->getN() + this->m_pdata->getNGhosts());

                // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
                Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y,
                                          h_pos.data[j].z);
                Scalar3 dx = pi - pj;

                // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
                unsigned int typej = __scalar_as_int(h_pos.data[j].w);
                assert(typej < this->m_pdata->getNTypes());

                // access charge (if needed)
                Scalar qj = Scalar(0.0);

                if (evaluator::needsCharge())
                    qj = h_charge.data[j];

                // apply periodic boundary conditions
                dx = box.minImage(dx);

                // calculate r_ij squared (FLOPS: 5)
                Scalar rsq = dot(dx, dx);

                // get parameters for this type pair
                unsigned int typpair_idx = this->m_typpair_idx(typei, typej);
                const auto &param = this->m_params[typpair_idx];
                Scalar rcutsq = h_rcutsq.data[typpair_idx];
                Scalar ronsq = Scalar(0.0);
                if (this->m_shift_mode == this->xplor)
                    ronsq = h_ronsq.data[typpair_idx];

                // design specifies that energies are shifted if
                // 1) shift mode is set to shift
                // or 2) shift mode is explor and ron > rcut
                bool energy_shift = false;
                if (this->m_shift_mode == this->shift)
                    energy_shift = true;
                else if (this->m_shift_mode == this->xplor) {
                    if (ronsq > rcutsq)
                        energy_shift = true;
                }

                // compute the force and potential energy
                Scalar force_divr = Scalar(0.0);
                Scalar pair_eng = Scalar(0.0);
                evaluator eval(rsq, rcutsq, param);
                if (evaluator::needsCharge())
                    eval.setCharge(qi, qj);

                bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng,
                                                         energy_shift);

                if (evaluated) {
                    // modify the potential for xplor shifting
                    if (this->m_shift_mode == this->xplor) {
                        if (rsq >= ronsq && rsq < rcutsq) {
                            // Implement XPLOR smoothing (FLOPS: 16)
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
                            // note: I'm not sure why the minus sign needs to be there: my notes have a
                            // + But this is verified correct via plotting
                            force_divr = s * old_force_divr -
                                         ds_dr_divr * old_pair_eng;
                        }
                    }
                    // add the force, potential energy and virial to the particle i
                    // (FLOPS: 8)
                    fi += dx * force_divr;
                    pei += pair_eng;

                    // add the force to particle j if we are using the third law (MEM TRANSFER: 10
                    // scalars / FLOPS: 8) only add force to local particles
                }

                // finally, increment the force, potential energy and virial for particle i
                unsigned int mem_idx = i;
                h_force.data[mem_idx].x += fi.x;
                h_force.data[mem_idx].y += fi.y;
                h_force.data[mem_idx].z += fi.z;
                h_force.data[mem_idx].w += pei;
            }
        }
    }

namespace detail
    {
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialPairFEP(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialPairFEP<T>, PotentialPairFEP<T>, std::shared_ptr<PotentialPairFEP<T>>>
        potentialpairFEP(m, name.c_str());
    potentialpairFEP
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, unsigned int, Scalar>())
        .def_property("charge_value", &PotentialPairFEP<T>::get_charge_override, &PotentialPairFEP<T>::set_charge_override)
        .def_property("type_override", &PotentialPairFEP<T>::get_type_override, &PotentialPairFEP<T>::set_type_override);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __POTENTIAL_PAIR_H__
