// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_EXTERNAL_MAGNETIC_FIELD_H__
#define __EVALUATOR_EXTERNAL_MAGNETIC_FIELD_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include <math.h>

/*! \file EvaluatorExternalMagneticField.h
    \brief Defines the external potential evaluator to induce a magnetic field
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
//! Class for evaluating an magnetic field
/*! <b>General Overview</b>
    The external potential \f$V(\theta) \f$ is implemented using the following formula:

    \f[
    V(\theta}) = - \vec{B} \cdot \vec{n}_i(\theta)
    \f]

    where \f$B\f$ is the strength of the magnetic field and \f$\vec{n}_i\f$ is the magnetic moment of particle i.
*/
class EvaluatorExternalMagneticField
    {
    public:
    //! type of parameters this external potential accepts
    struct param_type
        {
        Scalar3 B;
        Scalar3 mu;

#ifndef __HIPCC__
        param_type() : B(make_scalar3(0, 0, 0)), mu(make_scalar3(0, 0, 0)) { }

        param_type(pybind11::dict params)
            {
            B = params["B"].cast<Scalar3>();
            mu = params["mu"].cast<Scalar3>();
            }

        param_type(Scalar3 B_, Scalar3 mu_) : B(B_), mu(mu_) { }


        pybind11::dict toPython()
            {
            pybind11::dict d;
            d["B"] = B;
            d["mu"] = mu;
            return d;
            }
#endif // ifndef __HIPCC__
        } __attribute__((aligned(16)));

    typedef void* field_type;

    //! Constructs the constraint evaluator
    /*! \param X position of particle
        \param box box dimensions
        \param params per-type parameters of external potential
    */
    DEVICE EvaluatorExternalMagneticField(Scalar3 X,
                                          const BoxDim& box,
                                          const param_type& params,
                                          const field_type& field)
        : m_quat(X), m_B(params.B), m_mu(params.mu)
        {
        }

    //! ExternalMagneticField needs charges
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge value
    /*! \param qi Charge of particle i
     */
    DEVICE void setCharge(Scalar qi) { }

    //! Declares additional virial contributions are needed for the external field
    /*! No contribution
     */
    DEVICE static bool requestFieldVirialTerm()
        {
        return false;
        }

    //! Evaluate the force, energy and virial
    /*! \param F force vector
        \param energy value of the energy
        \param virial array of six scalars for the upper triangular virial tensor
    */
    DEVICE void evalForceEnergyAndVirial(Scalar3& F, Scalar3& T, Scalar& energy, Scalar* virial)
        {
        
	Scalar3 dir = m_mu * u_i;

	T = cross(dir, m_B);

	energy = - dot(dir,m_B);

        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("b_field");
        }
#endif

    protected:
    Scalar4 m_quat; //!< particle position
    Scalar3 m_mu;   //!< particle charge
    Scalar3 m_B;   //!< the field vector
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_EXTERNAL_LAMELLAR_H__
