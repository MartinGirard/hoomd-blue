// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_SPHERE_H__
#define __MANIFOLD_CLASS_SPHERE_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

/*! \file ManifoldSphere.h
    \brief Defines the manifold class for the Sphere surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for constructing the Sphere surface
/*! <b>General Overview</b>

    ManifoldSphere is a low level computation class that computes the distance and normal vector to the Sphere surface.

    <b>Sphere specifics</b>

    ManifoldSphere constructs the surface:
    R^2 = (x-P_x)^2 + (y-P_y)^2 + (z-P_z)^2

    These are the parameters:
    - \a P_x = center position of the sphere in x-direction;
    - \a P_y = center position of the sphere in y-direction;
    - \a P_z = center position of the sphere in z-direction;
    - \a R = radius of the sphere;

*/

class ManifoldSphere
    {
    public:
        //! Constructs the manifold class
        /*! \param _Px center position in x-direction
            \param _Py center position in y-direction
            \param _Pz center position in z-direction
            \param _R radius
        */
        DEVICE ManifoldSphere(const Scalar _R, const Scalar3 _P)
            : Px(_P.x), Py(_P.y), Pz(_P.z), R_sq(_R*_R)
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(const Scalar3& point)
        {
            return  (point.x - Px)*(point.x - Px) + (point.y - Py)*(point.y - Py) + (point.z - Pz)*(point.z - Pz) - R_sq;
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Sphere surface at input point
        */

        DEVICE Scalar3 derivative(const Scalar3& point)
        {
            return make_scalar3(2*(point.x - Px), 2*(point.y - Py), 2*(point.z - Pz));
        }


        DEVICE bool check_fit_to_box(const BoxDim& box)
        {
            Scalar3 lo = box.getLo();
            Scalar3 hi = box.getHi();
            Scalar sqR = fast::sqrt(R_sq);
            if (Px + sqR > hi.x || Px - sqR < lo.x ||
                Py + sqR > hi.y || Py - sqR < lo.y ||
                Pz + sqR > hi.z || Pz - sqR < lo.z)
                {
                return false; // Sphere does not fit inside box
                }
                else
                { 
                return true;
                }
        }

        pybind11::dict getDict()
        {
            pybind11::dict v;
            v["r"] = sqrt(R_sq);
            v["P"] = pybind11::make_tuple(Px, Py, Pz);
            return v;
        }

        Scalar getR(){ return sqrt(R_sq);};

        pybind11::tuple getP(){ return pybind11::make_tuple(Px, Py, Pz);}

        static unsigned int dimension()
            {
            return 2;
            }

    protected:
        Scalar Px;
        Scalar Py;
        Scalar Pz;
        Scalar R_sq;
    };

//! Exports the Sphere manifold class to python
void export_ManifoldSphere(pybind11::module& m);

#endif // __MANIFOLD_CLASS_SPHERE_H__
