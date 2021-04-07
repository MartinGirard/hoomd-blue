// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_GYROID_H__
#define __MANIFOLD_CLASS_GYROID_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

/*! \file ManifoldGyroid.h
    \brief Defines the manifold class for the Gyroid minimal surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for constructing the Gyroid minimal surface
/*! <b>General Overview</b>

    ManifoldGyroid is a low level computation class that computes the distance and normal vector to the Gyroid surface.

    <b>Gyroid specifics</b>

    ManifoldGyroid constructs the surface:
    R^2 = (x-P_x)^2 + (y-P_y)^2 + (z-P_z)^2

    These are the parameters:
    \a Nx The number of unitcells in x-direction
    \a Ny The number of unitcells in y-direction
    \a Nz The number of unitcells in z-direction
    \a epsilon Defines the specific constant mean curvture companion

*/

class ManifoldGyroid
    {
    public:
        //! Constructs the manifold class
         /* \param _Nx The number of unitcells in x-direction
            \param _Ny The number of unitcells in y-direction
            \param _Nz The number of unitcells in z-direction
            \param _epsilon Defines the specific constant mean curvture companion
        */
        DEVICE ManifoldGyroid(const int _Nx, const int _Ny, const int _Nz, const Scalar _epsilon)
            : Nx(_Nx), Ny(_Ny), Nz(_Nz), Lx(0), Ly(0), Lz(0), epsilon(_epsilon)
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(const Scalar3& point)
        {
            return fast::sin(Lx*point.x)*fast::cos(Ly*point.y) + fast::sin(Ly*point.y)*fast::cos(Lz*point.z) + fast::sin(Lz*point.z)*fast::cos(Lx*point.x) - epsilon;
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Gyroid surface at input point
        */

        DEVICE Scalar3 derivative(const Scalar3& point)
        {
            Scalar cx,sx;
            fast::sincos(Lx*point.x,sx,cx);
            Scalar cy,sy;
            fast::sincos(Ly*point.y,sy,cy);
            Scalar cz,sz;
            fast::sincos(Lz*point.z,sz,cz);

            return make_scalar3(Lx*(cx*cy - sz*sx),Ly*(cy*cz - sx*sy), Lz*(cz*cx - sy*sz));
        }

        DEVICE bool adjust_to_box(const BoxDim& box)
        {
            Scalar3 box_length = box.getHi() - box.getLo();

            Lx = 2*M_PI*Nx/box_length.x;
            Ly = 2*M_PI*Ny/box_length.y;
            Lz = 2*M_PI*Nz/box_length.z;

            return true; //Gyroid surface is adjusted to box and, therefore, is always accepted
        }

        pybind11::dict getDict()
        {
            pybind11::dict v;
            v["N"] = pybind11::make_tuple(Nx, Ny, Nz);
            v["epsilon"] = epsilon;
            return v;
        }

        static unsigned int dimension()
            {
            return 2;
            }

    protected:
        int Nx;
        int Ny;
        int Nz;
        Scalar Lx;
        Scalar Ly;
        Scalar Lz;
        Scalar epsilon;
    };

//! Exports the Gyroid manifold class to python
void export_ManifoldGyroid(pybind11::module& m);

#endif // __MANIFOLD_CLASS_GYROID_H__
