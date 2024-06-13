// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CosineChannelGeometry.cc
 * \brief Export function MPCD cosine channel geometry.
 */

#include "CosineChannelGeometry.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
void export_CosineChannelGeometry(pybind11::module& m)
    {
    pybind11::class_<CosineChannelGeometry, std::shared_ptr<CosineChannelGeometry>>(
        m,
        CosineChannelGeometry::getName().c_str())
        .def(pybind11::init<Scalar, Scalar, Scalar, unsigned int, bool>())
        .def_property_readonly("amplitude", &CosineChannelGeometry::getAmplitude)
        .def_property_readonly("hw_narrow", &CosineChannelGeometry::getHnarrow)
        .def_property_readonly("repetitions", &CosineChannelGeometry::getRepetitions)
        .def_property_readonly("no_slip", &CosineChannelGeometry::getNoSlip);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
