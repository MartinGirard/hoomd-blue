// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_TWOSTEPCONSTANTVOLUME_H
#define HOOMD_TWOSTEPCONSTANTVOLUME_H

#include "ComputeThermo.h"
#include "IntegrationMethodTwoStep.h"
#include "Thermostat.h"
#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>
namespace hoomd::md
    {

class PYBIND11_EXPORT TwoStepConstantVolume : public IntegrationMethodTwoStep
    {
    public:
    TwoStepConstantVolume(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group,
                          std::shared_ptr<Thermostat> thermostat)
        : IntegrationMethodTwoStep(sysdef, group), m_thermostat(thermostat)
        {
        }

    virtual ~TwoStepConstantVolume() { }

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);
    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    void setThermostat(std::shared_ptr<Thermostat> thermostat)
        {
        m_thermostat = thermostat;
        }

    void setLimit(std::shared_ptr<Variant>& limit)
        {
        m_limit = limit;
        }

    [[nodiscard]] auto getLimit() const
        {
        return m_limit;
        }

    auto getKernelLimitValues(uint64_t timestep)
        {
        const auto use_limit = static_cast<bool>(m_limit);
        Scalar maximum_displacement = use_limit ? m_limit->operator()(timestep) : 0.0;
        return std::make_pair(use_limit, maximum_displacement);
        }

    //! Get needed pdata flags
    /*! in anisotropic mode, we need the rotational kinetic energy
     */
    virtual PDataFlags getRequestedPDataFlags()
        {
        PDataFlags flags;
        if (m_aniso)
            {
            flags[pdata_flag::rotational_kinetic_energy] = 1;
            }
        return flags;
        }

    protected:
    std::shared_ptr<Thermostat> m_thermostat;
    std::shared_ptr<Variant> m_limit;
    };

    } // namespace hoomd::md

#endif // HOOMD_TWOSTEPCONSTANTVOLUME_H