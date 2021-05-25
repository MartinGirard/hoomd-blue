# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features

"""Compute system properties."""

from hoomd.md import _md
from hoomd.operation import Compute
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import log
import hoomd


class _Thermo(Compute):

    def __init__(self, filter):
        self._filter = filter


class ThermodynamicQuantities(_Thermo):
    """Compute thermodynamic properties of a group of particles.

    Args:
        filter (``hoomd.filter``): Particle filter to compute thermodynamic
            properties for.

    :py:class:`ThermodynamicQuantities` acts on a given group of particles and
    calculates thermodynamic properties of those particles when requested. All
    specified :py:class:`ThermodynamicQuantities` objects can be added to a
    logger for logging during a simulation, see :py:class:`hoomd.logging.Logger`
    for more details.

    Examples::

        f = filter.Type('A')
        compute.ThermodynamicQuantities(filter=f)
    """

    def __init__(self, filter):
        super().__init__(filter)

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            thermo_cls = _md.ComputeThermo
        else:
            thermo_cls = _md.ComputeThermoGPU
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = thermo_cls(self._simulation.state._cpp_sys_def, group)
        super()._attach()

    @log
    def kinetic_temperature(self):
        r""":math:`kT_k`, instantaneous thermal energy of the group :math:`[energy]`.

        Calculated as:

          .. math::

            kT_k = 2 \cdot \frac{K}{N_{\mathrm{dof}}}
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.kinetic_temperature
        else:
            return None

    @log
    def pressure(self):
        r""":math:`P`, instantaneous pressure of the group :math:`[pressure]`.

        Calculated as:

        .. math::

            P = \frac{ 2 \cdot K_{\mathrm{trans}} + W }{D \cdot V},

        where :math:`D` is the dimensionality of the system, :math:`V` is the
        total volume of the simulation box (or area in 2D), and :math:`W` is
        calculated as:

        .. math::

            W = \frac{1}{2} \sum_{i \in \mathrm{filter}} \sum_{j}
            \vec{F}_{ij} \cdot \vec{r_{ij}} + \sum_{k} \vec{F}_{k} \cdot
            \vec{r_{k}},

        where :math:`i` and :math:`j` are particle tags, :math:`\vec{F}_{ij}`
        are pairwise forces between particles and :math:`\vec{F}_k` are forces
        due to explicit constraints, implicit rigid body constraints, external
        walls, and fields.
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.pressure
        else:
            return None

    @log(category='sequence')
    def pressure_tensor(self):
        r"""Instantaneous pressure tensor of the group :math:`[pressure]`.

        (:math:`P_{xx}`, :math:`P_{xy}`, :math:`P_{xz}`, :math:`P_{yy}`,
        :math:`P_{yz}`, :math:`P_{zz}`). calculated as:

          .. math::

              P_{ij} = \left[  \sum_{k \in \mathrm{filter}} m_k v_{k,i}
              v_{k,j} + \sum_{k \in \mathrm{filter}} \sum_{l} \frac{1}{2}
              \left(\vec{r}_{kl,i} \vec{F}_{kl,j} + \vec{r}_{kl,j}
              \vec{F}_{kl, i} \right) \right]/V

        where :math:`V` is the total simulation box volume (or area in 2D).
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.pressure_tensor
        else:
            return None

    @log
    def kinetic_energy(self):
        r""":math:`K`, total kinetic energy of particles in the group :math:`[energy]`.

        .. math::

            K = K_{\mathrm{rot}} + K_{\mathrm{trans}}

        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.kinetic_energy
        else:
            return None

    @log
    def translational_kinetic_energy(self):
        r""":math:`K_{\mathrm{trans}}`.

        Translational kinetic energy of all particles in the group :math:`[energy]`.

        .. math::

            K_{\mathrm{trans}} = \frac{1}{2}\sum_{i \in \mathrm{filter}}
            m_i|\vec{v}_i|^2

        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.translational_kinetic_energy
        else:
            return None

    @log
    def rotational_kinetic_energy(self):
        r""":math:`K_{\mathrm{rot}}`.

        Rotational kinetic energy of all particles in the group :math:`[energy]`.

        Calculated as:

        .. math::

            K_{\mathrm{rot}} = \frac{1}{2} \sum_{i \in \mathrm{filter}}
            \frac{L_{x,i}^2}{I_{x,i}} + \frac{L_{y,i}^2}{I_{y,i}} +
            \frac{L_{z,i}^2}{I_{z,i}},

        where :math:`I` is the moment of inertia and :math:`L` is the angular
        momentum in the (diagonal) reference frame of the particle.
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.rotational_kinetic_energy
        else:
            return None

    @log
    def potential_energy(self):
        r""":math:`U`.

        Potential energy that the group contributes to the entire system
        [energy].

        The potential energy is calculated as a sum of per-particle energy
        contributions:

        .. math::

            U = \sum_{i \in \mathrm{filter}} U_i,

        where :math:`U_i` is defined as:

        .. math::

            U_i = U_{\mathrm{pair}, i} + U_{\mathrm{bond}, i} +
            U_{\mathrm{angle}, i} + U_{\mathrm{dihedral}, i} +
            U_{\mathrm{improper}, i} + U_{\mathrm{external}, i} +
            U_{\mathrm{other}, i}

        and each term on the RHS is calculated as:

        .. math::

            U_{\mathrm{pair}, i} &= \frac{1}{2} \sum_j V_{\mathrm{pair}, ij}

            U_{\mathrm{bond}, i} &= \frac{1}{2} \sum_{(j, k) \in
            \mathrm{bonds}} V_{\mathrm{bond}, jk}

            U_{\mathrm{angle}, i} &= \frac{1}{3} \sum_{(j, k, l) \in
            \mathrm{angles}} V_{\mathrm{angle}, jkl}

            U_{\mathrm{dihedral}, i} &= \frac{1}{4} \sum_{(j, k, l, m) \in
            \mathrm{dihedrals}} V_{\mathrm{dihedral}, jklm}

            U_{\mathrm{improper}, i} &= \frac{1}{4} \sum_{(j, k, l, m) \in
            \mathrm{impropers}} V_{\mathrm{improper}, jklm}

        In each summation above, the indices go over all particles and we only
        use terms where one of the summation indices (:math:`j`, :math:`k`,
        :math:`l`, or :math:`m`) is equal to :math:`i`. External and other
        potentials are summed similar to the other terms using per-particle
        contributions.
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.potential_energy
        else:
            return None

    @log
    def degrees_of_freedom(self):
        r""":math:`N_{\mathrm{dof}}`.

        Number of degrees of freedom given to the group by its integration
        method.

        Calculated as:

        .. math::

            N_{\mathrm{dof}} = N_{\mathrm{dof, trans}}
                                + N_{\mathrm{dof, rot}}
        """
        if self._attached:
            return self._cpp_obj.degrees_of_freedom
        else:
            return None

    @log
    def translational_degrees_of_freedom(self):
        r""":math:`N_{\mathrm{dof, trans}}`.

        Number of translational degrees of freedom given to the group by its
        integration method.

        When using a single integration method that is momentum conserving and
        operates on all particles,
        :math:`N_{\mathrm{dof, trans}} = DN - D - N_{constraints}`, where
        :math:`D` is the dimensionality of the system.

        Note:
            The removal of :math:`D` degrees of freedom accounts for the fixed
            center of mass in using periodic boundary conditions. When the
            *filter* in :py:class:`ThermodynamicQuantities` selects a subset
            of all particles, the removed degrees of freedom are spread
            proportionately.
        """
        if self._attached:
            return self._cpp_obj.translational_degrees_of_freedom
        else:
            return None

    @log
    def rotational_degrees_of_freedom(self):
        r""":math:`N_{\mathrm{dof, rot}}`.

        Number of rotational degrees of freedom given to the group by its
        integration method.
        """
        if self._attached:
            return self._cpp_obj.rotational_degrees_of_freedom
        else:
            return None

    @log
    def num_particles(self):
        """:math:`N`, number of particles in the group."""
        if self._attached:
            return self._cpp_obj.num_particles
        else:
            return None


class HarmonicAveragedThermodynamicQuantities(Compute):
    """Compute harmonic averaged thermodynamic properties of particles.

    Args:
        filter (``hoomd.filter``): Particle filter to compute thermodynamic
            properties for.
        kT (float): Temperature of the system.
        harmonic_pressure (float): Harmonic contribution to the pressure.
            If ommitted, the HMA pressure can still be computed, but will be
            similar in precision to the conventional pressure.

    :py:class:`HarmonicAveragedThermodynamicQuantities` acts on a given group
    of particles and calculates harmonically mapped average (HMA) properties
    of those particles when requested. HMA computes properties more precisely
    (with less variance) for atomic crystals in NVT simulations.  The presence
    of dffusion (vacancy hopping, etc.) will prevent HMA from providing
    improvement.  HMA tracks displacements from the lattice positions, which
    are saved either during first call to `Simulation.run` or when the compute
    is first added to the simulation, whichever occurs last.

    Note:
        `HarmonicAveragedThermodynamicQuantities` is an implementation of the
        methods section of Sabry G. Moustafa, Andrew J. Schultz, and David A.
        Kofke. (2015).  "Very fast averaging of thermal properties of crystals
        by molecular simulation". Phys. Rev. E 92, 043303
        doi:10.1103/PhysRevE.92.043303

    Examples::

        hma = hoomd.compute.HarmonicAveragedThermodynamicQuantities(
            filter=hoomd.filter.Type('A'), kT=1.0)


    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles compute
            thermodynamic properties for.

        kT (hoomd.variant.Variant): Temperature of the system.

        harmonic_pressure (float): Harmonic contribution to the pressure.
    """

    def __init__(self, filter, kT, harmonic_pressure=0):

        # store metadata
        param_dict = ParameterDict(kT=float(kT),
                                   harmonic_pressure=float(harmonic_pressure))
        # set defaults
        self._param_dict.update(param_dict)

        self._filter = filter
        # initialize base class
        super().__init__()

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            thermoHMA_cls = _md.ComputeThermoHMA
        else:
            thermoHMA_cls = _md.ComputeThermoHMAGPU
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = thermoHMA_cls(self._simulation.state._cpp_sys_def,
                                      group, self.kT, self.harmonic_pressure)
        super()._attach()

    @log
    def potential_energy(self):
        """Average potential energy :math:`[energy]`."""
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.potential_energy
        else:
            return None

    @log
    def pressure(self):
        """Average pressure :math:`[pressure]`."""
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.pressure
        else:
            return None
