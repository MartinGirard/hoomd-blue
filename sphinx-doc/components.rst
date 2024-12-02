Extending HOOMD-blue
====================

Extend **HOOMD-blue** with a **component** implemented in C++ for performance-critical
tasks, such as pair potential evaluation.

Creating a component
--------------------

The `hoomd-component-template`_ repository provides a starting point for your
component. It includes template C++ and Python modules, an example unit test, CMake
scripts to build the component, and GitHub Actions workflows.

.. _hoomd-component-template: https://github.com/glotzerlab/hoomd-component-template

The following repositories template various custom operations:

* `MD pair potential <https://github.com/glotzerlab/hoomd-md-pair-template>`__
* `HPMC pair and external potentials <https://github.com/glotzerlab/hpmc-energy-template>`__
* `HPMC hard shapes <https://github.com/glotzerlab/hoomd-hpmc-shape-template/>`__

Building an external component
------------------------------

To build an external component:

1. Build and install **HOOMD-blue** from source.
2. Obtain the component's source::

    $ git clone https://github.com/<organization>/<component>

3. Configure::

    $ cmake -B build/<component> -S <component>

4. Build the component::

    $ cmake --build build/<component>

5. Install the component::

    $ cmake --install build/<component>

Once installed, the component is available for import via::

    import hoomd.<component>

.. note::

    Replace ``<organization>`` and ``<component>`` with the names of the organization
    and repository of the component you would like to build.
