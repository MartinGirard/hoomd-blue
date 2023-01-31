//
// Created by girard01 on 1/20/23.
//

#include "MolecularForceCompute.h"

#ifndef HOOMD_MOLECULEENSEMBLE_H
#define HOOMD_MOLECULEENSEMBLE_H


namespace hoomd::md {
    class MolecularHashAction;
    class MolecularHashCompute;

    class MoleculeEnsemble : public MolecularForceCompute {
    public:
        MoleculeEnsemble(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         bool include_all_bonded = true);

        auto& getHashes(){
            return m_hashes;
        }

        auto& getMoecularLock(){
            return m_molecule_lock;
        }

        void rebuild_table(); //! if system has changed, we need to cluster molecules

        void register_action(std::shared_ptr<MolecularHashAction> action);

        void deregister_action(std::shared_ptr<MolecularHashAction> action);

        void update_hash_sizes();

        void computeHashes(std::size_t);

        auto getMaximumHash() const{
            return 1u << m_hash_size;
        }

        auto get_hash_description(unsigned int hash);

    protected:
        std::shared_ptr<ParticleGroup> m_group;

        bool m_include_all_bonded;
        unsigned int m_hash_size = 0; //! the number of bits currently associated with the hashes of this set of molecules
        GlobalArray<unsigned int> m_hashes; //! the hashes of all molecules within this set
        GlobalArray<Scalar> m_chemical_potentials; //! chemical potentials associated with a given hash, if any
        GlobalArray<unsigned int> m_molecule_lock; //! for use with CAS lock (hash changes may operate only on a subset of the whole hash, and two changes may result in a data race)

        std::vector<std::weak_ptr<MolecularHashAction>> m_registered_actions;
        std::vector<std::weak_ptr<MolecularHashCompute>> m_registered_computes;
    };
}

#endif //HOOMD_MOLECULEENSEMBLE_H