//
// Created by girard01 on 2/14/23.
//


#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"

#include "hoomd/GPUPartition.cuh"

#ifdef __HIPCC__
#include "hoomd/WarpTools.cuh"
#endif // __HIPCC__

#include <assert.h>
#include <type_traits>

#ifndef HOOMD_VIRTUALSITEGPU_CUH
#define HOOMD_VIRTUALSITEGPU_CUH
namespace hoomd::md::kernel {
    struct VirtualSiteUpdateKernelArgs {
        Scalar4* d_postype;
        unsigned int* d_mol_list;
        Index2D indexer;
        uint64_t N;
    };

    struct VirtualSiteDecomposeKernelArgs{
        Scalar4* forces;
        Scalar4* postype;
        Scalar* virial;
        Scalar* net_virial;
        uint64_t virial_pitch;
        uint64_t net_virial_pitch;
        unsigned int* d_mol_list;
        Index2D indexer;
        uint64_t N;
    };


#ifdef __HIPCC__

    template<class Mapping>
    __global__ void gpu_update_virtual_site_kernel(Scalar4* postype,
                                                   const unsigned int* d_molecule_list,
                                                   Index2D moleculeIndexer,
                                                   const typename Mapping::param_type param,
                                                   const uint64_t N
                                                   ) {
        auto site = threadIdx.x + blockIdx.x * blockDim.x;
        if(site >= moleculeIndexer.getH())
            return;

        auto constexpr site_length = Mapping::length;
        const auto site_index = moleculeIndexer(site_length, site);
        if(site_index >= N) // virtual particle is not local
            return;

        // the contents of the molecules should directly be mappable to the
        // index structure
        typename Mapping::index_type indices;
        for(unsigned char s = 0; s < site_length; s++){
            indices.indices[s] = d_molecule_list[moleculeIndexer(s, site)];
        }

        Mapping virtual_site(indices, param);
        virtual_site.reconstructSite(postype);
    }

    template<class Mapping, bool compute_virial>
    __global__ void gpu_decompose_virtual_site_kernel(Scalar4* forces,
                                                      Scalar4* net_forces,
                                                      Scalar* virial,
                                                      Scalar* net_virial,
                                                      Scalar4* postype,
                                                      uint64_t virial_pitch,
                                                      uint64_t net_virial_pitch,
                                                      const unsigned int* d_molecule_list,
                                                      Index2D moleculeIndexer,
                                                      const typename Mapping::param_type param,
                                                      const uint64_t N){
        auto site = threadIdx.x + blockIdx.x * blockDim.x;
        if(site >= moleculeIndexer.getH())
            return;

        auto constexpr site_length = Mapping::length;
        const auto site_index = moleculeIndexer(site_length, site);

        if(site_index >= N) { // virtual particle is not local; we just zero forces / virials and return
            net_forces[site_index] = {0., 0., 0., 0.};
            if(compute_virial){
                for(auto i = 0; i <6; i++)
                    net_virial[i * net_virial_pitch + site_index] = 0;
            }
            return;
        }


        typename Mapping::index_type indices;
        for(unsigned char s = 0; s < site_length; s++){
            indices.indices[s] = d_molecule_list[moleculeIndexer(s, site)];
        }

        Mapping virtual_site(indices, param);

        virtual_site.project(
                postype,
                forces,
                net_forces,
                virial,
                net_virial,
                virial_pitch,
                net_virial_pitch
                );
    }

#endif

    template<class Mapping>
    hipError_t gpu_update_virtual_sites(const VirtualSiteUpdateKernelArgs &args,
                                        const typename Mapping::param_type param) {
        return hipSuccess;
    }

    template<class Mapping>
    hipError_t gpu_decompose_virtual_sites(const VirtualSiteDecomposeKernelArgs& args,
                                           const typename Mapping::param_type param){
        // memset forces to 0 first

        // then compute the ctr forces

        return hipSuccess;
    }
}
#endif //HOOMD_VIRTUALSITEGPU_CUH
