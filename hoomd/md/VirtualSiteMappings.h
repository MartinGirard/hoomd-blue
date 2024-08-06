//
// Created by girard01 on 2/8/23.
//

#ifndef HOOMD_VIRTUALSITEMAPPINGS_H
#define HOOMD_VIRTUALSITEMAPPINGS_H

#include <array>
#include <hoomd/HOOMDMath.h>

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd::md {

    struct VSMap {
        static constexpr unsigned char length = 0;
        struct param_type{};
        struct index_type{};
    };

namespace virtualsites {

    template<unsigned char N>
    struct Linear : VSMap{
        static constexpr unsigned char length = N;
        struct param_type{
            Scalar coefficient[N];
        };

        struct index_type{
            uint64_t indices[N];
        };

        inline HOSTDEVICE Scalar3 reconstruct(Scalar4* postypes){
            Scalar3 pos{0., 0., 0.};

            for(auto i = 0; i < N; i++){
                Scalar4 postype = postypes[indices.indices[i]];
                Scalar3 xyz{postype.x, postype.y, postype.z};
                pos += xyz * params.coefficient[i];
            }
            return pos;
        }

        template<bool compute_virial>
        inline HOSTDEVICE void project(
                Scalar4* postype,
                Scalar4* net_forces,
                Scalar* net_virial,
                uint64_t net_virial_pitch,
                uint64_t source
                ){
            Scalar virial[6] = {
                    net_virial[0 * net_virial_pitch + source],
                    net_virial[1 * net_virial_pitch + source],
                    net_virial[2 * net_virial_pitch + source],
                    net_virial[3 * net_virial_pitch + source],
                    net_virial[4 * net_virial_pitch + source],
                    net_virial[5 * net_virial_pitch + source]
            };

            Scalar4 site_postype = postype[source];

            for(auto i = 0; i < N; i++){
                const auto id = indices.indices[N];
                const auto F = net_forces[source] * params.coefficient[N];
                net_forces[id] += F;

                const auto j_postype = postype[id];

                Scalar3 dr{
                    site_postype.x - j_postype.x,
                    site_postype.y - j_postype.y,
                    site_postype.z - j_postype.z
                };

                if(compute_virial){
                    net_virial[0 * net_virial_pitch + id] += virial[0] - F.x * dr.x;
                    net_virial[1 * net_virial_pitch + id] += virial[1] - F.x * dr.y;
                    net_virial[2 * net_virial_pitch + id] += virial[2] - F.x * dr.z;
                    net_virial[3 * net_virial_pitch + id] += virial[3] - F.y * dr.y;
                    net_virial[4 * net_virial_pitch + id] += virial[4] - F.y * dr.z;
                    net_virial[5 * net_virial_pitch + id] += virial[5] - F.z * dr.z;
                }
            }

            // zero out other contributions
            net_forces[source] = {0., 0., 0., 0.};
            if(compute_virial){
                net_virial[0 * net_virial_pitch + source] = 0;
                net_virial[1 * net_virial_pitch + source] = 0;
                net_virial[2 * net_virial_pitch + source] = 0;
                net_virial[3 * net_virial_pitch + source] = 0;
                net_virial[4 * net_virial_pitch + source] = 0;
                net_virial[5 * net_virial_pitch + source] = 0;
            }

        }

        Linear<N>(index_type indices_, param_type params_) : indices(indices_), params(params_){}

        index_type indices;
        param_type params;
    };


    struct Linear2 : Linear<2>{};
    struct Linear3 : Linear<3>{};
    struct Linear4 : Linear<4>{};

}

}
#endif //HOOMD_VIRTUALSITEMAPPINGS_H
