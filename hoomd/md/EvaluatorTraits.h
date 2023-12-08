//
// Created by girard01 on 12/1/23.
//

#ifndef HOOMD_EVALUATORTRAITS_H
#define HOOMD_EVALUATORTRAITS_H

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif



namespace hoomd::md {

    namespace detail{
        struct _evaluator_has_charge{};
    }

    struct charge_product_traits : detail::_evaluator_has_charge{
        DEVICE void setCharge(Scalar qi, Scalar qj)
        {
            qiqj = qi * qj;
        }
    protected:
        Scalar qiqj;
    };

    struct charge_traits : detail::_evaluator_has_charge{
        DEVICE void setCharge(Scalar qi, Scalar qj)
        {
            q_i = qi; q_j = qj;
        }
    protected:
        Scalar q_i = 0, q_j = 0;
    };

    template<class evaluator>
    struct requires_charge {
        static constexpr bool value = std::is_base_of<evaluator, detail::_evaluator_has_charge>::value;
    };
}



#endif //HOOMD_EVALUATORTRAITS_H
