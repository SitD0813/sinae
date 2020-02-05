// This file is the part of "sinae", an automatic differentiation library written in C99.
//
// Copyright © 2020 SitD0813 <sitd0813@gmail.com>
//
// This file is licensed under the MIT License.
// See LICENSE.txt for more informtation or you can obtain a copy at https://opensource.org/licenses/MIT/.

//! \file  sinae_op.h
//! \brief This file includes symbolic calculation operators.

#ifndef SINAE_HELPER_H_INCLUDED_
#define SINAE_HELPER_H_INCLUDED_

#include "sinae_core.h"


/* Element-wise unary operatros */

sn_op* sn_abs(sn_op* x);
sn_op* sn_exp(sn_op* x);
sn_op* sn_negative(sn_op* x);
sn_op* sn_reciprocal(sn_op* x);
sn_op* sn_sqrt(sn_op* x);


/* Unary operatros */

sn_op* sn_sum(sn_op* x);


/* Element-wise binary operatros */

sn_op* sn_add(sn_op* x0, sn_op* x1);
sn_op* sn_subtract(sn_op* x0, sn_op* x1);
sn_op* sn_multiply(sn_op* x0, sn_op* x1);
sn_op* sn_divide(sn_op* x0, sn_op* x1);


/* Binary operatros */

sn_op* sn_matmul(sn_op* x0, sn_op* x1, SN_UINT overwrap);


#endif // !SINAE_HELPER_H_INCLUDED_