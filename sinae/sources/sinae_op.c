// This file is the part of "sinae", an automatic differentiation library written in C99.
//
// Copyright © 2020 SitD0813 <sitd0813@gmail.com>
//
// This file is licensed under the MIT License.
// See LICENSE.txt for more informtation or you can obtain a copy at https://opensource.org/licenses/MIT/.

//! \file  sinae_op.c
//! \brief This file implements sinae_op.h.

#include "../sinae_op.h"

#include <math.h>


/* Helper macros */

#define SN_DEFINE_ELEMENT_WISE_UNARY_OPERATOR(OP_NAME, FLOW, DFLOW)                         \
    sn_mda* OP_NAME##_flow_(sn_op* self, const sn_mda* x[]) {                               \
        sn_mda* y = sn_mda_create(x[0]->rank, x[0]->shape);                                 \
        SN_UINT size = sn_mda_size(x[0]);                                                   \
        for (SN_UINT i = 0; i < size; ++i) {                                                \
            y->ptr[i] = FLOW(x[0]->ptr[i]);                                                 \
        }                                                                                   \
        return y;                                                                           \
    }                                                                                       \
    sn_mda** OP_NAME##_dflow_(sn_op* self, const sn_mda* x[]) {                             \
        sn_mda** dy_dx_list = SN_DYNAMIC_ARRAY(sn_mda*, 1);                                 \
        dy_dx_list[0] = sn_mda_diagonal_full(x[0]->rank, x[0]->shape, 0.0);                 \
        SN_UINT one_side_size = sn_mda_size(x[0]);                                          \
        for (SN_UINT i = 0; i < one_side_size; ++i) {                                       \
            SN_MATRIX_GET(dy_dx_list[0]->ptr, one_side_size, i, i) = DFLOW(x[0]->ptr[i]);   \
        }                                                                                   \
        return dy_dx_list;                                                                  \
    }                                                                                       \
    sn_op* sn_##OP_NAME(sn_op* x) {                                                         \
        return sn_op_create(OPERATOR, &OP_NAME##_flow_, &OP_NAME##_dflow_, 1, (sn_op**)&x); \
    }

static sn_mda* element_wise_binary_operator_flow_(sn_op* self, const sn_mda* x[], SN_FLOAT f(SN_FLOAT, SN_FLOAT)) {
    SN_ASSERT((x[0]->rank == x[1]->rank || x[0]->rank * x[1]->rank == 0)); // If the rank of x0 and x1 does not match.
    sn_mda* y = (x[0]->rank == 0) ? sn_mda_create(x[1]->rank, x[1]->shape) : sn_mda_create(x[0]->rank, x[0]->shape);
    SN_UINT size = (x[0]->rank == 0) ? sn_mda_size(x[1]) : sn_mda_size(x[0]);
    if (x[0]->rank == x[1]->rank) { // If both are in same shape.
#ifndef SN_NDEBUG
        for (SN_UINT i = 0; i < x[0]->rank; ++i) {
            SN_ASSERT(x[0]->shape[i] == x[1]->shape[i]);
        }
#endif
        for (SN_UINT i = 0; i < size; ++i) {
            y->ptr[i] = f(x[0]->ptr[i], x[1]->ptr[i]);
        }
    }
    else if (x[0]->rank == 0) { // If x[0] is scalar.
        for (SN_UINT i = 0; i < size; ++i) {
            y->ptr[i] = f(x[0]->ptr[0], x[1]->ptr[i]);
        }
    }
    else if (x[1]->rank == 0) { // If x[1] is scalar.
        for (SN_UINT i = 0; i < size; ++i) {
            y->ptr[i] = f(x[0]->ptr[i], x[1]->ptr[0]);
        }
    }
    // (x[0]->rank == 0) ? sn_mda_destroy(x[0]) : sn_mda_destroy(x[1]);
    return y;
}

#define SN_DEFINE_ELEMENT_WISE_BINARY_OPERATOR(OP_NAME, FLOW, DFLOW)                                              \
    sn_mda* OP_NAME##_flow_(sn_op* self, const sn_mda* x[]) {                                                     \
        return element_wise_binary_operator_flow_(self, x, FLOW);                                                 \
    }                                                                                                             \
    sn_mda** OP_NAME##_dflow_(sn_op* self, const sn_mda* x[]) {                                                    \
        /* !!! Not implemented !!! */                                                                             \
        return NULL;                                                                                              \
    }                                                                                                             \
    sn_op* sn_##OP_NAME(sn_op* x0, sn_op* x1) {                                                                   \
        return sn_op_create(OPERATOR, &(OP_NAME##_flow_), &(OP_NAME##_dflow_), 2, SN_TEMP_ARRAY(sn_op*, x0, x1)); \
    }


/* Element-wise unary operators. */

static inline SN_FLOAT dabs_(SN_FLOAT x) { return (x >= (SN_FLOAT)0.0) ? 1.0 : -1.0; }
SN_DEFINE_ELEMENT_WISE_UNARY_OPERATOR(abs, fabs, dabs_);
SN_DEFINE_ELEMENT_WISE_UNARY_OPERATOR(exp, exp, exp);
static inline SN_FLOAT dnegative_(SN_FLOAT x) { return -1.0; }
SN_DEFINE_ELEMENT_WISE_UNARY_OPERATOR(negative, -, dnegative_);
static inline SN_FLOAT dreciprocal_(SN_FLOAT x) { return -(SN_FLOAT)1.0 / (SN_FLOAT)(pow(x, 2.0)); }
SN_DEFINE_ELEMENT_WISE_UNARY_OPERATOR(reciprocal, (SN_FLOAT)1.0/, dreciprocal_);
static inline SN_FLOAT dsqrt_(SN_FLOAT x) { return 1.0 / (2 * sqrt(x)); }
SN_DEFINE_ELEMENT_WISE_UNARY_OPERATOR(sqrt, sqrt, dsqrt_);


/* Unary operators */

static sn_mda* sum_flow_(sn_op* self, const sn_mda* x[]) {
    SN_UINT size = sn_mda_size(x[0]);
    SN_FLOAT y = 0.0;
    for (SN_UINT i = 0; i < size; ++i) {
        y += x[0]->ptr[i];
    }
    return sn_mda_full(0, NULL, y);
}
static sn_mda** sum_dflow_(sn_op* self, const sn_mda* x[]) {
    sn_mda** dy_dx_list = SN_DYNAMIC_ARRAY(sn_mda*, 1);
    dy_dx_list[0] = sn_mda_full(1, x[0]->shape, 1.0);
    return dy_dx_list;
}
sn_op* sn_sum(sn_op* x) {
    return sn_op_create(OPERATOR, &sum_flow_, &sum_dflow_, 1, &x);
}


/* Element-wise binary operators */

static inline SN_FLOAT add_(SN_FLOAT x0, SN_FLOAT x1) { return x0 + x1; }
SN_DEFINE_ELEMENT_WISE_BINARY_OPERATOR(add, &add_, NULL);
static inline SN_FLOAT subtract_(SN_FLOAT x0, SN_FLOAT x1) { return x0 - x1; }
SN_DEFINE_ELEMENT_WISE_BINARY_OPERATOR(subtract, &subtract_, NULL);
static inline SN_FLOAT multiply_(SN_FLOAT x0, SN_FLOAT x1) { return x0 * x1; }
SN_DEFINE_ELEMENT_WISE_BINARY_OPERATOR(multiply, &multiply_, NULL);
static inline SN_FLOAT divide_(SN_FLOAT x0, SN_FLOAT x1) { return x0 / x1; }
SN_DEFINE_ELEMENT_WISE_BINARY_OPERATOR(divide, &divide_, NULL);


/* Binary operators */

static sn_mda* matmul_flow_(sn_op * self, const sn_mda* x[]) {
    return sn_mda_gmatmul(x[0], x[1], *((SN_UINT*)&(self->x[2])));
}
static sn_mda** matmul_dflow_(sn_op* self, const sn_mda* x[]) {
    /* !!! Not implemented !!! */
    return NULL;
}
sn_op* sn_matmul(sn_op* x0, sn_op* x1, SN_UINT overwrap) {
    sn_op* obj = (sn_op*)SN_MALLOC(sizeof(sn_op) + 2 * sizeof(sn_op*) + sizeof(SN_UINT));
    obj->ref_count = 1;
    obj->type = OPERATOR;
    obj->flow = &matmul_flow_;
    obj->dflow = &matmul_dflow_;
    obj->x_count = 2;
    obj->x[0] = x0;
    obj->x[1] = x1;
    *((SN_UINT*)&(obj->x[2])) = overwrap;
    return obj;
}