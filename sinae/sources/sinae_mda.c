// This file is the part of "sinae", an automatic differentiation library written in C99.
//
// Copyright © 2020 SitD0813 <sitd0813@gmail.com>
//
// This file is licensed under the MIT License.
// See LICENSE.txt for more informtation or you can obtain a copy at https://opensource.org/licenses/MIT/.

//! \file  sinae_mda.c
//! \brief This file implements sinae_mda.h.

#include "../sinae_mda.h"

#include <stdarg.h>


/* struct sn_list_st */

static SN_UINT sizeof_shape_(SN_UINT rank, const SN_UINT shape[]) {
    SN_UINT size = 1;
    if (shape) {
        for (SN_UINT i = 0; i < rank; ++i) {
            size *= shape[i];
        }
    }
    return size;
}

sn_mda* sn_mda_create(SN_UINT rank, const SN_UINT shape[]) {
    SN_UINT size = sizeof_shape_(rank, shape);
    sn_mda* obj = (sn_mda*)SN_MALLOC(sizeof(sn_mda) + size * sizeof(SN_FLOAT) + rank * sizeof(SN_UINT));
    obj->rank = rank;
    obj->shape = (rank > 0 ? (SN_UINT*)&(obj->ptr[size]) : NULL);
    if (shape) {
        for (SN_UINT i = 0; i < rank; ++i) {
            obj->shape[i] = shape[i];
        }
    }
    return obj;
}

sn_mda* sn_mda_full(SN_UINT rank, const SN_UINT shape[], SN_FLOAT value) {
    sn_mda* obj = sn_mda_create(rank, shape);
    SN_UINT size = sn_mda_size(obj);
    for (SN_UINT i = 0; i < size; ++i) {
        obj->ptr[i] = value;
    }
    return obj;
}

sn_mda* sn_mda_diagonal_full(SN_UINT one_side_rank, const SN_UINT one_side_shape[], SN_FLOAT value) {
    SN_UINT* obj_shape = SN_DYNAMIC_ARRAY(SN_UINT, 2 * one_side_rank);
    for (SN_UINT i = 0; i < one_side_rank; ++i) {
        obj_shape[i] = one_side_shape[i];
        obj_shape[one_side_rank + i] = one_side_shape[i];
    }
    sn_mda* obj = sn_mda_full(2 * one_side_rank, obj_shape, 0.0);
    SN_FREE(obj_shape);

    SN_UINT one_side_size = sizeof_shape_(one_side_rank, one_side_shape);
    for (SN_UINT i = 0; i < one_side_size; ++i) {
        SN_MATRIX_GET(obj->ptr, one_side_size, i, i) = value;
    }
    return obj;
}

sn_mda* sn_mda_copy(const sn_mda* self) {
    sn_mda* obj = sn_mda_create(self->rank, self->shape);
    SN_UINT size = sn_mda_size(obj);
    for (SN_UINT i = 0; i < size; ++i) {
        obj->ptr[i] = self->ptr[i];
    }
    return obj;
}

void sn_mda_destroy(sn_mda* self) {
    SN_FREE(self);
}

SN_UINT sn_mda_size(const sn_mda* self) {
    return sizeof_shape_(self->rank, self->shape);
}

static SN_UINT sn_mda_get_offset_(const sn_mda* self, const SN_UINT index[]) {
#ifndef SN_NDEBUG
    for (SN_UINT i = 0; i < self->rank; ++i) {
        SN_ASSERT(index[i] < self->shape[i]);
    }
#endif
    SN_UINT offset = 0;
    for (SN_UINT i = self->rank - 1; i > 0; --i) {
        offset += index[i];
        offset *= self->shape[i - 1];
    }
    offset += index[0];
    return offset;
}

SN_FLOAT* sn_mda_get(sn_mda* self, const SN_UINT index[]) {
    return &(self->ptr[sn_mda_get_offset_(self, index)]);
}

SN_FLOAT sn_mda_view(const sn_mda* self, const SN_UINT index[]) {
    return self->ptr[sn_mda_get_offset_(self, index)];
}

sn_mda* sn_mda_gmatmul(const sn_mda* x0, const sn_mda* x1, SN_UINT overwrap) {
#ifndef SN_NDEBUG
    for (SN_UINT i = 0; i < overwrap; ++i) {
        SN_ASSERT(x0->shape[x0->rank - overwrap + i] == x1->shape[i]);
    }
#endif
    SN_UINT* y_shape = SN_DYNAMIC_ARRAY(SN_UINT, x0->rank - overwrap + x1->rank - overwrap);
    for (SN_UINT i = 0; i < x0->rank - overwrap; ++i) {
        y_shape[i] = x0->shape[i];
    }
    for (SN_UINT i = 0; i < x1->rank - overwrap; ++i) {
        y_shape[x0->rank - overwrap + i] = x1->shape[overwrap + i];
    }
    sn_mda* y = sn_mda_create(x0->rank - overwrap + x1->rank - overwrap, y_shape);

    SN_UINT x0_front_size = sizeof_shape_(x0->rank - overwrap, x0->shape);
    SN_UINT overwrap_size = sizeof_shape_(overwrap, x1->shape);
    SN_UINT x1_back_size = sizeof_shape_(x1->rank - overwrap, &(x1->shape[overwrap]));

    for (SN_UINT j = 0; j < x1_back_size; ++j) {
        for (SN_UINT i = 0; i < x0_front_size; ++i) {
            SN_FLOAT temp_sum = 0;
            for (SN_UINT k = 0; k < overwrap_size; ++k) {
                temp_sum += SN_MATRIX_GET(x0->ptr, x0_front_size, i, k) * SN_MATRIX_GET(x1->ptr, overwrap_size, k, j);
            }
            SN_MATRIX_GET(y->ptr, x0_front_size, i, j) = temp_sum;
        }
    }

    SN_FREE(y_shape);
    return y;
}