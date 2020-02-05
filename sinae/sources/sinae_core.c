// This file is the part of "sinae", an automatic differentiation library written in C99.
//
// Copyright © 2020 SitD0813 <sitd0813@gmail.com>
//
// This file is licensed under the MIT License.
// See LICENSE.txt for more informtation or you can obtain a copy at https://opensource.org/licenses/MIT/.

//! \file  sinae_core.c
//! \brief This file implements sinae_core.h.

#include "../sinae_core.h"

#include <stdarg.h>


/* struct sn_map_st */


sn_map* sn_map_create(SN_UINT capacity, sn_op* keys[], sn_mda* values[]) {
    sn_map* obj = (sn_map*)SN_MALLOC(sizeof(sn_map));
    obj->capacity = capacity;
    obj->count = 0;
    obj->keys = (sn_op**)SN_MALLOC(capacity * (sizeof(sn_op*) + sizeof(sn_mda*)));
    obj->values = (sn_mda**)&(obj->keys[capacity]);
    if (keys && values) {
        for (SN_UINT i = 0; i < capacity; ++i) {
            sn_map_insert(obj, keys[i], values[i]);
        }
    }
    return obj;
}

sn_map* sn_map_from(SN_UINT capacity, ... /* sn_op*, sn_mda*, ... */) {
    sn_map* obj = sn_map_create(capacity, NULL, NULL);
    va_list vl;
    va_start(vl, capacity);
    for (SN_UINT i = 0; i < capacity; ++i) {
        sn_op* key = va_arg(vl, sn_op*);
        sn_mda* value = va_arg(vl, sn_mda*);
        sn_map_insert(obj, key, value);
    }
    va_end(vl);
    return obj;
}

void sn_map_extend(sn_map* self, SN_UINT offset) {
    sn_op** new_keys = (sn_op**)SN_MALLOC((self->capacity + offset) * (sizeof(sn_op*) + sizeof(sn_mda*)));
    sn_mda** new_values = (sn_mda**)&(new_keys[self->capacity + offset]);
    for (SN_UINT i = 0; i < self->count; ++i) {
        new_keys[i] = self->keys[i];
        new_values[i] = self->values[i];
    }
    SN_FREE(self->keys);
    self->capacity += offset;
    self->keys = new_keys;
    self->values = new_values;
}

void sn_map_destroy(sn_map* self) {
    for (SN_UINT i = 0; i < self->count; ++i) {
        sn_mda_destroy(self->values[i]);
    }
    SN_FREE(self->keys);
    SN_FREE(self);
}

void sn_map_insert(sn_map* self, sn_op* key, sn_mda* value) {
    if (self->count < self->capacity) {
        self->keys[self->count] = key;
        self->values[self->count] = value;
        ++(self->count);
    }
    else {
        sn_map_extend(self, self->capacity);
        sn_map_insert(self, key, value);
    }
}

sn_mda* sn_map_get(sn_map* self, sn_op* key) {
    for (SN_UINT i = 0; i < self->count; ++i) {
        if (self->keys[i] == key) {
            return self->values[i];
        }
    }
    SN_ASSERT(false);
    return NULL;
}

sn_mda** sn_map_get_all(sn_map* self, sn_op* key) {
    SN_UINT count = 0;
    for (SN_UINT i = 0; i < self->count; ++i) {
        if (self->keys[i] == key) {
            ++count;
        }
    }
    sn_mda** values = SN_DYNAMIC_ARRAY(sn_mda*, count);
    count = 0;
    for (SN_UINT i = 0; i < self->count; ++i) {
        if (self->keys[i] == key) {
            values[count] = self->values[i];
            ++count;
        }
    }
    return values;
}


/* struct sn_op_st */

sn_op* sn_op_create(sn_op_type type, sn_flow_fn* flow, sn_dflow_fn* dflow, SN_UINT x_count, sn_op* x[]) {
    sn_op* obj = (sn_op*)SN_MALLOC(sizeof(sn_op) + x_count * sizeof(sn_op*));
    obj->ref_count = 1;
    obj->type = type;
    obj->flow = flow;
    obj->dflow = dflow;
    obj->x_count = x_count;
    if (x) {
        for (SN_UINT i = 0; i < x_count; ++i) {
            ++(x[i]->ref_count);
            obj->x[i] = x[i];
        }
    }
    return obj;
}

void sn_op_destroy_one(sn_op* self) {
    if (self != NULL) {
        switch (self->type) {
        case CONSTANT:
            sn_mda_destroy(*((sn_mda**)(self->x)));
        default:
            SN_FREE(self);
        }
    }
}

void sn_op_destroy(sn_op* self) {
    if (self != NULL) {
        --(self->ref_count);
        if (self->ref_count < 2) {
            for (SN_UINT i = 0; i < self->x_count; ++i) {
                sn_op_destroy(self->x[i]);
            }
            sn_op_destroy_one(self);
        }
    }
}

static sn_mda* op_flow_(sn_op* self, sn_map* feed) {
    sn_mda* y = NULL;
    if (self->type == CONSTANT) {
        y = sn_mda_copy(*((sn_mda**)(self->x)));
    }
    else if (self->type == OPERATOR) {
        sn_mda** x = SN_DYNAMIC_ARRAY(sn_mda*, self->x_count);
        for (SN_UINT i = 0; i < self->x_count; ++i) {
            x[i] = op_flow_(self->x[i], feed);
            sn_map_insert(feed, self, x[i]);
        }
        y = self->flow(self, x);
        SN_FREE(x);
    }
    else if (self->type == PLACEHOLDER) {
        y = sn_mda_copy(sn_map_get(feed, self));
    }
    else {
        SN_ASSERT(false);
    }
    return y;
}

sn_mda* sn_op_usflow(sn_op* self, sn_map* feed) {
    /* !!! Not implemented !!! */
    return NULL;
}

sn_mda* sn_op_flow(sn_op* self, sn_map* feed) {
    sn_mda* y = op_flow_(self, feed);
    sn_map_destroy(feed);
    return y;
}

static sn_map* op_dflow_(sn_op* self, sn_map* feed) {
    sn_map* dy_dx_map = NULL;
    if (self->type == CONSTANT) {
        /* !!! Not implemented !!! */
        sn_mda* dy_dm = NULL;
    }
    else if (self->type == OPERATOR) {
        sn_map** dm_dx_maps = SN_DYNAMIC_ARRAY(sn_map*, self->x_count);
        SN_UINT dy_dx_map_capacity = 0;
        for (SN_UINT i = 0; i < self->x_count; ++i) {
            dm_dx_maps[i] = op_dflow_(self->x[i], feed);
            dy_dx_map_capacity += dm_dx_maps[i]->count;
        }

        sn_mda** x = sn_map_get_all(feed, self);
        sn_mda** dy_dm_list = self->dflow(self, x);

        dy_dx_map = sn_map_create(dy_dx_map_capacity, NULL, NULL);
        for (SN_UINT i = 0; i < self->x_count; ++i) {
            for (SN_UINT j = 0; j < dm_dx_maps[i]->count; ++j) {
                sn_map_insert(dy_dx_map, dm_dx_maps[i]->keys[j], sn_mda_gmatmul(dy_dm_list[i], dm_dx_maps[i]->values[j], x[i]->rank));
            }
        }

        for (SN_UINT i = 0; i < self->x_count; ++i) {
            sn_mda_destroy(dy_dm_list[i]);
        }
        SN_FREE(dy_dm_list);
        SN_FREE(x);
        for (SN_UINT i = 0; i < self->x_count; ++i) {
            sn_map_destroy(dm_dx_maps[i]);
        }
        SN_FREE(dm_dx_maps);
    }
    else if (self->type == PLACEHOLDER) {
        sn_mda* feed_mda = sn_map_get(feed, self);
        sn_mda* dy_dx = sn_mda_diagonal_full(feed_mda->rank, feed_mda->shape, 1.0);
        dy_dx_map = sn_map_create(1, &self, &dy_dx);
    }
    else {
        SN_ASSERT(false);
    }

    return dy_dx_map;
}

sn_map* sn_op_usdflow(sn_op* self, sn_map* feed) {
    /* !!! Not implemented !!! */
    return NULL;
}

sn_map* sn_op_dflow(sn_op* self, sn_map* feed) {
    sn_mda_destroy(op_flow_(self, feed));
    sn_map* dy_dx = op_dflow_(self, feed);
    sn_map_destroy(feed);
    return dy_dx;
}


sn_op* sn_placeholder(void) {
    return sn_op_create(PLACEHOLDER, NULL, NULL, 0, NULL);
}

sn_op* sn_const(sn_mda* array) {
    sn_op* obj = (sn_op*)SN_MALLOC(sizeof(sn_op) + sizeof(sn_mda**));
    obj->ref_count = 1;
    obj->type = CONSTANT;
    obj->flow = NULL;
    obj->dflow = NULL;
    obj->x_count = 0;
    *((sn_mda**)(obj->x)) = array;
    return obj;
}

sn_op* sn_scalar(SN_FLOAT scalar) {
    return sn_const(sn_mda_full(0, NULL, scalar));
}