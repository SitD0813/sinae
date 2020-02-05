// This file is the part of "sinae", an automatic differentiation library written in C99.
//
// Copyright © 2020 SitD0813 <sitd0813@gmail.com>
//
// This file is licensed under the MIT License.
// See LICENSE.txt for more informtation or you can obtain a copy at https://opensource.org/licenses/MIT/.

//! \file  sinae_core.h
//! \brief This file includes core functionalities of the library.

#ifndef SINAE_CORE_H_INCLUDED_
#define SINAE_CORE_H_INCLUDED_

#include <stdbool.h>

#include "sinae_macro.h"
#include "sinae_mda.h"


/* Forward declarations */

//! \ingroup hashed_multimap_group
typedef struct sn_map_st sn_map;

//! \ingroup operator_group
typedef struct sn_op_st sn_op;


/* struct sn_map_st */

//! \defgroup hashed_multimap_group Hashed multimap (sn_map)
//! \brief    Provides a hashed multimap object where { key: sn_op*, value: sn_mda* }.
//! \{

struct sn_map_st {
    SN_UINT capacity;
    SN_UINT count;
    sn_op** keys;
    sn_mda** values;
};

//! \brief Creates a sn_map object.
sn_map* sn_map_create(SN_UINT capacity, sn_op* keys[], sn_mda* values[]);
//! \brief Creates a sn_map object.
sn_map* sn_map_from(SN_UINT capacity, ... /* sn_op*, sn_mda*, ... */);
//! \brief Destroys the object.
void sn_map_destroy(sn_map* self);
//! \brief Extend the capacity of the object preserving the values.
void sn_map_extend(sn_map* self, SN_UINT offset);
//! \brief Inserts an key-value pair to the object. Increases the capacity by double if the capacity is exhausted.
void sn_map_insert(sn_map* self, sn_op* key, sn_mda* value);
//! \brief Returns the first value associated with the key.
sn_mda* sn_map_get(sn_map* self, sn_op* key);
//! \brief Returns a dynamically allocated array of all values associated with the key.
sn_mda** sn_map_get_all(sn_map* self, sn_op* key);

//! \}


/* struct sn_op_st */

//! \defgroup operator_group Operator (sn_op)
//! \brief    Provides a symbolic calculation operator object.
//!
//! \details  Currently, sn_op can have multiple inputs but only a single output.
//!
//! \{

//! \brief Function type which performs an actual calculation.
typedef sn_mda* sn_flow_fn(sn_op* op, const sn_mda* x[]);
//! \brief Function type which performs a gradient calculation.
typedef sn_mda** sn_dflow_fn(sn_op* op, const sn_mda* x[]);

//! \brief Enum type to distinguish the type of sn_op object.
typedef enum sn_op_type_en {
    CONSTANT,
    OPERATOR,
    PLACEHOLDER,
} sn_op_type;

struct sn_op_st {
    SN_UINT ref_count;
    sn_op_type type;
    sn_flow_fn* flow;
    sn_dflow_fn* dflow;
    SN_UINT x_count;
    sn_op* x[];
};

//! \brief Creates a sn_op object.
sn_op* sn_op_create(sn_op_type type, sn_flow_fn* flow, sn_dflow_fn* dflow, SN_UINT x_count, sn_op* x[]);
//! \brief Destroys the object without managing a reference counting.
void sn_op_destroy_one(sn_op* self);
//! \brief Recursively destroys the object, managing a reference counting.
void sn_op_destroy(sn_op* self);
//! \brief Calculates a symbolic expression.
sn_mda* sn_op_usflow(sn_op* self, sn_map* feed);
//! \brief Calculates a symbolic expression and destroys the \p feed.
sn_mda* sn_op_flow(sn_op* self, sn_map* feed);
//! \brief Calculates a gradient of symbolic expression.
sn_map* sn_op_usdflow(sn_op* self, sn_map* feed);
//! \brief Calculates a gradient of symbolic expression and destroys the \p feed.
sn_map* sn_op_dflow(sn_op* self, sn_map* feed);

//! \brief Creates a placeholder.
sn_op* sn_placeholder(void);
//! \brief Creates a multi-dimentional array constant.
sn_op* sn_const(sn_mda* array);
//! \brief Creates a scalar constant.
sn_op* sn_scalar(SN_FLOAT scalar);

//! \}


#endif // !SINAE_CORE_H_INCLUDED_