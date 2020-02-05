// This file is the part of "sinae", an automatic differentiation library written in C99.
//
// Copyright © 2020 SitD0813 <sitd0813@gmail.com>
//
// This file is licensed under the MIT License.
// See LICENSE.txt for more informtation or you can obtain a copy at https://opensource.org/licenses/MIT/.

//! \file  sinae_mda.h
//! \brief This file includes a column-wise multi-dimensional array object.

#ifndef SINAE_MDA_H_INCLUDED_
#define SINAE_MDA_H_INCLUDED_

#include "sinae_macro.h"


/* Forward declarations */

//! \ingroup multi-dimentional_array_group
typedef struct sn_mda_st sn_mda;


/* struct sn_mda_st */

//! \defgroup multi-dimentional_array_group Multi-dimentional array (sn_mda)
//! \brief    Provides a column-major multi-dimensional array object.
//! \{

struct sn_mda_st {
    SN_UINT rank;   //!< Rank of the array.
    SN_UINT* shape; //!< Shape of the array.
    SN_FLOAT ptr[]; //!< Pointer to the data.
};

//! \brief Creates a sn_mda object.
sn_mda* sn_mda_create(SN_UINT rank, const SN_UINT shape[]);
//! \brief Creates a sn_mda object initialized with a given value.
sn_mda* sn_mda_full(SN_UINT rank, const SN_UINT shape[], SN_FLOAT value);
//! \brief Creates a diagonal sn_mda object initialized with a given value.
sn_mda* sn_mda_diagonal_full(SN_UINT one_side_rank, const SN_UINT one_side_shape[], SN_FLOAT value);
//! \brief Returns the copy of the object.
sn_mda* sn_mda_copy(const sn_mda* self);
//! \brief Destroys the object.
void sn_mda_destroy(sn_mda* self);
//! \brief Returns the size of the array.
SN_UINT sn_mda_size(const sn_mda* self);
//! \brief Gets the pointer to the element of the array.
SN_FLOAT* sn_mda_get(sn_mda* self, const SN_UINT index[]);
//! \brief Views the value of the element of the array.
SN_FLOAT sn_mda_view(const sn_mda* self, const SN_UINT index[]);
//! \brief   Performs generalized matrix multiplication.
//! \details When \p x0 = (2, 3, 5, 1), \p x1 = (5, 1, 2) and \p overwrap = 2, treats x0 as (2x3, 5x1) and x1 as (5x1, 2) and performs matrix multiplication.
sn_mda* sn_mda_gmatmul(const sn_mda* x0, const sn_mda* x1, SN_UINT overwrap);
//! \brief Performs matrix multiplication.
#define sn_mda_matmul(x0, x1) sn_mda_gmatmul(x0, x1, 1);

//! \}


#endif // !SINAE_MDA_H_INCLUDED_