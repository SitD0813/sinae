// This file is the part of "sinae", an automatic differentiation library written in C99.
//
// Copyright © 2020 SitD0813 <sitd0813@gmail.com>
//
// This file is licensed under the MIT License.
// See LICENSE.txt for more informtation or you can obtain a copy at https://opensource.org/licenses/MIT/.

//! \file  sinae_macro.h
//! \brief This file includes macros.

#ifndef SINAE_MACRO_H_INCLUDED_
#define SINAE_MACRO_H_INCLUDED_


/* Override macros here */

/* -------------------- */


/* Debug macros */

#ifndef SN_ASSERT
    #ifdef SN_NDEBUG
        //! \brief Overridable assert macro disabled if "SN_NDEBUG" is defined.
        #define SN_ASSERT(CONDITION) (0)
    #else
        #include <assert.h>
        //! \brief Overridable assert macro disabled if "SN_NDEBUG" is defined.
        #define SN_ASSERT(CONDITION) assert(CONDITION)
    #endif // SN_NDEBUG
#endif // !SN_ASSERT


/* Overridable type macros */

#ifndef SN_FLOAT
    //! \brief Overridable floating point type used in calculations.
    #define SN_FLOAT double
#endif // !SN_FLOAT

#ifndef SN_UINT
    #include <stdint.h>
    //! \brief Overridable unsigned integer type used to represent the size of an object, etc.
    #define SN_UINT uintmax_t
#endif // !SN_SIZE_T

/* Overridable memory allocation macros */

#ifndef SN_MALLOC
    #include <stdlib.h>
    #define SN_MALLOC(SIZE) (malloc(SIZE))
#endif

#ifndef SN_FREE
    #include <stdlib.h>
    #define SN_FREE(PTR) (free(PTR))
#endif


/* Function-like macros */

//! \brief Returns dynamically allocated array.
#define SN_DYNAMIC_ARRAY(TYPE, SIZE) ((TYPE*)SN_MALLOC((SIZE) * sizeof(TYPE)))

//! \brief Returns an array using compound literal.
#define SN_TEMP_ARRAY(TYPE, ...) ((TYPE[]) { __VA_ARGS__ })

//! \brief Returns a const array using compound literal.
#define SN_CONST_TEMP_ARRAY(TYPE, ...) ((const TYPE[]) { __VA_ARGS__ })

//! \brief Returns a const SN_UINT array using compound literal.
#define SN_INDEX(...) SN_CONST_TEMP_ARRAY(SN_UINT, __VA_ARGS__)

//! \brief Returns a const SN_UINT array using compound literal.
#define SN_SHAPE(...) SN_CONST_TEMP_ARRAY(SN_UINT, __VA_ARGS__)

//! \brief Interprets the given pointer as a column-major matrix and returns the element.
#define SN_MATRIX_GET(PTR, ROW_SIZE, ROW_INDEX, COLUMN_INDEX) ((PTR)[ROW_INDEX + COLUMN_INDEX * ROW_SIZE])


#endif // !SINAE_MACRO_H_INCLUDED_
