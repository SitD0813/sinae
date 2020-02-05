#include <stdio.h>

#include "../sinae/sinae.h"


/* Defining custom operator */

// add_and_sum: Adds two sn_mda in the same shape and returns sum in a scalar.
static sn_mda* add_and_sum_flow_(sn_op* self, const sn_mda* x[]) {
    // Assert if the shape of sn_mda does not match.
    SN_ASSERT(x[0]->rank == 1 && x[1]->rank == 1);
    SN_ASSERT(x[0]->shape[0] == x[1]->shape[0]);
    SN_UINT size = sn_mda_size(x[0]);
    SN_FLOAT y = 0.0;
    for (SN_UINT i = 0; i < size; ++i) {
        y += x[0]->ptr[i] + x[1]->ptr[i];
    }
    // Return scalar
    return sn_mda_full(0, NULL, y);
}

// Output shape of sn_mda in dflow
//
// y = f(x0, x1) when y = [rank = 1, shape = (3,)] and (x0, x1) = [rank = 2, shape = (2, 3)]
// rank = 1 + 2 = 3, shape = (3,) + (2, 3) = (3, 2, 3)
// { { { dy[0]/dx0[0, 0], dy[0]/dx[0, 1] },
//     { dy[0]/dx0[1, 0], dy[0]/dx[1, 1] },
//     { dy[0]/dx0[2, 0], dy[0]/dx[2, 1] } }
//
//   { { dy[1]/dx0[0, 0], dy[1]/dx[0, 1] },
//     { dy[1]/dx0[1, 0], dy[1]/dx[1, 1] },
//     { dy[1]/dx0[2, 0], dy[1]/dx[2, 1] } }
//
//   { { dy[2]/dx0[0, 0], dy[2]/dx[0, 1] },
//     { dy[2]/dx0[1, 0], dy[2]/dx[0, 1] },
//     { dy[2]/dx0[2, 0], dy[2]/dx[0, 1] } } }
//
// In this case,
// y = add_and_sum(x0, x1) when y = [rank = 0, shape = NULL] and (x0, x1) = [rank = 1, shape = (5,)]
// rank = 0 + 1 = 1, shape = (0,) + (5,) = (5,)
//  [ { dy/x0[0], dy/x0[1], dy/x0[2], dy/x0[3], dy/x0[4] },
//    { dy/x1[0], dy/x1[1], dy/x1[2], dy/x1[3], dy/x1[4] } ]

static sn_mda** add_and_sum_dflow_(sn_op* self, const sn_mda* x[]) {
    sn_mda** dy_dx_list = SN_DYNAMIC_ARRAY(sn_mda*, 2);
    dy_dx_list[0] = sn_mda_full(1, x[0]->shape, 1.0);
    dy_dx_list[1] = sn_mda_full(1, x[0]->shape, 1.0);
    return dy_dx_list;
}

// Adds two sn_mda in the same shape and returns sum in a scalar.
sn_op* add_and_sum(sn_op* x0, sn_op* x1) {
    return sn_op_create(OPERATOR, &add_and_sum_flow_, &add_and_sum_dflow_, 2, SN_TEMP_ARRAY(sn_op*, x0, x1));
}


int main(void) {
    /* Evaluating an expression and gradient calculation */

    // Creates placeholders and operators.
    sn_op* placeholder0 = sn_placeholder();
    sn_op* placeholder1 = sn_placeholder();
    sn_op* op = add_and_sum(placeholder0, placeholder1);
    op = sn_negative(op);

    // Creates a sn_mda for input.
    sn_mda* mda0 = sn_mda_full(1, (SN_UINT[]) { 5 }, 5.0);
    sn_mda* mda1 = sn_mda_full(1, (SN_UINT[]) { 5 }, 7.0);

    // Evaluates the expression.
    // sn_op_flow destroys the sn_map objects.
    sn_mda* result = sn_op_flow(op, sn_map_from(2, placeholder0, sn_mda_copy(mda0), placeholder1, sn_mda_copy(mda1)));
    printf("result: rank = %u, value = %f\n", result->rank, result->ptr[0]);

    // Calculates the gradient of the expression.
    sn_map* gradient_map = sn_op_dflow(op, sn_map_from(2, placeholder0, mda0, placeholder1, mda1));
    sn_mda* gradient0 = sn_map_get(gradient_map, placeholder0);
    sn_mda* gradient1 = sn_map_get(gradient_map, placeholder1);

    printf("gradient0: rank = %u, value = { %f, %f, %f, %f, %f }\n", gradient0->rank, gradient0->ptr[0], gradient0->ptr[1], gradient0->ptr[2], gradient0->ptr[3], gradient0->ptr[4]);
    printf("gradient1: rank = %u, value = { %f, %f, %f, %f, %f }\n", gradient1->rank, gradient1->ptr[0], gradient1->ptr[1], gradient1->ptr[2], gradient1->ptr[3], gradient1->ptr[4]);
    
    // Destroys every associated operator.
    sn_op_destroy(op);

    // Destroys the sn_map object and sn_mda objects.
    sn_map_destroy(gradient_map);

    return 0;
}