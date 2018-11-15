#ifndef MATH_EVAL_H
#define MATH_EVAL_H

#include "matrix.hpp"

Matrix meshgrid( int dim,  double start_val, double step_val, double _stop_val);

Matrix feval( double (*f)(double x, double y), const Matrix & X, const Matrix & Y);

#endif
