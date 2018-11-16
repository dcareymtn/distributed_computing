#ifndef CMATH_EVAL_H
#define CMATH_EVAL_H

void c_hello();

double feval_c( double (*f)(int dim, double * vec), int dim, double * vec);

double root_mean_square( int dim, double * vec);

double x2_plus_y2(double x, double y);

double sum_of_the_squares( int dim, double * vec);

double rastrigin( double x, double y);

double rastrigin( int dim, double * vec);

#endif
