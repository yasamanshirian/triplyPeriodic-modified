#ifndef scalar_source_h
#define scalar_source_h
#include <math.h> 
double Scalar_Source(double x, double y, double z , double t){
	double A = 1.;
        double k = 8.;
	return A*cos(k*x);
}
#endif
