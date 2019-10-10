#ifndef vector_source_h
#define vector_source_h
#include <math.h> 
double Vector_Source_x(double x, double y,double z,double t){
	//double K=2;       
        return 0.;
}
double Vector_Source_y(double x, double y,double z,double t){
        //double K = 0.25;
	double A = 1.;
	return A*(cos(1.*x)+cos(2.*x)+cos(4.*x));
}
double Vector_Source_z(double x, double y,double z,double t){
	return 0.;
}
#endif
