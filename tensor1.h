#ifndef tensor1_h
#define tensor1_h
#include "gridsize.h"
#include "tensor0.h"
class communicator;
class tensor1 
{
  communicator* comm_;
  gridsize *p_;
  public:  
  int Nx() const{return p_->Nx();}
  int Ny() const{return p_->Ny();}
  int Nz() const{return p_->Nz();}
  int bs() const{return p_->bs();}
  double Lx() const{return p_->Lx();}
  double Ly() const{return p_->Ly();}
  double Lz() const{return p_->Lz();}
  double dx() const{return p_->dx();}
  double dy() const{return p_->dy();}
  double dz() const{return p_->dz();}
  gridsize* parameter() const{return p_;}
  tensor1* my_ptr() {return this;}
  tensor0 x;
  tensor0 y;
  tensor0 z;
  int size() const{return p_->size();}
  tensor1& operator=(const double&);
  tensor1& operator+=(const double&);
  tensor1& operator*=(const double&);
  tensor1& operator-=(const double&);
  tensor1& operator/=(const double&);
  tensor1& operator=(const tensor1&);
  tensor1& operator+=(const tensor1&);
  tensor1& operator*=(const tensor1&);
  tensor1& operator-=(const tensor1&);
  tensor1& operator/=(const tensor1&);
  tensor1& operator=(const tensor0&);
  tensor1& operator+=(const tensor0&);
  tensor1& operator*=(const tensor0&);
  tensor1& operator-=(const tensor0&);
  tensor1& operator/=(const tensor0&);
  tensor1(gridsize*,communicator*);
  void Update_Ghosts();
  void Update_Ghosts_CUM();
  tensor1(){}
  ~tensor1(){}
  //statistics
  void make_mean_zero(); //make mean value of T.x, T.y, and T.z zero
  double sum(); //sum of all elements in x, y, and z
  double sum_squares(); //sum of all elements squared in x, y, and z
  double rms(); //sqrt(mean_squares)
  double mean_squares(); //sum of mean_squares of x, y, and z
  double max();
  double min();
  double max_cfl(double); //should provide dt to compute maximum convective cfl of three directions
  //arithmatics
  void PlusEqual_Mult(tensor1&,double);
  void PlusEqual_Mult(tensor1&,tensor1&);
  void PlusEqual_Mult(double,tensor1&);
  void Equal_Divide(tensor1&,double);
  void Equal_Divide(tensor1&,tensor1&);
  void Equal_Divide(double,tensor1&);
  void Equal_Divide(tensor1&,tensor0&);
  void Equal_Divide(tensor0&,tensor1&);
  void Equal_Mult(tensor1&,double);
  void Equal_Mult(tensor1&,tensor1&);
  void Equal_Mult(double,tensor1&);
  void Equal_LinComb(double,tensor1&,double,tensor1&);
  void Equal_LinComb(tensor1&,tensor1&,tensor1&,tensor1&);
  void Equal_LinComb(double,tensor1&,tensor1&,tensor1&);
  void Equal_LinComb(tensor1&,tensor1&,double,tensor1&);
  void Equal_LinComb(double,tensor1&,double,tensor1&,double,tensor1&); //special lin. comb. for lagrangian momentum two way coupling 
  //interpolation
  void Equal_I_C2F(tensor0&);
  void Equal_Ix_C2F(tensor1&);
  void Equal_Iy_C2F(tensor1&);
  void Equal_Iz_C2F(tensor1&);
  void Equal_I_F2C(tensor1&);
  //Differentiation
  void Equal_Grad_C2F(tensor0&);
  void Equal_Del2(tensor1&);
};
#endif
