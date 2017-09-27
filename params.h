#ifndef params_h
#define params_h
#include <string>
#include <math.h>
const double PI(3.141592653589793);
const double TWO_PI(2*3.141592653589793);

class params
{
  int Nx_tot_,Ny_tot_,Nz_tot_; //total number of points in each direction
  int Np_; //total number of particles
  int Np_track_; //number of particles to track
  int bs_; //bordersize
  double T_final_,dt_,threshold_; //final time of simulation and delta t and changng parameters from 1 to 2 threshold 
  bool Statistics_,Stat_print_; //1:statistics calculation on 0:off
  int Data_freq_fast_,Data_freq_slow_; //data storing frequency fro Restart files and Tecplot files
  std::string Data_dir_; //data storage directory
  std::string Stat_dir_; //statistics storage directory
  int Initial_; //0:initial turbulence from U/V/W .bin  1: Restart.bin 2: everything is uniform  3: TaylorGreen
  int Initial_C_; //0:initial turbulence from C .bin  1: Restart.bin
  int PreCond_,Solver_; //determine preconditioner and solver
  bool cooling_; //0:if coling is off 1:if cooling is on
  int Iteration_,Iteration1_,Iteration2_; // Number of iteration in Poisson iterative solve before and after threshold
  bool TWC_; //True if momentum teo way coupling is activated
  bool ParticleGravity_; //true if gravity should be exerted on particles
  double epsilon_;
  //dimensional parameters
  double Lx_,Ly_,Lz_; //domain size
  double gx_,gy_,gz_,gx1_,gy1_,gz1_,gx2_,gy2_,gz2_; //gravitioanl forces
  double A_,A1_,A2_; //forcing coefficient befor and after threshold
  double Cp_,Cv_,R_; //specific heat coeffs.
  double Mu0_,k_; //Dynamic viscosity and conductivity
  double Rho0_; //Initial density 
  double U0_; //Convective velocity
  double T0_; //Initial temperature
  double np0_; //particle number density
  double Dp_; //particle diameter
  double Rhop_; //particle density
  double Cvp_; //particel specific heat coeff.
  double epsilonp_; //particle emissivity
  double I0_,I01_,I02_; //lamp intensity
  double D_;//Molecular diffusivity
  double Nu_; //Nusselt number
  double Tp_; //Particle momentum relaxation time
  double mp_; //particle mass
  double P0_; //thermodynamic pressure
  double W_; //reference length = Ly
  
 public:
  double Lx() const{ return Lx_; } 
  double Ly() const{ return Ly_; } 
  double Lz() const{ return Lz_; } 
  double Tp() const{return Tp_;} //Particle momentum relaxation time
  double R() {return R_;}
  double P0() {return P0_;}
  double mp() {return mp_;}
  
  double Nu() {return Nu_;}
  double I0() {return I0_;}
  double I01() {return I01_;}
  double I02() {return I02_;}
  double D() {return D_;}
  double epsilonp() {return epsilonp_;}
  double Cvp() {return Cvp_;}
  double Rhop() {return Rhop_;}
  double Dp() {return Dp_;}
  double np0() {return np0_;}
  double k() {return k_;}
  double Cv() {return Cv_;}
  double Cp() {return Cp_;}
  double Mu0() {return Mu0_;} 
  double T0() {return T0_;}
  double U0() {return U0_;}
  double Rho0() {return Rho0_;}

  double A() {return A_;}
  double A1() const{return A1_;}
  double A2() const{return A2_;}
  double gx1() const{return gx1_;}
  double gx2() const{return gx2_;}
  double gy1() const{return gy1_;}
  double gy2() const{return gy2_;}
  double gz1() const{return gz1_;}
  double gz2() const{return gz2_;}
  double gx() {return gx_;}
  double gy() {return gy_;}
  double gz() {return gz_;}

  double Vol() const{return Lx_*Ly_*Lz_;}
  int Np_track() const{return Np_track_;}

  //non dims:

  int N0() const{ return (int)(np0_*Vol()); } // Initial total number of particles
  double Re() const { return Rho0_*U0_*W_/Mu0_; }
  double Fr() const { return U0_/sqrt(gz_*W_); } //assume gravity in z direction is the reference g
  double St() const { return N0()*mp_/(Lx_*Ly_*Lz_*Rho0_); }
  double Pr() const { return Mu0_*Cp_/k_; }
  double Sc() const { return Rho0_/(Mu0_*D_)}
  double gamma() const { return Cp_/Cv_; }
  double GAMMA() const { return Cvp_/Cv_; }

  int Nx_tot() const{return Nx_tot_;}
  int Ny_tot() const{return Ny_tot_;}
  int Nz_tot() const{return Nz_tot_;}
  int bs() const{return bs_;}
  double T_final() const{return T_final_;}
  double dt() const{return dt_;}
  double threshold() const{return threshold_;}
  int data_freq_fast() const{return Data_freq_fast_;}
  int data_freq_slow() const{return Data_freq_slow_;}
  std::string data_dir() const{return Data_dir_;}
  std::string stat_dir() const{return Stat_dir_;}
  bool Statistics() const{return Statistics_;}
  bool Stat_print() const{return Stat_print_;}
  int Initial() const{return Initial_;}
  int Iteration() {return Iteration_;}
  int Iteration1() const{return Iteration1_;}
  int Iteration2() const{return Iteration2_;}
  double epsilon() const{return epsilon_;}
  bool Is_Cooling() const{return cooling_;}
  bool TWC() const{return TWC_;}
  bool ParticleGravity() const{return ParticleGravity_;}
  int PreCond() const{return PreCond_;}
  int Solver() const{return Solver_;}
  params(){}
  params(char*);
  ~params(){}
  void update(double);
};

#endif
