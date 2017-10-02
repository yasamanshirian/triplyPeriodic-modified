#ifndef grid_h
#define grid_h
class proc;
class params;
class gridsize;
class communicator;

#include "tensor0.h"
#include "tensor1.h"
#include "particle.h"
#include <mpi.h>
#include "poisson.h"
#include <fstream>
class grid
{
  params *param_;
  proc *pc_;
  gridsize *size_;
  communicator *com_;
  poisson PS_;
  double RK4_preCoeff[4];
  double RK4_postCoeff[4];
  double mean_energy_transferred;
  std::fstream touch_check;
  std::ofstream stat_Tg;
  std::ofstream stat_Tp;
  std::ofstream stat_HT;
  std::ofstream stat_TKE2;
  std::ofstream stat_TKE;
  std::ofstream stat_TKE_U;
  std::ofstream stat_TKE_V;
  std::ofstream stat_TKE_W;
  std::ofstream stat_P0;
  std::ofstream stat_CMax;
  std::ofstream stat_CMin;
  std::ofstream stat_CMean;
  std::ofstream stat_RhoMax;
  std::ofstream stat_RhoMin;
  std::ofstream stat_RhoMean;
  std::ofstream stat_ParticleMaxCFL;
  std::ofstream stat_GasMaxCFL;
  std::ofstream stat_GasMaxDiffCFL;
  std::ofstream stat_NumIteration;
  std::ofstream stat_BalanceIndex;
  bool Is_touch_; //=1 if touch=1 =0 if not
  int sign_fnc(double); //=1 if double>=0 otherwise is -1
  double ABS(double); //=|double|
  void Write_info(); //Wrtie information to info.txt
  void open_stat_file(char*,std::ofstream&); //open a stat file
 public:
  particle part; //particle tracking class
  tensor1 RU; //rho*u stored at cell faces
  tensor1 RU_int; //previous RK4 substep value
  tensor1 RU_new; //next RK4 substep value
  tensor1 RU_np1; //next timestep value
  tensor1 RU_WP; //momentum with pressure gradient effect (should match the divergence condition)
  tensor1 RHS_RU;
  tensor1 U; //u stored at cell center
  tensor1 CU: //concentration *u stored at cell faces
  tensor0 P; //pressure stored at cell center
  tensor0 dP; //pressure delta form (used in Poisson equation) stored at cell center
  tensor0 RHS_Pois; //one part of RHS of Poisson equation
  tensor0 C; //particle concentration stored at cell center
  tensor0 Scalar_Concentration; //scalar concentration stored at cell center
  tensor0 g;//source function in scalar momentum equation
  tensor0 Scalar_Concentration_int;
  tensor0 Scalar_Concentration_new;
  tensor0 Scalar_Concentration_np1;
  tensor0 Rho; //Gas density stored at cell center
  tensor0 Rho_int;
  tensor0 Rho_new;
  tensor0 Rho_np1;
  tensor0 RHS_Rho;
  tensor0 RHS_Scalar_Concentration;
  tensor1 Rho_face; //Gas density stored at cell faces
  tensor1 Scalar_Concentration_face;//scalar concentration stored at cell faces
  tensor0 T; //Gas temperature stored at cell cneter
  tensor1 dummy; //dummt array for computations
  tensor1 dummy2; //dummt array for computations
  tensor0 divergence; //to store divergence of momentum/velocity
  tensor0 RHS_Part_Temp; //Interpolated RHS of particle energy equation (due to the algorithm it has to be saved)
  double P0; // mean thermodynamic pressure
  double P0_int;
  double P0_new;
  double P0_np1;
  double dP0_dt;    // P0 rate of change
  double sigma_RHS;   // sum of RHS of Poisson equation over all the domain exept for the P0 rate of change term
  double T_cur;
  int num_timestep;
  int RK4_count;
  grid(){}
  grid(gridsize*,params*,proc*,communicator*);
  ~grid();
  void Initialize();
  void Store();
  void TimeAdvance(); //advance one time step
  void Update_Rho();
  void Update_Scalar_Conecntration();
  void grid::C_Source();
  void Update_RU_WOP();
  void Update_P0();
  void Compute_Div_U_new();
  void Compute_RHS_Pois();
  void Solve_Poisson();
  void Update_RU_WP();
  void TimeAdvance_RK4();
  void Test_Poisson();
  void Update_Particle();
  void Statistics();
  bool Touch(); //=0/1 in touch.check (1: exit code now, 0: contunue) this function actually read from file
};
#endif               
