#include <iostream>
#include <mpi.h>
#include <cmath>
#include <chrono>
#include "params.h"
#include "proc.h"
#include "gridsize.h"
#include "tensor0.h"
#include "tensor1.h"
#include "communicator.h"
#include "poisson.h"
#include "grid.h"

using namespace std::chrono;

int main (int argc,char *argv[] ) 
{
  if (argc != 2) 
    {
      std::cout << "Please supply a parameter file!" << std::endl;
      exit(1);
    }
  MPI_Init(&argc, &argv);
  // Define required objects
  params PARAM(argv[1]);
  double U_mean[3],RhoU_[3];
  double Rho0_=PARAM.Rho0();
  
  U_mean[0]=PARAM.U0();
  U_mean[1]=PARAM.V0();
  U_mean[2]=PARAM.W0();
  
  RhoU_[0]=U_mean[0]*Rho0_;
  RhoU_[1]=U_mean[1]*Rho0_;
  RhoU_[2]=U_mean[2]*Rho0_;
  
  double V_mean[3],RhoV_[3];
  double Rho0_forV =PARAM.Rho_forV();
  
  V_mean[0]=PARAM.V0_1();
  V_mean[1]=PARAM.V0_2();
  V_mean[2]=PARAM.V0_3();
  
  RhoV_[0]=V_mean[0]*Rho0_forV;
  RhoV_[1]=V_mean[1]*Rho0_forV;
  RhoV_[2]=V_mean[2]*Rho0_forV;
  
  
  proc PROC;
  gridsize GSIZE(&PARAM,&PROC);
  communicator COMM(&GSIZE,&PARAM,&PROC);
  grid GRID(&GSIZE,&PARAM,&PROC,&COMM);
  GRID.Initialize();
  // Time integration loop
  auto start = high_resolution_clock::now();
  do {
      PARAM.update(GRID.T_cur);
      // RK4 loop
      for (GRID.RK4_count=0;GRID.RK4_count<4;GRID.RK4_count++)
	{
	  //GRID.Update_Rho();
	  //GRID.Update_P0();
	  //GRID.Update_Particle();
	  if (PARAM.solve_for_scalar())GRID.Update_Passive_Scalar();
          if (PARAM.solve_for_vector()){
 	  	GRID.Update_RV_WOQ();
	  	GRID.Compute_RHS_Pois_Q();
	  	GRID.Solve_Poisson_Q();
	  	GRID.Update_RV_WQ();
          }
          GRID.Update_RU_WOP();
	  //GRID.Compute_Div_U_new();
	  GRID.Compute_RHS_Pois();
	  GRID.Solve_Poisson();
	  GRID.Update_RU_WP();
	  GRID.TimeAdvance_RK4();
	}
     
      GRID.RU_np1.make_mean_U0(RhoU_);
      if(PARAM.elongated_box() == 1){
      	GRID.RU_np1.y.kill_strong_modes();
      	GRID.RU_np1.z.kill_strong_modes();
      }
      if(PARAM.elongated_box() == 2){
      	if(GRID.num_timestep % 100 == 0) GRID.CopyBox();
                
      }
      //GRID.RV_np1.make_mean_U0(RhoV_);
      //GRID.RU_np1.make_mean_zero();
      GRID.TimeAdvance();
      GRID.Statistics();
  }while ((GRID.T_cur<PARAM.T_final())&&(!GRID.Touch()));
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop-start);
  cout << "run time: " << duration.count() << 
        << " average time per RK4 step : " << duration.count()*PARAM.T_final()/PARAM.dt() << endl;
  MPI_Finalize();
  
}
 
