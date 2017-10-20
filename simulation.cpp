#include <iostream>
#include <mpi.h>
#include "params.h"
#include "proc.h"
#include "gridsize.h"
#include "tensor0.h"
#include "tensor1.h"
#include "communicator.h"
#include "poisson.h"
#include "grid.h"
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
  proc PROC;
  gridsize GSIZE(&PARAM,&PROC);
  communicator COMM(&GSIZE,&PARAM,&PROC);
  grid GRID(&GSIZE,&PARAM,&PROC,&COMM);
  GRID.Initialize();
  // Time integration loop
  do {
      PARAM.update(GRID.T_cur);
      // RK4 loop
      for (GRID.RK4_count=0;GRID.RK4_count<4;GRID.RK4_count++)
	{
	  GRID.Update_Rho();
	  GRID.Update_P0();
	  GRID.Update_Particle();
	  GRID.Update_RU_WOP();
      GRID.Update_Scalar_Concentration();
	  GRID.Compute_Div_U_new();
	  GRID.Compute_RHS_Pois();
	  GRID.Solve_Poisson();
	  GRID.Update_RU_WP();
	  GRID.TimeAdvance_RK4();
	}
      //GRID.RU_np1.make_mean_zero();
      GRID.TimeAdvance();
      GRID.Statistics();
  }while ((GRID.T_cur<PARAM.T_final())&&(!GRID.Touch()));
  //std::cout<< "before finalizing mpi"<<std::endl;
  MPI_Finalize();
  //std::cout<< "after finalizing mpi"<<std::endl;
}
  
