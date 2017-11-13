#ifndef poisson_h
#define poisson_h
class params;
class proc;
class gridsize;
class tensor0;
class tensor1;
class communicator;
extern "C" 
{
#include "/home/yshirian/tools/fft/fft_3d.h" //required for SANDIA parallel fft package (using FFTW)
/* required for Hypre package */
#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
}
class poisson
{
  params *param_;
  proc *pc_;
  gridsize *size_;
  double epsilon_; //convergence criterion
  int length_; //length of data each process has which is asumed to be the same for all processes
  /* fft package required variables */
  fft_plan_3d *plan;
  FFT_DATA *fft_data;
  int nbuff;
  int in_ilo,in_ihi,in_jlo,in_jhi,in_klo,in_khi;
  int out_ilo,out_ihi,out_jlo,out_jhi,out_klo,out_khi;
  int *lengthS_; //get size of each process grid (need to compute ilower in the BIG MATRIX).
  /* Hypre package required variables*/
  HYPRE_IJMatrix A;
  HYPRE_IJVector b;
  HYPRE_IJVector x;
  HYPRE_Solver solver, precond;
  int nnz; //number of nonzeros per row in the Big matrix
  double values[7]; //in this case each row has exactly 7 non-zero elements. This array keep the values
  int cols[7]; // this array keep the j index of non-zero elements of the row
  /* assuming constant mesh size */
  double inv_dx2;
  double inv_dy2;
  double inv_dz2;

  /* Hypre package required variables: ParCSR format*/
  HYPRE_ParCSRMatrix parcsr_A;
  HYPRE_ParVector par_b;
  HYPRE_ParVector par_x;

  int bs_; //bordersize
  int Nx_tot_,Ny_tot_,Nz_tot_,NxNy_tot_; //global grid size
  int Nx_,Ny_,Nz_; //local grid size (including the ghost cells)
  int ilower,iupper; //range of rows "owned" by this process in the Big matrix
  //int Index(int,int,int); //get local 3d indecies and return Big matrix index (used in Hypre solver)
  tensor0 Index; //Hash table incluse Big Matrix index (used in Hypre solver)
  double *rhs_values, *x_values; //extra memoty required for Hypre
  int *rows; //extra memory required for Hypre


  tensor0 center_coeff,RHS_new; //extra memory required for Iterative Poisson solve
  tensor1 dummy;//extra memory required for Iterative Poisson solve

  void CCP_FFT(fft_plan_3d*,FFT_DATA*,int,int,int,int,tensor0&,tensor0&,int,double,double,double); //Solve Constant Coefficient Possion using FFT

  void FFT(tensor1&,tensor0&,tensor0&); //Iterative Constant Coefficient solve (FFT based)
  void AMG(tensor1&,tensor0&,tensor0&); //AMG Solver
  void PCG(tensor1&,tensor0&,tensor0&); //PCG Solver
  void GMRES(tensor1&,tensor0&,tensor0&); //GMRES Solver
  void FlexGMRES(tensor1&,tensor0&,tensor0&); //FlexGMRES Solver
  void LGMRES(tensor1&,tensor0&,tensor0&); //LGMRES Solver
  void BiCGSTAB(tensor1&,tensor0&,tensor0&); //BiCGSTAB Solver

  int num_iteration_; //actual number of iteration required for convergence
 public:
  poisson(){}
  ~poisson();
  poisson(params*,proc*,gridsize*,communicator*);
  void Solve(tensor1&,tensor0&,tensor0&); //(C,P,RHS) div(C*grad(P))=RHS   P should includes your initial guess, Note: C is a vector quantity corresponding to coefficeints evaluated on cel faces, VCP:Variable Coefficient Poisson
  int num_iteration() const {return num_iteration_;}//This function returns the actual number of iteration it takes to converge
};

#endif
