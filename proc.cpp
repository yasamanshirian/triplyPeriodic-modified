#include "proc.h"
#include <math.h>
proc::proc()
{
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &totalNumProcessors_);
  NumProcCalculator();
  proc_i_=myRank_%NumProcX_;
  proc_j_=(myRank_/NumProcX_)%NumProcY_;
  proc_k_=myRank_/(NumProcX_*NumProcY_);

  procRIGHT_=((proc_i_+1)%NumProcX_)+proc_j_*NumProcX_+proc_k_*NumProcX_*NumProcY_;
  procLEFT_=((proc_i_-1+NumProcX_)%NumProcX_)+proc_j_*NumProcX_+proc_k_*NumProcX_*NumProcY_;
  procTOP_=proc_i_+((proc_j_+1)%NumProcY_)*NumProcX_+proc_k_*NumProcX_*NumProcY_;
  procBOT_=proc_i_+((proc_j_-1+NumProcY_)%NumProcY_)*NumProcX_+proc_k_*NumProcX_*NumProcY_;
  procFRONT_=proc_i_+proc_j_*NumProcX_+((proc_k_+1)%NumProcZ_)*NumProcX_*NumProcY_;
  procREAR_=proc_i_+proc_j_*NumProcX_+((proc_k_-1+NumProcZ_)%NumProcZ_)*NumProcX_*NumProcY_;
  procROOT_=0;
}


void proc::NumProcCalculator()
{
  int n=totalNumProcessors_;
  int a=int(pow(n,1./3.))+1;
  for (int i(a);i>0;i--)
    if (n%i==0) 
      {
	NumProcX_=i; 
	break;
      }
  n=n/NumProcX_;
  a=int(sqrt(n))+1;
  for (int i(a);i>0;i--)
    if (n%i==0) 
      {
	NumProcY_=i; 
	break;
      }
  NumProcZ_=n/NumProcY_;
}
