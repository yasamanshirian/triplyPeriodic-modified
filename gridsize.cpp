#include "params.h"
#include "proc.h"
#include "gridsize.h"
gridsize::gridsize(params* par,proc *P)
{
  pc_=P;
  
  Nx_tot_ = par->Nx_tot();
  Ny_tot_ = par->Ny_tot();
  Nz_tot_ = par->Nz_tot();
  
  bs_ = par->bs();
  Lx_ = par->Lx();
  Ly_ = par->Ly();
  Lz_ = par->Lz();
  
  //determine number of local grid points per process, including ghost cells

  Nx_=((P->I()<(Nx_tot_%P->NX()))?Nx_tot_/P->NX()+1:Nx_tot_/P->NX())+2*par->bs();
  Ny_=((P->J()<(Ny_tot_%P->NY()))?Ny_tot_/P->NY()+1:Ny_tot_/P->NY())+2*par->bs();
  Nz_=((P->K()<(Nz_tot_%P->NZ()))?Nz_tot_/P->NZ()+1:Nz_tot_/P->NZ())+2*par->bs();

  dx_=Lx_/Nx_tot_;
  dy_=Ly_/Ny_tot_;
  dz_=Lz_/Nz_tot_;
  
  if (P->IsRoot())
    {
      Nxs_=new int[P->TOT()];  Nys_=new int[P->TOT()];  Nzs_=new int[P->TOT()];
      OFFSET_x_=new int[P->TOT()];  OFFSET_y_=new int[P->TOT()];  OFFSET_z_=new int[P->TOT()];
    }
  MPI_Gather(&Nx_,1,MPI_INT,Nxs_,1,MPI_INT,P->ROOT(),MPI_COMM_WORLD); //root process gets Nx from all processes 
  MPI_Gather(&Ny_,1,MPI_INT,Nys_,1,MPI_INT,P->ROOT(),MPI_COMM_WORLD); //root process gets Ny from all processes 
  MPI_Gather(&Nz_,1,MPI_INT,Nzs_,1,MPI_INT,P->ROOT(),MPI_COMM_WORLD); //root process gets Nz from all processes 
  if (P->IsRoot())
    {
      for (int i(0);i<P->TOT();i++) //first subtract ghost cells
	{
	  Nxs_[i]-=2*bs_; 
	  Nys_[i]-=2*bs_; 
	  Nzs_[i]-=2*bs_; 
	}
      
      int rank(0);
      for (int k(0);k<P->NZ();k++)
	for (int j(0);j<P->NY();j++)
	  for (int i(0);i<P->NX();i++)
	    {
	      OFFSET_x_[rank]=0;
	      for (int ii(0);ii<i;ii++)
		OFFSET_x_[rank]+=Nxs_[ii+j*P->NX()+k*P->NX()*P->NY()];
	      
	      OFFSET_y_[rank]=0;
	      for (int jj(0);jj<j;jj++)
		OFFSET_y_[rank]+=Nys_[i+jj*P->NX()+k*P->NX()*P->NY()];
	      
	      OFFSET_z_[rank]=0;
	      for (int kk(0);kk<k;kk++)
		OFFSET_z_[rank]+=Nzs_[i+j*P->NX()+kk*P->NX()*P->NY()];
	      rank++;
	    } 
    }
  MPI_Scatter(OFFSET_x_,1,MPI_INT,&il_,1,MPI_INT,P->ROOT(),MPI_COMM_WORLD);
  MPI_Scatter(OFFSET_y_,1,MPI_INT,&jl_,1,MPI_INT,P->ROOT(),MPI_COMM_WORLD);
  MPI_Scatter(OFFSET_z_,1,MPI_INT,&kl_,1,MPI_INT,P->ROOT(),MPI_COMM_WORLD);
 
  ih_=il_+Nx_-2*bs_-1;
  jh_=jl_+Ny_-2*bs_-1;
  kh_=kl_+Nz_-2*bs_-1;
  

  xl_=il_*dx_;
  xh_=(ih_+1)*dx_;
  yl_=jl_*dy_;
  yh_=(jh_+1)*dy_;
  zl_=kl_*dz_;
  zh_=(kh_+1)*dz_;
  
}

gridsize::~gridsize()
{
  if (pc_->IsRoot())
    {
      delete [] Nxs_;  delete [] Nys_;  delete [] Nzs_;
      delete [] OFFSET_x_;  delete [] OFFSET_y_;  delete [] OFFSET_z_;
    }
}
