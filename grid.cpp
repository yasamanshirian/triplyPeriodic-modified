#include <fstream>
#include <sstream>
#include <math.h>
#include <algorithm> 
#include <complex>
#include "proc.h"
#include "params.h"
#include "gridsize.h"
#include "communicator.h"
#include "grid.h"
#include "scalar_source.h"
#include "vector_source.h"

grid::grid(gridsize* s,params* p,proc *pc,communicator* com): RU(s,com),RU_int(s,com),RU_new(s,com),RU_np1(s,com),U_tilde(s,com),RV(s,com),RV_int(s,com),RV_new(s,com),RV_np1(s,com),RV_LES(s,com),RV_LES_int(s,com),RV_LES_new(s,com),RV_LES_np1(s,com),Passive_Scalar_int(s,com),Passive_Scalar_new(s,com),Passive_Scalar_np1(s,com),Passive_Scalar_face(s,com),Passive_Scalar(s,com),Passive_Scalar_LES_int(s,com),Passive_Scalar_LES_new(s,com),Passive_Scalar_LES_np1(s,com),Passive_Scalar_LES(s,com),RHS_Passive_Scalar(s,com), S1(s,com), S2(s,com), RU_WP(s,com),RHS_RU(s,com),U(s,com),P(s,com),dP(s,com),RHS_Pois(s,com),RHS_RV(s,com),V(s,com),V_LES(s,com), Q(s,com),RHS_Pois_Q(s,com),RHS_Pois_Q_LES(s,com),divergence(s,com),dummy(s,com),dummy2(s,com),PS_(p,pc,s,com){
  size_=s;
  param_=p;
  pc_=pc;
  com_=com;
  RK4_preCoeff[0]=0.5;
  RK4_preCoeff[1]=0.5;
  RK4_preCoeff[2]=1.;
  RK4_preCoeff[3]=1./6.;
  RK4_postCoeff[0]=1./6.;
  RK4_postCoeff[1]=1./3.;
  RK4_postCoeff[2]=1./3.;
  RK4_postCoeff[3]=1./6.;
  Is_touch_=0;
  //information about the portion of cells belong to this processor for filtering
  in_ilo=size_->il();
  in_ihi=size_->ih();
  in_jlo=size_->jl();
  in_jhi=size_->jh();
  in_klo=size_->kl();
  in_khi=size_->kh();
  out_ilo=in_ilo;
  out_ihi=in_ihi;
  out_jlo=in_jlo;
  out_jhi=in_jhi;
  out_klo=in_klo;
  out_khi=in_khi;
  bs_=size_->bs();
  nbuff_fft = (in_ihi-in_ilo)*(in_jhi-in_jlo)*(in_khi-in_klo);
  plan_kernel=fft_3d_create_plan(MPI_COMM_WORLD,size_->Nx_tot(),size_->Ny_tot(),size_->Nz_tot(),in_ilo,in_ihi,in_jlo,in_jhi,in_klo,in_khi,out_ilo,out_ihi,out_jlo,out_jhi,out_klo,out_khi,0,0,&nbuff_fft);
  plan=fft_3d_create_plan(MPI_COMM_WORLD,size_->Nx_tot(),size_->Ny_tot(),size_->Nz_tot(),in_ilo,in_ihi,in_jlo,in_jhi,in_klo,in_khi,out_ilo,out_ihi,out_jlo,out_jhi,out_klo,out_khi,0,0,&nbuff_fft);
  //dynamic memory allocation for filtered tensor
  U_fft.x = new FFT_DATA[nbuff_fft];
  U_fft.y = new FFT_DATA[nbuff_fft];
  U_fft.z = new FFT_DATA[nbuff_fft];
         
  kernel_fft =new FFT_DATA[nbuff_fft];
  
    
  if (pc->IsRoot())
    {
      touch_check.open("touch.check",std::ios::out|std::ios::trunc);
      touch_check<<0;
      touch_check.close();
      if (p->Statistics())
	{
	  //open_stat_file("Tg",stat_Tg);
	  //open_stat_file("Tp",stat_Tp);
	  //open_stat_file("HT",stat_HT);
	  open_stat_file("RU",stat_TKE2);
	  open_stat_file("TKE",stat_TKE);
	  open_stat_file("TKE_U",stat_TKE_U);
	  open_stat_file("TKE_V",stat_TKE_V);
	  open_stat_file("TKE_W",stat_TKE_W);
	  open_stat_file("Passive_Scalar_mean",stat_Passive_Scalar_mean);
          open_stat_file("TKE_V1",stat_TKEV_V1);
	  open_stat_file("TKE_V2",stat_TKEV_V2);
	  open_stat_file("TKE_V3",stat_TKEV_V3);
	  open_stat_file("P0",stat_P0);
	  //open_stat_file("C_Max",stat_CMax);
	  //open_stat_file("C_Min",stat_CMin);
	  //open_stat_file("C_Mean",stat_CMean);
	  //open_stat_file("Max_CFL_Vp",stat_ParticleMaxCFL);
	  open_stat_file("Max_CFL_U",stat_GasMaxCFL);
	  open_stat_file("Max_Diff_CFL_U",stat_GasMaxDiffCFL);
	  open_stat_file("Num_Iteration",stat_NumIteration);
	  //open_stat_file("Balance_Index",stat_BalanceIndex);
	}
    }
}

grid::~grid()
{
  
  delete[] U_fft.x;
  delete[] U_fft.y;
  delete[] U_fft.z;
  delete[] kernel_fft;
  
  if (pc_->IsRoot())
    {
      if (param_->Statistics())
	{
	  stat_TKE2.close();
	  stat_TKE.close();
	  stat_TKE_U.close();
	  stat_TKE_V.close();
	  stat_TKE_W.close();
          stat_Passive_Scalar_mean.close();
          stat_TKEV_V1.close();
	  stat_TKEV_V2.close();
	  stat_TKEV_V3.close();
	  stat_P0.close();
	  //stat_CMax.close();
	  //stat_CMin.close();
	  //stat_CMean.close();
	 // stat_ParticleMaxCFL.close();
	  stat_GasMaxCFL.close();
	  stat_GasMaxDiffCFL.close();
	  stat_NumIteration.close();
	  //stat_BalanceIndex.close();
	}
    }
}

bool grid::Touch()
{
  bool ans;
  if (pc_->IsRoot())
    {
      touch_check.open("touch.check",std::ios::in);
      touch_check>>ans;
      touch_check.close();
      if (ans) std::cout<<"Someone touched me! Simulation stopped."<<std::endl;
    }
  MPI_Bcast(&ans,1,MPI_INT,0,MPI_COMM_WORLD);
  Is_touch_=ans;
  //if (pc_->IsRoot()) std::cout << "Inside touch before Store() " << std::endl;
  Store();
  return ans;
}

int grid::sign_fnc(double a)
{
  return (a>=0)?1:-1;
}

double grid::ABS(double a)
{
  return (a>=0)?a:-a;
}

void grid::Initialize()
{
  double TWO_PI(2*3.141592653589793);
  
  double Two_PI_Over_Lx = TWO_PI/size_->Lx();
  double Two_PI_Over_Ly = TWO_PI/size_->Ly();
  double Two_PI_Over_Lz = TWO_PI/size_->Lz();

  if (param_->Initial()==5)
    {
      double X,Y,Z;
      int I,J,K;
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    {
	      I=i-size_->il()+size_->bs();
	      J=j-size_->jl()+size_->bs();
	      K=k-size_->kl()+size_->bs();
	      X=size_->dx()/2.+i*size_->dx();
	      Y=size_->dy()/2.+j*size_->dy();
	      Z=size_->dz()/2.+k*size_->dz();
	      RU.x(I,J,K) = Two_PI_Over_Ly*sin(Two_PI_Over_Lx * (X-size_->dx()/2.)) * cos(Two_PI_Over_Ly * Y);
	      RU.y(I,J,K) = -Two_PI_Over_Lx*cos(Two_PI_Over_Lx * X) * sin(Two_PI_Over_Ly * (Y-size_->dy()/2.));
	      RU.z(I,J,K)=0;
	    }
      RU.Update_Ghosts();
      Rho=param_->Rho0();
      //part.load_random();
      P=0;
      P0=param_->P0();
      T_cur=0;
      num_timestep=0;
    }

  if (param_->Initial()==4)
   {
      double X,Y,Z;
      int I,J,K;
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    {
	      I=i-size_->il()+size_->bs();
	      J=j-size_->jl()+size_->bs();
	      K=k-size_->kl()+size_->bs();
	      X=size_->dx()/2.+i*size_->dx();
	      Y=size_->dy()/2.+j*size_->dy();
	      Z=size_->dz()/2.+k*size_->dz();
	      RU.x(I,J,K) =- Two_PI_Over_Lz*cos(Two_PI_Over_Lz * Z) * sin(Two_PI_Over_Lx*(X-size_->dx()/2.));
	      RU.y(I,J,K) = 0;
	      RU.z(I,J,K) = Two_PI_Over_Lx*sin(Two_PI_Over_Lz*(Z-size_->dz()/2.)) * cos(Two_PI_Over_Lx*X);
	    }
      RU.Update_Ghosts();
      Rho=param_->Rho0();
      //part.load_random();
      P=0;
      P0=param_->P0();
      T_cur=0;
      num_timestep=0;
    }
  
  if (param_->Initial()==3)
    {
      double X,Y,Z;
      int I,J,K;
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    {
	      I=i-size_->il()+size_->bs();
	      J=j-size_->jl()+size_->bs();
	      K=k-size_->kl()+size_->bs();
	      X=size_->dx()/2.+i*size_->dx();
	      Y=size_->dy()/2.+j*size_->dy();
	      Z=size_->dz()/2.+k*size_->dz();
	      RU.x(I,J,K) = 0;
	      RU.y(I,J,K) = Two_PI_Over_Lz*sin(Two_PI_Over_Ly*(Y-size_->dy()/2.))*cos(Two_PI_Over_Lz*Z);
	      RU.z(I,J,K) = -Two_PI_Over_Ly*cos(Two_PI_Over_Ly*Y)*sin(Two_PI_Over_Lz*(Z-size_->dz()/2.));
	    }
      RU.Update_Ghosts();
      Rho=param_->Rho0();
      //part.load_random();
      P=0;
      P0=param_->P0();
      T_cur=0;
      num_timestep=0;
    }

  if (param_->Initial()==2) //user-designed initial condition
    {
      RU=0;
      Rho=param_->Rho0();
      //part.load_random();
      P=0;
      P0=param_->P0();
      T_cur=0;
      num_timestep=0;
      }
  
  if (param_->Initial()==1)
    {
      //These files are initial condition for timestep=num_timestep
      com_->read(RU,"Restart_RU.bin");
      if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=RU LOADED*+=*+=*+=*+="<<std::endl;
      Rho = param_->Rho0();

      com_->read(P,"Restart_P.bin");  
      if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=P LOADED*+=*+=*+=*+="<<std::endl;

      //part.Load_All();
      //if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=Particles LOADED*+=*+=*+=*+="<<std::endl;
      
      std::ifstream numbers("Restart_numbers.dat");
      numbers>>P0;
      numbers>>T_cur;
      numbers>>num_timestep;
      numbers.close();
      if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=Constant numbers LOADED*+=*+=*+=*+="<<std::endl;
    }

  if (param_->Initial()==0)
    {
      com_->read(RU.x,"U.bin");
      com_->read(RU.y,"V.bin");
      com_->read(RU.z,"W.bin");
      RU*=param_->Rho0();
      Rho=param_->Rho0();
      P=0;
      //part.load_random();
      P0=param_->P0();
      T_cur=0;
      num_timestep=0;
    }
    if (param_->Initial_V()==5)
    {
      double X,Y,Z;
      int I,J,K;
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    {
	      I=i-size_->il()+size_->bs();
	      J=j-size_->jl()+size_->bs();
	      K=k-size_->kl()+size_->bs();
	      X=size_->dx()/2.+i*size_->dx();
	      Y=size_->dy()/2.+j*size_->dy();
	      Z=size_->dz()/2.+k*size_->dz();
	      RV.x(I,J,K) = Two_PI_Over_Ly*sin(Two_PI_Over_Lx * (X-size_->dx()/2.)) * cos(Two_PI_Over_Ly * Y);
	      RV.y(I,J,K) = -Two_PI_Over_Lx*cos(Two_PI_Over_Lx * X) * sin(Two_PI_Over_Ly * (Y-size_->dy()/2.));
	      RV.z(I,J,K)=0;
	    }
      RV.Update_Ghosts();
      Rho_forV = param_->Rho_forV();
      Q=0;
      }

  if (param_->Initial_V()==4)
    {
      double X,Y,Z;
      int I,J,K;
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    {
	      I=i-size_->il()+size_->bs();
	      J=j-size_->jl()+size_->bs();
	      K=k-size_->kl()+size_->bs();
	      X=size_->dx()/2.+i*size_->dx();
	      Y=size_->dy()/2.+j*size_->dy();
	      Z=size_->dz()/2.+k*size_->dz();
	      RV.x(I,J,K) =- Two_PI_Over_Lz*cos(Two_PI_Over_Lz * Z) * sin(Two_PI_Over_Lx*(X-size_->dx()/2.));
	      RV.y(I,J,K) = 0;
	      RV.z(I,J,K) = Two_PI_Over_Lx*sin(Two_PI_Over_Lz*(Z-size_->dz()/2.)) * cos(Two_PI_Over_Lx*X);
	    }
      RV.Update_Ghosts();
      Rho_forV = param_->Rho_forV();
      Q=0;
      }
  
  if (param_->Initial_V()==3)
    {
      double X,Y,Z;
      int I,J,K;
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    {
	      I=i-size_->il()+size_->bs();
	      J=j-size_->jl()+size_->bs();
	      K=k-size_->kl()+size_->bs();
	      X=size_->dx()/2.+i*size_->dx();
	      Y=size_->dy()/2.+j*size_->dy();
	      Z=size_->dz()/2.+k*size_->dz();
	      RV.x(I,J,K) = 0;
	      RV.y(I,J,K) = Two_PI_Over_Lz*sin(Two_PI_Over_Ly*(Y-size_->dy()/2.))*cos(Two_PI_Over_Lz*Z);
	      RV.z(I,J,K) = -Two_PI_Over_Ly*cos(Two_PI_Over_Ly*Y)*sin(Two_PI_Over_Lz*(Z-size_->dz()/2.));
	    }
      RV.Update_Ghosts();
      Rho_forV=param_->Rho_forV();
      Q=0;
      }

  if (param_->Initial_V()==2) //user-designed initial condition
    {
      RV=0;
      Rho_forV=param_->Rho_forV();
      Q=0;
      }
  
  if (param_->Initial_V()==1)
    {
      //These files are initial condition for timestep=num_timestep
      com_->read(RV,"Restart_RV.bin");
      if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=RV LOADED*+=*+=*+=*+="<<std::endl;
      
      com_->read(Q,"Restart_Q.bin");  
      if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=Q LOADED*+=*+=*+=*+="<<std::endl;
      
      Rho_forV=param_->Rho_forV();
      }

  if (param_->Initial_V()==0)
    {
      com_->read(RV.x,"V1.bin");
      com_->read(RV.y,"V2.bin");
      com_->read(RV.z,"V3.bin");
      RV*=param_->Rho_forV();
      Rho_forV=param_->Rho_forV();
      Q=0;
      }
    if (param_->Initial_C()==5)
    {
      double X,Y,Z;
      int I,J,K;
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    {
	      I=i-size_->il()+size_->bs();
	      J=j-size_->jl()+size_->bs();
	      K=k-size_->kl()+size_->bs();
	      X=size_->dx()/2.+i*size_->dx();
	      Y=size_->dy()/2.+j*size_->dy();
	      Z=size_->dz()/2.+k*size_->dz();
	      Passive_Scalar(I,J,K) = sin(Two_PI_Over_Lx*X) * sin(Two_PI_Over_Ly*Y);
	      
	    }
      Passive_Scalar.Update_Ghosts();
    }

  if (param_->Initial_C()==4)
    {
      double X,Y,Z;
      int I,J,K;
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    {
	      I=i-size_->il()+size_->bs();
	      J=j-size_->jl()+size_->bs();
	      K=k-size_->kl()+size_->bs();
	      X=size_->dx()/2.+i*size_->dx();
	      Y=size_->dy()/2.+j*size_->dy();
	      Z=size_->dz()/2.+k*size_->dz();
	      Passive_Scalar(I,J,K) = sin(Two_PI_Over_Lz * Z) * sin(Two_PI_Over_Lx * X);
	    }
      Passive_Scalar.Update_Ghosts();
    }
  
  if (param_->Initial_C()==3)
    {
      double X,Y,Z;
      int I,J,K;
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    {
	      I=i-size_->il()+size_->bs();
	      J=j-size_->jl()+size_->bs();
	      K=k-size_->kl()+size_->bs();
	      X=size_->dx()/2.+i*size_->dx();
	      Y=size_->dy()/2.+j*size_->dy();
	      Z=size_->dz()/2.+k*size_->dz();
	      Passive_Scalar(I,J,K) = sin(Two_PI_Over_Ly * Y) * sin(Two_PI_Over_Lz * Z);
	    }
      Passive_Scalar.Update_Ghosts();
    }


    
    if (param_->Initial_C()==1)
    {
        //These files are initial condition for timestep=num_timestep
        com_->read(Passive_Scalar,"Restart_Passive_Scalar.bin");
        if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=Passive_Scalar LOADED*+=*+=*+=*+="<<std::endl;
    }
    
    if (param_->Initial_C()==0)
    {	
    	com_->read(Passive_Scalar,"Passive_Scalar.bin");
    }
    
  //prepare variables for time integration loop
  RU_int=RU;
  RU_np1=RU;
  RV_int=RV;
  RV_np1=RV;
  RV_LES_np1=RV;
  RV_LES_int=RV;
  Passive_Scalar_int = Passive_Scalar;
  Passive_Scalar_np1 = Passive_Scalar;
  Passive_Scalar_LES_int = Passive_Scalar;
  Passive_Scalar_LES_np1 = Passive_Scalar;
  //P0_int=P0;
  //P0_np1=P0;
  //Need to compute rho at faces once here, after this, Compute_RHS_Pois computes it
  //Need to compute Passive_Scalar at faces once here, after this, Update_Passive_Scalar computes it
  Passive_Scalar_face.Equal_I_C2F(Passive_Scalar);
  Write_info();
}

void grid::Store()
{
  if (num_timestep==1)
    {
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    dummy.x(i-size_->il()+size_->bs(),j-size_->jl()+size_->bs(),k-size_->kl()+size_->bs())=size_->dx()/2.+i*size_->dx();
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    dummy.y(i-size_->il()+size_->bs(),j-size_->jl()+size_->bs(),k-size_->kl()+size_->bs())=size_->dy()/2.+j*size_->dy();
      for (int k=size_->kl();k<=size_->kh();k++)
	for (int j=size_->jl();j<=size_->jh();j++)
	  for (int i=size_->il();i<=size_->ih();i++)
	    dummy.z(i-size_->il()+size_->bs(),j-size_->jl()+size_->bs(),k-size_->kl()+size_->bs())=size_->dz()/2.+k*size_->dz();
      std::ostringstream filename_out_Data;
      filename_out_Data<<param_->data_dir()<<"XYZ.bin";
      std::string filename=filename_out_Data.str();
      com_->write(dummy,(char*)(filename.c_str()));
      
      if ( param_->solve_for_scalar()){
      	filename_out_Data.str("");
      	filename_out_Data.clear();
      	filename_out_Data<<param_->data_dir()<<"S1.bin";
      	filename=filename_out_Data.str();
      	com_->write(S1,(char*)(filename.c_str()));
      }
      if (param_-> solve_for_vector()){
       filename_out_Data.str("");
       filename_out_Data.clear();
       filename_out_Data<<param_->data_dir()<<"S2.bin";
       filename=filename_out_Data.str();
       com_->write(S2,(char*)(filename.c_str()));
      }  
    }
  // MORE FREQUENT DATA STORING
  //std::cout<<"my number of steps" << num_timestep<<std::endl;
  if ((num_timestep%param_->data_freq_fast()==0)||(Is_touch_))
    { 

      std::ostringstream filename_out_Data;
      	
      if (param_->filtering()){
	filename_out_Data<<param_->data_dir()<<"U_tilde"<<"_"<<num_timestep<<".bin";
      	std::string filename=filename_out_Data.str();
      	com_->write(U_tilde,(char*)(filename.c_str()));
        filename_out_Data.str("");
        filename_out_Data.clear();
    
      	
      }
      
      
      filename_out_Data<<param_->data_dir()<<"RU"<<"_"<<num_timestep<<".bin";
      std::string filename=filename_out_Data.str();
      com_->write(RU,(char*)(filename.c_str()));
      
      if ( param_->solve_for_vector()){
       if (param_->filtering()){
       		 RV = RV_LES;
      		 filename_out_Data.str("");
      		 filename_out_Data.clear();
      		 filename_out_Data<<param_->data_dir()<<"RV_LES"<<"_"<<num_timestep<<".bin";
      		 filename=filename_out_Data.str();
      		 com_->write(RV,(char*)(filename.c_str()));
	}
       else{
		 filename_out_Data.str("");
      		 filename_out_Data.clear();
      		 filename_out_Data<<param_->data_dir()<<"RV"<<"_"<<num_timestep<<".bin";
      		 filename=filename_out_Data.str();
      		 com_->write(RV,(char*)(filename.c_str()));

	  }
      }
      /*if (param_->filtering())
      {
	filename_out_Data.str("");
        filename_out_Data.clear();
        filename_out_Data<<param_->data_dir()<<"RU_tilde"<<"_"<<num_timestep<<".bin";
        filename=filename_out_Data.str();
        com_->write(RU_tilde,(char*)(filename.c_str()));
      

      }*/
      if ( param_->solve_for_scalar()){
      	if (param_->filtering()){
       		 Passive_Scalar = Passive_Scalar_LES;      
     		 filename_out_Data.str("");
      		 filename_out_Data.clear();
      		 filename_out_Data<<param_->data_dir()<<"C_LES"<<"_"<<num_timestep<<".bin";
      		 filename=filename_out_Data.str();
      		 com_->write(Passive_Scalar,(char*)(filename.c_str()));
        }
        else{
		 filename_out_Data.str("");
      		 filename_out_Data.clear();
      		 filename_out_Data<<param_->data_dir()<<"C"<<"_"<<num_timestep<<".bin";
      		 filename=filename_out_Data.str();
      		 com_->write(Passive_Scalar,(char*)(filename.c_str()));
     
        }
      }
     
      if (pc_->IsRoot())
	{
	  filename_out_Data.str("");
	  filename_out_Data.clear();
	  filename_out_Data<<param_->data_dir()<<"numbers"<<"_"<<num_timestep<<".dat";
	  filename=filename_out_Data.str();
	  std::ofstream numbers((char*)(filename.c_str()));
	  numbers<<P0<<std::endl;
	  numbers<<T_cur<<std::endl;
	  numbers<<num_timestep<<std::endl;
	  numbers.close();
	}
      /*
      if (pc_->IsRoot())
	{
	  std::cout<<"Launching the overlap test\n";
	}
      part.isOverlap();
      */
    }
  // LESS FREQUENT DATA STORING
  if ((num_timestep%param_->data_freq_slow()==0)||(Is_touch_))
    {
      //part.Store_All(num_timestep);
      std::ostringstream filename_out_Data;
      filename_out_Data<<param_->data_dir()<<"P"<<"_"<<num_timestep<<".bin";
      std::string filename=filename_out_Data.str();
      com_->write(P,(char*)(filename.c_str()));
     
     if ( param_->solve_for_vector()){
      	 
      filename_out_Data.str("");
      filename_out_Data.clear();
      filename_out_Data<<param_->data_dir()<<"Q"<<"_"<<num_timestep<<".bin";
      filename=filename_out_Data.str();
      com_->write(Q,(char*)(filename.c_str()));
      
     }
    }
}

/**void grid::Update_Rho()
{
  RHS_Rho.Equal_Div_F2C(RU_int); //In fact, here we compute minus RHS_Rho, i.e. div(RU)
  Rho_np1.PlusEqual_Mult(-(param_->dt()*RK4_postCoeff[RK4_count]),RHS_Rho); //Update Rho_np1
  if (RK4_count!=3) Rho_new.Equal_LinComb(1,Rho,-param_->dt()*RK4_preCoeff[RK4_count],RHS_Rho); //update Rho_new
  else Rho_new=Rho_np1;
}**/

void grid::TimeAdvance()
{
 
  Passive_Scalar = Passive_Scalar_np1;
  Passive_Scalar_LES = Passive_Scalar_LES_np1;
  RU=RU_np1;
  RV=RV_np1;
  RV_LES = RV_LES_np1;
  //P0=P0_np1;
  Passive_Scalar_int = Passive_Scalar_np1;
  Passive_Scalar_LES_int = Passive_Scalar_LES_np1;
  RU_int=RU_np1;
  RV_int=RV_np1;
  RV_LES_int = RV_LES_np1;
  
  //part.x=part.x_np1; part.y=part.y_np1; part.z=part.z_np1; part.u=part.u_np1; part.v=part.v_np1; part.w=part.w_np1; part.T=part.T_np1;
  T_cur+=param_->dt();
  num_timestep++;
}


void grid::ConstructKernel()
{
  
  //number of kernel cells
  ker_Ncells = int(param_->kernel_size()/size_->dx());
  //std::cout << "number of cells : "<< ker_Ncells << std::endl;
  //number of points involved, we always have odd number of points involved.
  ker_Np = ker_Ncells;
  if (ker_Ncells % 2 == 0) ker_Np = ker_Ncells + 1;
  
 

  //std::cout << "kernel size:"<< ker_Np << std::endl;
  //std::cout << "half kernel:"<< (ker_Np -1)/2 << std::endl;


  int count;
  count=0;
  //physical values of the kernel, uniform if ker_Ncells is even, and if it's odd, two boundary cells have half value
  for (int k=in_klo;k<=in_khi;k++)
    for (int j=in_jlo;j<=in_jhi;j++)
      for (int i=in_ilo;i<=in_ihi;i++)
        {
          kernel_fft[count].im=0;
	  kernel_fft[count].re=0;
          if (i <= (ker_Np -1)/2 || i>=  size_->Nx_tot() - (ker_Np - 1)/2)
		if (j <= (ker_Np -1)/2 || j>=  size_->Ny_tot() - (ker_Np - 1)/2)
			if (k <= (ker_Np -1)/2 || k>=  size_->Nz_tot() - (ker_Np - 1)/2){
              			kernel_fft[count].re=1./(ker_Ncells*ker_Ncells*ker_Ncells);
				 if (ker_Ncells % 2 == 0){
					if (i == (ker_Np - 1)/2 || i == size_->Nx_tot() - (ker_Np - 1)/2) kernel_fft[count].re/=2;
					if (j == (ker_Np - 1)/2 || j == size_->Ny_tot() - (ker_Np - 1)/2) kernel_fft[count].re/=2;
        				if (k == (ker_Np - 1)/2 || k == size_->Nz_tot() - (ker_Np - 1)/2) kernel_fft[count].re/=2;
	  			}
			}
	  	
               
          

	 count++;
	}
  /*if( pc_->IsRoot()){
  count=0;
  
  for (int k=in_klo;k<=1;k++)
    for (int j=in_jlo;j<=in_jhi;j++)
      for (int i=in_ilo;i<=in_ihi;i++)
        {
	
	std::cout << "kernel("<< i<< ","<< j<< "," << k << ")="<< kernel_fft[count].re <<std::endl;
        std::cout << "U("<< i<< ","<< j<< "," << k << ")="<< RU_int.x(i-in_ilo+bs_,j-in_jlo+bs_,k-in_klo+bs_) <<std::endl;


	count++;
   	} 
  }*/
  fft_3d(kernel_fft,kernel_fft,1,plan_kernel);
    


}

void grid::FilterVelocity()
{
   
  U.Equal_Divide(RU_int,Rho);
 
  int count=0;
  for (int k=in_klo;k<=in_khi;k++)
    for (int j=in_jlo;j<=out_jhi;j++)
      for (int i=in_ilo;i<=out_ihi;i++)
         {
          U_fft.x[count].im=0;
          U_fft.x[count].re=U.x(i-in_ilo+bs_,j-in_jlo+bs_,k-in_klo+bs_);
	  
          U_fft.y[count].im=0;
          U_fft.y[count].re=U.y(i-in_ilo+bs_,j-in_jlo+bs_,k-in_klo+bs_);
	  
          U_fft.z[count].im=0;
          U_fft.z[count++].re=U.z(i-in_ilo+bs_,j-in_jlo+bs_,k-in_klo+bs_);
	  
         }
  
  fft_3d(U_fft.x,U_fft.x,1,plan);
  fft_3d(U_fft.y,U_fft.y,1,plan);
  fft_3d(U_fft.z,U_fft.z,1,plan);
  count=0;
  for (int k=in_klo;k<=in_khi;k++)
    for (int j=in_jlo;j<=in_jhi;j++)
      for (int i=in_ilo;i<=in_ihi;i++)
        {
          U_fft.x[count].im = U_fft.x[count].re*kernel_fft[count].im + U_fft.x[count].im*kernel_fft[count].re;
	  U_fft.x[count].re = -U_fft.x[count].im*kernel_fft[count].im + U_fft.x[count].re*kernel_fft[count].re;
          
	  U_fft.y[count].im = U_fft.y[count].re*kernel_fft[count].im + U_fft.y[count].im*kernel_fft[count].re;
          U_fft.y[count].re = -U_fft.y[count].im*kernel_fft[count].im + U_fft.y[count].re*kernel_fft[count].re;
          
	  U_fft.z[count].im = U_fft.z[count].re*kernel_fft[count].im + U_fft.z[count].im*kernel_fft[count].re;
          U_fft.z[count].re = -U_fft.z[count].im*kernel_fft[count].im + U_fft.z[count].re*kernel_fft[count].re;
          count++;
          
	}
  //IFT
  fft_3d(U_fft.x,U_fft.x,-1,plan);
  fft_3d(U_fft.y,U_fft.y,-1,plan);
  fft_3d(U_fft.z,U_fft.z,-1,plan);



  count=0;
  
  //RU_tilde=RU_int;
  //if (pc_->IsRoot()) std::cout << "RU(0,0,0): "<< RU_tilde.x(bs_,bs_,bs_) << std::endl;
  //dividing the filtered velocity by the total number of mesh points to have isometric fourier transform.  
  int tot = size_->size_tot();
   for (int k=in_klo;k<=in_khi;k++)
     for (int j=in_jlo;j<=in_jhi;j++)
       for (int i=in_ilo;i<=in_ihi;i++)
         {
         
           U_tilde.x(i-in_ilo+bs_,j-in_jlo+bs_,k-in_klo+bs_)=U_fft.x[count].re/(tot);
           U_tilde.y(i-in_ilo+bs_,j-in_jlo+bs_,k-in_klo+bs_)=U_fft.y[count].re/(tot);
           U_tilde.z(i-in_ilo+bs_,j-in_jlo+bs_,k-in_klo+bs_)=U_fft.z[count++].re/(tot);
           


	 }
  

   U_tilde.Update_Ghosts();
   //if (pc_->IsRoot()) std::cout << "RU_tilde(0,0,0): "<< RU_tilde.x(bs_,bs_,bs_) << std::endl;
   //U_tilde.Equal_Divide(RU_tilde,Rho);


}
void grid::V_Source(double T)
{
    double X,Y,Z;
    int I,J,K;
    //double (*s)(double,double,double,double) = scalar_initial;
    for (int k=size_->kl();k<=size_->kh();k++)
        for (int j=size_->jl();j<=size_->jh();j++)
            for (int i=size_->il();i<=size_->ih();i++)
            {
                I=i-size_->il()+size_->bs();
                J=j-size_->jl()+size_->bs();
                K=k-size_->kl()+size_->bs();
                
                X=i*size_->dx()+size_->dx()/2.;
                Y=j*size_->dy()+size_->dy()/2.;
                Z=k*size_->dz()+size_->dz()/2.;
		S2.x(I,J,K) = Vector_Source_x(X-size_->dx()/2.,Y,Z,T);
                S2.y(I,J,K) = Vector_Source_y(X,Y-size_->dy()/2.,Z,T);
                S2.z(I,J,K) = Vector_Source_z(X,Y,Z-size_->dz()/2.,T);
		}
    S2.Update_Ghosts();
    
    
}

void grid::Update_RV_WOQ()
{
     
   //interpolation
  V.Equal_Divide(RV_int,Rho_forV); //comupte v_int at faces (note: Rho_face is already computed from previous sub-step @ Compute_RHS_Pois)
  divergence.Equal_Div_F2C(V); //Divergence of v_int stored at cell center   Note: V at cell faces is already computed @Update_particle
  dummy2.Equal_Grad_C2F(divergence);
  dummy.Equal_Del2(V); //compute div(grad(v_i)) and store it in the dummy variable
  RHS_RV.Equal_LinComb(param_->eta0()/3.,dummy2,param_->eta0(),dummy); //RHS = -mp/Vcell*RHS + mu/3*grad(div(U)) + mu*div(grad(U))
  //convection in x direction:
  dummy.Equal_I_C2F(RV_int.x); //interpolate u to neighbour edges //Note:even though U&RU are stored on cell faces we use C2F interpolation here, should be careful!
  dummy2.Equal_Ix_C2F(U); //interpolate RU in x direction
  dummy *= dummy2;
  dummy2.x.Equal_Div_F2C(dummy);
  RHS_RV.x -= dummy2.x;
  //convection in y direction:
  dummy.Equal_I_C2F(RV_int.y); //interpolate u to neighbour edges
  dummy2.Equal_Iy_C2F(U); //interpolate RU in y direction
  dummy *= dummy2;
  dummy2.y.Equal_Div_F2C(dummy);
  RHS_RV.y -= dummy2.y;
  //convection in z direction:
  dummy.Equal_I_C2F(RV_int.z); //interpolate u to neighbour edges
  dummy2.Equal_Iz_C2F(U); //interpolate RU in z direction
  dummy *= dummy2;
  dummy2.z.Equal_Div_F2C(dummy);
  RHS_RV.z -= dummy2.z;
  //Add artificial force
  if(param_->S2_type()==0){
	grid::V_Source(T_cur);
	S2.x -= S2.x.mean();
	S2.y -= S2.y.mean();
	S2.z -= S2.z.mean();
	RHS_RV+=S2;
  }
  else if (param_->S2_type()==1){
      dummy.x.Equal_Divide(RU_int.x,Rho);
      dummy.x.Equal_Iy_C2F(dummy.x);
      RHS_RV.y.PlusEqual_Mult(-1,dummy.x);	
	
  }
  else if(param_->S2_type()==2){
    RHS_RV.PlusEqual_Mult(param_->A(),RU_int);
    
  }
  RV_np1.PlusEqual_Mult(param_->dt()*RK4_postCoeff[RK4_count],RHS_RV); //Update RV_np1
  if (RK4_count!=3) RV_new.Equal_LinComb(1,RV,param_->dt()*RK4_preCoeff[RK4_count],RHS_RV); //update RU_new
  else RV_new = RV_np1;
}

void grid::Compute_RHS_Pois_Q()
{
  //Rho_forV is considered constant
  V.Equal_Divide(RV_new,Rho_forV); //Compute V_new at cell faces and store it in U
  RHS_Pois_Q.Equal_Div_F2C(V); //Compute div(v_new_wop) and store it in RHS_Pois_Q
 
  RHS_Pois_Q *= (1./(param_->dt()*RK4_preCoeff[RK4_count]));
  RHS_Pois_Q.make_mean_zero(); //make RHS_Pois zero mean 
}

void grid::Solve_Poisson_Q()
{
  dummy = (1./Rho_forV);
  PS_.Solve(dummy,Q,RHS_Pois_Q);
}

void grid::Update_RV_WQ()
{
  dummy.Equal_Grad_C2F(Q); //Compute gradient of hydrodynamic pressure
  RV_np1.PlusEqual_Mult(-(param_->dt()*RK4_postCoeff[RK4_count]),dummy); //Update RU_np1 with pressure 
  if (RK4_count!=3) RV_new.PlusEqual_Mult(-(param_->dt()*RK4_preCoeff[RK4_count]),dummy); //update RU_new with pressure
  else RV_new=RV_np1;
}

void grid::Update_RV_LES_WOQ()
{
     
   //interpolation
  V_LES.Equal_Divide(RV_LES_int,Rho_forV); //comupte v_int at faces (note: Rho_face is already computed from previous sub-step @ Compute_RHS_Pois)
  divergence.Equal_Div_F2C(V_LES); //Divergence of v_int stored at cell center   Note: V at cell faces is already computed @Update_particle
  dummy2.Equal_Grad_C2F(divergence);
  dummy.Equal_Del2(V_LES); //compute div(grad(v_i)) and store it in the dummy variable
  RHS_RV.Equal_LinComb(param_->eta0()/3.,dummy2,param_->eta0(),dummy); //RHS = -mp/Vcell*RHS + mu/3*grad(div(U)) + mu*div(grad(U))
  //convection in x direction:
  dummy.Equal_I_C2F(RV_LES_int.x); //interpolate u to neighbour edges //Note:even though U&RU are stored on cell faces we use C2F interpolation here, should be careful!
  dummy2.Equal_Ix_C2F(U_tilde); //interpolate RU in x direction (note : U_tilde is already computed in @FilterVelocity)
  dummy *= dummy2;
  dummy2.x.Equal_Div_F2C(dummy);
  RHS_RV.x -= dummy2.x;
  //convection in y direction:
  dummy.Equal_I_C2F(RV_LES_int.y); //interpolate u to neighbour edges
  dummy2.Equal_Iy_C2F(U_tilde); //interpolate RU in y direction
  dummy *= dummy2;
  dummy2.y.Equal_Div_F2C(dummy);
  RHS_RV.y -= dummy2.y;
  //convection in z direction:
  dummy.Equal_I_C2F(RV_LES_int.z); //interpolate u to neighbour edges
  dummy2.Equal_Iz_C2F(U_tilde); //interpolate RU in z direction
  dummy *= dummy2;
  dummy2.z.Equal_Div_F2C(dummy);
  RHS_RV.z -= dummy2.z;
  //Add artificial force
  if(param_->S2_type()==0){
	grid::V_Source(T_cur);
	S2.x -= S2.x.mean();
	S2.y -= S2.y.mean();
	S2.z -= S2.z.mean();
	RHS_RV+=S2;
  }
  else if(param_->S2_type() == 1){
      //dummy.x.Equal_Divide(RU_tilde.x,Rho);
      dummy.x.Equal_Iy_C2F(U_tilde.x);
      RHS_RV.y.PlusEqual_Mult(-1,dummy.x);	
	
  }
 else if (param_->S2_type() == 2){
	RHS_RV.PlusEqual_Mult(param_->A(),RU_int);
  }
  
  RV_LES_np1.PlusEqual_Mult(param_->dt()*RK4_postCoeff[RK4_count],RHS_RV); //Update RV_np1
  if (RK4_count!=3) RV_LES_new.Equal_LinComb(1,RV_LES,param_->dt()*RK4_preCoeff[RK4_count],RHS_RV); //update RU_new
  else RV_LES_new = RV_LES_np1;
}

void grid::Compute_RHS_Pois_Q_LES()
{
  //Rho_forV is considered constant
  V_LES.Equal_Divide(RV_LES_new,Rho_forV); //Compute V_new at cell faces and store it in U
  RHS_Pois_Q_LES.Equal_Div_F2C(V_LES); //Compute div(v_new_wop) and store it in RHS_Pois_Q
  RHS_Pois_Q_LES *= (1./(param_->dt()*RK4_preCoeff[RK4_count]));
  RHS_Pois_Q_LES.make_mean_zero(); //make RHS_Pois zero mean 
}

void grid::Solve_Poisson_Q_LES()
{
  dummy = (1./Rho_forV);
  PS_.Solve(dummy,Q,RHS_Pois_Q_LES);
}

void grid::Update_RV_LES_WQ()
{
  dummy.Equal_Grad_C2F(Q); //Compute gradient of hydrodynamic pressure
  RV_LES_np1.PlusEqual_Mult(-(param_->dt()*RK4_postCoeff[RK4_count]),dummy); //Update RU_np1 with pressure 
  if (RK4_count!=3) RV_LES_new.PlusEqual_Mult(-(param_->dt()*RK4_preCoeff[RK4_count]),dummy); //update RU_new with pressure
  else RV_LES_new=RV_LES_np1;
}


void grid::C_Source(double T)
{
    double X,Y,Z;
    int I,J,K;
    for (int k=size_->kl();k<=size_->kh();k++)
        for (int j=size_->jl();j<=size_->jh();j++)
            for (int i=size_->il();i<=size_->ih();i++)
            {
                I=i-size_->il()+size_->bs();
                J=j-size_->jl()+size_->bs();
                K=k-size_->kl()+size_->bs();
                
                X=size_->dx()/2.+i*size_->dx();
                Y=size_->dy()/2.+j*size_->dy();
                Z=size_->dz()/2.+k*size_->dz();
                S1(I,J,K) = Scalar_Source(X,Y,X,T);
		}
    S1.Update_Ghosts();
        
}

void grid::Update_Passive_Scalar()
{
    
    
    U.Equal_Divide(RU_int,Rho);
    dummy.Equal_Mult(U,Passive_Scalar_face);//computing u_int at faces, note: we have already computed Passive_Scalar_face in previous RK4 substep
    dummy2.x.Equal_Div_F2C(dummy);
    dummy2.y.Equal_Del2(Passive_Scalar_int);//div(grad(C))
    grid::C_Source(T_cur);
    RHS_Passive_Scalar.Equal_LinComb(param_->D_M(),dummy2.y,-1,dummy2.x);
    if(param_->S1_type()==0){
	S1 -= S1.mean(); 
	RHS_Passive_Scalar+=S1;
	}
    else if (param_->S1_type()==1){
	dummy.x.Equal_Ix_F2C(U.x);
    	RHS_Passive_Scalar.PlusEqual_Mult(-1,dummy.x);
    }
    /*else if(param_->S1_type()==2){
     RHS_Passive_Scalar.PlusEqual_Mult(param_->A(),RU_int.x);

    }*/
    Passive_Scalar_np1.PlusEqual_Mult((param_->dt()*RK4_postCoeff[RK4_count]),RHS_Passive_Scalar); //Update Passive_Scalar_np1
    if (RK4_count!=3) Passive_Scalar_new.Equal_LinComb(1,Passive_Scalar,param_->dt()*RK4_preCoeff[RK4_count],RHS_Passive_Scalar); //update Passive_Scalar_new
    else Passive_Scalar_new=Passive_Scalar_np1;
    
    Passive_Scalar_face.Equal_I_C2F(Passive_Scalar_new);
}

void grid::Update_Passive_Scalar_LES()
{
    
    
    //U.Equal_Divide(RU_int,Rho);
    dummy.Equal_Mult(U_tilde,Passive_Scalar_face);//computing u_int at faces, note: we have already computed Passive_Scalar_face in previous RK4 substep
    dummy2.x.Equal_Div_F2C(dummy);
    dummy2.y.Equal_Del2(Passive_Scalar_LES_int);//div(grad(C))
    grid::C_Source(T_cur);
    RHS_Passive_Scalar.Equal_LinComb(param_->D_M(),dummy2.y,-1,dummy2.x);
    if(param_->S1_type()==0){
	S1 -= S1.mean(); 
	RHS_Passive_Scalar+=S1;
	}
    else if(param_->S1_type()==1){
	dummy.x.Equal_Ix_F2C(U_tilde.x);
    	RHS_Passive_Scalar.PlusEqual_Mult(-1,dummy.x);
    }
    /*else if(param_->S1_type()==2){
       RHS_Passive_Scalar.PlusEqual_Mult(param_->A(),RU_int);

    }*/
    Passive_Scalar_LES_np1.PlusEqual_Mult((param_->dt()*RK4_postCoeff[RK4_count]),RHS_Passive_Scalar); //Update Passive_Scalar_np1
    if (RK4_count!=3) Passive_Scalar_LES_new.Equal_LinComb(1,Passive_Scalar_LES,param_->dt()*RK4_preCoeff[RK4_count],RHS_Passive_Scalar); //update Passive_Scalar_new
    else Passive_Scalar_LES_new=Passive_Scalar_LES_np1;
    
    Passive_Scalar_face.Equal_I_C2F(Passive_Scalar_LES_new);
}



void grid::Update_RU_WOP()
{
  //at this point RHS_RU is equal to either zero or the values come from particle depends on TwoWayCoupling On or Off
  U.Equal_Divide(RU_int,Rho);
  divergence.Equal_Div_F2C(U); //Divergence of u_int stored at cell center   Note: U at cell faces is already computed @Update_particle
  dummy2.Equal_Grad_C2F(divergence);
  dummy.Equal_Del2(U); //compute div(grad(u_i)) and store it in the dummy variable
  RHS_RU.Equal_LinComb(param_->Mu0(),dummy); //RHS = -mp/Vcell*RHS + mu/3*grad(div(U)) + mu*div(grad(U))
  //convection in x direction:
  dummy.Equal_I_C2F(U.x); //interpolate u to neighbour edges //Note:even though U&RU are stored on cell faces we use C2F interpolation here, should be careful!
  dummy2.Equal_Ix_C2F(RU_int); //interpolate RU in x direction
  dummy *= dummy2;
  dummy2.x.Equal_Div_F2C(dummy);
  RHS_RU.x -= dummy2.x;
  //convection in y direction:
  dummy.Equal_I_C2F(U.y); //interpolate u to neighbour edges
  dummy2.Equal_Iy_C2F(RU_int); //interpolate RU in y direction
  dummy *= dummy2;
  dummy2.y.Equal_Div_F2C(dummy);
  RHS_RU.y -= dummy2.y;
  //convection in z direction:
  dummy.Equal_I_C2F(U.z); //interpolate u to neighbour edges
  dummy2.Equal_Iz_C2F(RU_int); //interpolate RU in z direction
  dummy *= dummy2;
  dummy2.z.Equal_Div_F2C(dummy);
  RHS_RU.z -= dummy2.z;
  //Add artificial force
  RHS_RU.PlusEqual_Mult(param_->A(),RU_int);
   
   RHS_RU.x += Rho*param_->gx();
   RHS_RU.y += Rho*param_->gy();
   RHS_RU.z += Rho*param_->gz();
    
  RU_np1.PlusEqual_Mult(param_->dt()*RK4_postCoeff[RK4_count],RHS_RU); //Update RU_np1
  if (RK4_count!=3) RU_new.Equal_LinComb(1,RU,param_->dt()*RK4_preCoeff[RK4_count],RHS_RU); //update RU_new
  else RU_new = RU_np1;
}

/*void grid::Update_P0()
{
  mean_energy_transferred = RHS_Part_Temp.mean(); // RHS_Part_Temp is -(watts from particles to each Eulerian cell )
  if (param_->Is_Cooling()) 
    dP0_dt=0; 
  else 
    dP0_dt = -param_->R()/param_->Cv()*mean_energy_transferred/size_->Vcell();
  
  P0_np1 += param_->dt()*RK4_postCoeff[RK4_count]*dP0_dt;
    
  if (RK4_count!=3) 
    P0_new = P0+param_->dt()*RK4_preCoeff[RK4_count]*dP0_dt;
  else 
    P0_new=P0_np1;
}*/

void grid::Compute_RHS_Pois()
{
  U.Equal_Divide(RU_new,Rho); //Compute U_new at cell faces and store it in U
  RHS_Pois.Equal_Div_F2C(U); //Compute div(u_new_wop) and store it in RHS_Pois
  RHS_Pois *= (1./(param_->dt()*RK4_preCoeff[RK4_count]));
  RHS_Pois.make_mean_zero(); //make RHS_Pois zero mean (theoritically we do not need this if dP0/dt term is included, but not computationally!)
}

void grid::Solve_Poisson()
{
  dummy = 1./Rho; //compute coefficients
  PS_.Solve(dummy,P,RHS_Pois);
}

void grid::Update_RU_WP()
{
  dummy.Equal_Grad_C2F(P); //Compute gradient of hydrodynamic pressure
  RU_np1.PlusEqual_Mult(-(param_->dt()*RK4_postCoeff[RK4_count]),dummy); //Update RU_np1 with pressure 
  if (RK4_count!=3) RU_new.PlusEqual_Mult(-(param_->dt()*RK4_preCoeff[RK4_count]),dummy); //update RU_new with pressure
  else RU_new=RU_np1;
}



void grid::TimeAdvance_RK4()
{
  RU_int=RU_new;
  RV_int=RV_new;
  RV_LES_int = RV_LES_new;
  //P0_int=P0_new;
  Passive_Scalar_int = Passive_Scalar_new;
  Passive_Scalar_LES_int = Passive_Scalar_LES_new;


  //part.x_int=part.x_new; part.y_int=part.y_new; part.z_int=part.z_new; part.u_int=part.u_new; part.v_int=part.v_new; part.w_int=part.w_new; part.T_int=part.T_new;
}

void grid::Statistics()
{
  if (!param_->Statistics()) return;
 
  double TKE2=U.mean_squares();
  if (param_->filtering()) {
	V.Equal_Divide(RV_LES,Rho_forV);
        
  	Passive_Scalar = Passive_Scalar_LES;
  }
  else  V.Equal_Divide(RV_np1,Rho_forV);
  
  double VV=V.mean_squares();
 
  double Passive_Scalar_mean = Passive_Scalar.mean_squares();
  double Particle_CFL_Max=0;
  double TKE,TKE_U,TKE_V,TKE_W;
  double TKEV,TKEV_V1,TKEV_V2,TKEV_V3;
  U.Equal_Divide(RU_np1,Rho);
  double Gas_CFL_Max=U.max_cfl(param_->dt());
  double u_max=U.max();
  U*=RU_np1;
  if (param_->filtering()) V*=RV_LES_np1;
  else V*=RV_np1;
  TKE_U=U.x.mean();
  TKE_V=U.y.mean();
  TKE_W=U.z.mean();
  TKE=TKE_U+TKE_V+TKE_W;
  TKEV_V1=V.x.mean();
  TKEV_V2=V.y.mean();
  TKEV_V3=V.z.mean();
  TKEV=TKEV_V1+TKEV_V2+TKEV_V3;
  double Gas_Max_Diff_CFL = param_->dt() / ( size_->dx() * size_->dx() * Rho / param_->Mu0()) * 6.; //when dx=dy=dz
  dummy.x = P0/param_->R()/Rho;
  if (!pc_->IsRoot()) return;
  stat_TKE2<<T_cur<<" "<<TKE2<<std::endl;
  stat_TKE<<T_cur<<" "<<TKE<<std::endl;
  stat_TKE_U<<T_cur<<" "<<TKE_U<<std::endl;
  stat_TKE_V<<T_cur<<" "<<TKE_V<<std::endl;
  stat_TKE_W<<T_cur<<" "<<TKE_W<<std::endl;
  stat_Passive_Scalar_mean<<T_cur<<" "<<Passive_Scalar_mean<<std::endl;
  stat_TKEV_V1<<T_cur<<" "<<TKEV_V1<<std::endl;
  stat_TKEV_V2<<T_cur<<" "<<TKEV_V2<<std::endl;
  stat_TKEV_V3<<T_cur<<" "<<TKEV_V3<<std::endl;
  stat_P0<<T_cur<<" "<<P0<<std::endl;
  stat_GasMaxCFL<<T_cur<<" "<<Gas_CFL_Max<<std::endl;    
  stat_GasMaxDiffCFL<<T_cur<<" "<<Gas_Max_Diff_CFL<<std::endl;
  stat_NumIteration<<T_cur<<" "<<PS_.num_iteration()<<std::endl;
  if (!param_->Stat_print()) return;
  std::cout<<std::endl<<"::::::::::TIME="<<T_cur<<"::::::::::STEP="<<num_timestep<<"::::::::::"<<std::endl;
  std::cout<</*"*** Particle Maximum CFL="<<Particle_CFL_Max<<*/"  ,  Gas Maximum CFL="<<Gas_CFL_Max<<"  ,  Gas Maximum diffusive CFL="<<Gas_Max_Diff_CFL<<std::endl;
  std::cout<<"*** P0="<<P0<<"   ,   Number of Poisson solve iterations="<<PS_.num_iteration()<<std::endl;
  std::cout<<"*** Twice TKE_U ="<<TKE_U<<"  ,  TKE_V ="<<TKE_V<<"  ,  TKE_W ="<<TKE_W<<"  , Twice TKE ="<<TKE<<"  , Twice TKE2="<<TKE2<<std::endl;
  std::cout<<"*** Passive_Scalar_mean ="<<Passive_Scalar_mean<<std::endl;
  std::cout<<"*** Twice TKEV_V1 ="<<TKEV_V1<<"  ,  TKEV_V2 ="<<TKEV_V2<<"  ,  TKEV_V3 ="<<TKEV_V3<<"  , Twice TKEV ="<<TKEV<<"  , Twice VV="<<VV<<std::endl;
  std::cout<</*"*** Particle u_max="<<Vp_max<<"  ,  Gas interpolated u_max="<<ug_max<<*/"  ,  Gas u_max="<<u_max<<std::endl;
  }


void grid::Write_info()
{

  if (pc_->IsRoot())
    {
      std::ofstream info("info.txt");
      info<<":::Global grid is "<<size_->Nx_tot()<<"x"<<size_->Ny_tot()<<"x"<<size_->Nz_tot()<<" with bordersize (number of ghost cells around local grid) ="<<size_->bs()<<std::endl;
      info<<std::endl;
      info<<":::Code started at t="<<T_cur<<" with dt="<<param_->dt()<<" ,and is supposed to finish at t="<<param_->T_final()<<std::endl;
      info<<std::endl;
      if (param_->Statistics()) info<<":::Some statistics are stored each timestep at .dat files. If files already exist code append new data to them. At each line there are two numbers: time and quantity of interest, which are separated by a space"<<std::endl;
      else info<<":::No statistics will be saved"<<std::endl;
      info<<std::endl;
      info<<":::Gas velocity field is storing each "<<param_->data_freq_fast()<< " timesteps (=each "<<param_->data_freq_fast()*param_->dt()<<" time units) at RU.bin and gas density, hydrodynamic pressure, particle concentration, and particle velocity field are storing each "<<param_->data_freq_slow()<<" timesteps (=each "<<param_->data_freq_slow()*param_->dt()<<" time units) at Rho.bin, P.bin, C.bin, and Vp.bin respectively. Moreover thermodynamic pressure (P0), current time, and number of ellapsed timesteps are being stored each "<<param_->data_freq_fast()<<" timesteps at numbers.dat. All files have a name extension according to the snapshot number in which they have been saved. You have to add the prefix \"Restart_\" to files in order to inform the code to retart code using those data."<<std::endl;
      info<<std::endl;
      info<<":::Initial condition for this run is: ";
      if (param_->Initial()==0) info<<"ARTIFICIAL TURBULENT READ FROM U.bin, V.bin, and W.bin"<<std::endl;
      if (param_->Initial()==1) info<<"RESTART FILES READ FROM Restart_RU.bin, Restart_Rho.bin, Restart_P.bin, Restart_C.bin, Restart_Vp.bin,and  Restart_numbers.dat"<<std::endl;
      if (param_->Initial()==2) info<<"USER-DESIGNED INITIAL CONDITION DEFINED IN grid::Initialize()@grid.cpp"<<std::endl;
      if (param_->Initial()==3) info<<"TAYLOR GREEN VORTEX in X DIRECTION"<<std::endl;
      if (param_->Initial()==4) info<<"TAYLOR GREEN VORTEX in Y DIRECTION"<<std::endl;
      if (param_->Initial()==5) info<<"TAYLOR GREEN VORTEX in Z DIRECTION"<<std::endl;
      info<<std::endl;
      if (param_->Is_Cooling()) info<<":::We are assuming zero net external energy from radiation, by introducing cooling term in the energy equation."<<std::endl;
      else info<<":::There is a finite net external energy to the system from radiation."<<std::endl;
      
      info<<std::endl;
      info<<":::Here is the list of dimensional quantities read from the input file:"<<std::endl;
      info<<std::endl;
      info<<"  DIMENSIONAL VARIABLE          BEFORE THRESHOLD                  AFTER THRESHOLD"<<"         UNIT"<<std::endl;
      info<<"          ----------------------------------------------------------------------------------"<<std::endl;
      info<<"          Lx                      "<<std::scientific<<param_->Lx()<<"                      "<<std::scientific<<param_->Lx()<<"           m"<<std::endl;
      info<<"          Ly----------------------"<<std::scientific<<param_->Ly()<<"----------------------"<<std::scientific<<param_->Ly()<<"-----------m"<<std::endl;
      info<<"          Lz                      "<<std::scientific<<param_->Lz()<<"                      "<<std::scientific<<param_->Lz()<<"           m"<<std::endl;
      info<<"          A-----------------------"<<std::scientific<<param_->A1()<<"----------------------"<<std::scientific<<param_->A2()<<"-----------1/s"<<std::endl;
      info<<"         Rho0                     "<<std::scientific<<param_->Rho0()<<"                      "<<std::scientific<<param_->Rho0()<<"           kg/m^3"<<std::endl;
      info<<"          U0----------------------"<<std::scientific<<param_->U0()<<"----------------------"<<std::scientific<<param_->U0()<<"-----------m/s"<<std::endl;
      info<<"          T0                      "<<std::scientific<<param_->T0()<<"                      "<<std::scientific<<param_->T0()<<"           K"<<std::endl;
      info<<"         Mu0----------------------"<<std::scientific<<param_->Mu0()<<"----------------------"<<std::scientific<<param_->Mu0()<<"-----------kg/(m.s)"<<std::endl;
      info<<"          k                       "<<std::scientific<<param_->k()<<"                      "<<std::scientific<<param_->k()<<"           kg.m/(K.s^3)"<<std::endl;
      info<<"          gx----------------------"<<std::scientific<<param_->gx1()<<"----------------------"<<std::scientific<<param_->gx2()<<"-----------m/s^2"<<std::endl;
      info<<"          gy                      "<<std::scientific<<param_->gy1()<<"                      "<<std::scientific<<param_->gy2()<<"           m/s^2"<<std::endl;
      info<<"          gz----------------------"<<std::scientific<<param_->gz1()<<"----------------------"<<std::scientific<<param_->gz2()<<"-----------m/s^2"<<std::endl;
      info<<"          Cp                      "<<std::scientific<<param_->Cp()<<"                      "<<std::scientific<<param_->Cp()<<"           J/(kg.K)"<<std::endl;
      info<<"          Cv----------------------"<<std::scientific<<param_->Cv()<<"----------------------"<<std::scientific<<param_->Cv()<<"-----------J/(kg.K)"<<std::endl;
      info<<"          n0                      "<<std::scientific<<param_->np0()<<"                      "<<std::scientific<<param_->np0()<<"           1/m^3"<<std::endl;
      info<<"          Dp----------------------"<<std::scientific<<param_->Dp()<<"----------------------"<<std::scientific<<param_->Dp()<<"-----------m"<<std::endl;
      info<<"        Rho_p                     "<<std::scientific<<param_->Rhop()<<"                      "<<std::scientific<<param_->Rhop()<<"           kg/m^3"<<std::endl;
      info<<"         Cv_p---------------------"<<std::scientific<<param_->Cvp()<<"----------------------"<<std::scientific<<param_->Cvp()<<"-----------J/(kg.K)"<<std::endl;
      info<<"       epsilon                    "<<std::scientific<<param_->epsilon()<<"                      "<<std::scientific<<param_->epsilon()<<"           "<<std::endl;
      info<<"          I0----------------------"<<std::scientific<<param_->I01()<<"----------------------"<<std::scientific<<param_->I02()<<"-----------J/(s.m^2)"<<std::endl;
      
      info<<std::endl;
      info<<":::Following dimensional quantities are resulted from the above:"<<std::endl;
      info<<std::endl;
      info<<"  DIMENSIONAL VARIABLE                VALUE"<<"                            UNIT"<<std::endl;
      info<<"          R-----------------------"<<std::scientific<<param_->R()<<"------------------------J/(kg.K)"<<std::endl;
      info<<"          P0----------------------"<<std::scientific<<param_->P0()<<"------------------------Pa"<<std::endl;
      info<<"         Vol----------------------"<<std::scientific<<param_->Vol()<<"------------------------m^3"<<std::endl;
      info<<"        Vcell---------------------"<<std::scientific<<size_->Vcell()<<"------------------------m^3"<<std::endl;
      info<<"        Tau_p---------------------"<<std::scientific<<param_->Tp()<<"------------------------s"<<std::endl;
      info<<"          mp----------------------"<<std::scientific<<param_->mp()<<"------------------------kg"<<std::endl;
      
      info<<std::endl;
      info<<":::Following non-dimensional quantities are resulted from the above:"<<std::endl;
      info<<std::endl;
      info<<"NONDIMENSIONAL VARIABLE              VALUE"<<std::endl;
      info<<"         Np0----------------------"<<std::scientific<<param_->N0()<<std::endl;
      info<<"          Re----------------------"<<std::scientific<<param_->Re()<<std::endl;
      info<<"          Fr----------------------"<<std::scientific<<param_->Fr()<<std::endl;
      info<<"          Pr----------------------"<<std::scientific<<param_->Pr()<<std::endl;
      info<<"        gamma---------------------"<<std::scientific<<param_->gamma()<<std::endl;
      info<<"        GAMMA---------------------"<<std::scientific<<param_->GAMMA()<<std::endl;
      info<<"          Nu----------------------"<<std::scientific<<param_->Nu()<<std::endl;

      info<<std::endl;
      info<<":::The poisson solver: Preconditioner: ";
      if (param_->PreCond()==0) info<<"NONE";
      if (param_->PreCond()==1) info<<"AMG";
      if (param_->PreCond()==2) info<<"Euclid";
      info<<"   ,   Solver: ";
      if (param_->Solver()==0) info<<"FFT Based"<<std::endl;
      if (param_->Solver()==1) info<<"AMG"<<std::endl;
      if (param_->Solver()==2) info<<"PCG"<<std::endl;
      if (param_->Solver()==3) info<<"GMRES"<<std::endl;
      if (param_->Solver()==4) info<<"FlexGMRES"<<std::endl;
      if (param_->Solver()==5) info<<"LGMRES"<<std::endl;
      if (param_->Solver()==6) info<<"BiCGSTAB"<<std::endl;
      info<<std::endl;
      info<<":::In this run "<<pc_->TOT()<<" processors have been exploited,  which are distributed in 3D as "<<pc_->NX()<<"x"<<pc_->NY()<<"x"<<pc_->NZ()<<" ."<<std::endl;
      info<<std::endl;
      info<<":::What does each processor do:"<<std::endl<<std::endl;
      info.close();
    }

  //Critical section begins
  {
    int x_temp=0;
    com_->Sequential_Begin(x_temp);
    std::ofstream info("info.txt",std::ios::out|std::ios::app);
    info<<"  ***Processor number: "<<pc_->RANK()<<std::endl;
    info<<"  Location in CPU grid: ("<<pc_->I()<<","<<pc_->J()<<","<<pc_->K()<<") , Neighbor processes: TOP:"<<pc_->TOP()<<" BOT:"<<pc_->BOT()<<" RIGHT:"<<pc_->RIGHT()<<" LEFT:"<<pc_->LEFT()<<" FRONT:"<<pc_->FRONT()<<" REAR:"<<pc_->REAR()<<std::endl;
    info<<"  Local grid size: "<<size_->Nx()<<"x"<<size_->Ny()<<"x"<<size_->Nz()<<", Global grid index: i:"<<size_->il()<<"->"<<size_->ih()<<" j:"<<size_->jl()<<"->"<<size_->jh()<<" k:"<<size_->kl()<<"->"<<size_->kh()<<"  ,  Global coordinate: x:"<<size_->xl()<<"->"<<size_->xh()<<" y:"<<size_->yl()<<"->"<<size_->yh()<<" z:"<<size_->zl()<<"->"<<size_->zh()<<std::endl;
    //info<<" I have initially "<<part.Np<<" particles."<<std::endl<<std::endl;
    info.close();
    com_->Sequential_End(x_temp);
  }
  //critical section ends
}

/*void grid::Test_Poisson()
{
  P=0; //guess for poisson is 0
  double k1=3;
  double k2=5;
  double k3=4;
  double X,Y,Z;
  int I,J,K;
  for (int k=size_->kl();k<=size_->kh();k++)
    for (int j=size_->jl();j<=size_->jh();j++)
      for (int i=size_->il();i<=size_->ih();i++)
	{
	  I=i-size_->il()+size_->bs();
	  J=j-size_->jl()+size_->bs();
	  K=k-size_->kl()+size_->bs();
	  X=size_->dx()/2.+i*size_->dx();
	  Y=size_->dy()/2.+j*size_->dy();
	  Z=size_->dz()/2.+k*size_->dz();
	  dummy2.x(I,J,K)=sin(k1*X)*sin(k2*Y)*sin(k3*Z);
	  Rho(I,J,K)=1+1./(10+sin(X)*sin(Y)*sin(Z));
	}
  dummy2.x.Update_Ghosts();
  Rho.Update_Ghosts();
  Rho_face.Equal_I_C2F(Rho);
  RU.Equal_Grad_C2F(dummy2.x);
  RU*=Rho_face;
  RHS_Pois.Equal_Div_F2C(RU);
  PS_.Solve(Rho_face,P,RHS_Pois);
  RU.Equal_Grad_C2F(P);
  RU*=Rho_face;
  dummy2.y.Equal_Div_F2C(RU);
  dummy2.y-=RHS_Pois;
  std::cout<<"max error ="<<dummy2.y.max_abs()<<std::endl;
  
  for (int k=size_->kl();k<=size_->kh();k++)
    for (int j=size_->jl();j<=size_->jh();j++)
      for (int i=size_->il();i<=size_->ih();i++)
	{
	  I=i-size_->il()+size_->bs();
	  J=j-size_->jl()+size_->bs();
	  K=k-size_->kl()+size_->bs();
	  X=size_->dx()/2.+i*size_->dx();
	  Y=size_->dy()/2.+j*size_->dy();
	  Z=size_->dz()/2.+k*size_->dz();
	}
  X=Rho.max();
  Y=Rho.min();
  std::cout<<"Max_Rho/Min_Rho="<<X/Y<<std::endl;
}*/



void grid::open_stat_file(const char *name,std::ofstream &file)
{
  std::string s(name);
  std::ostringstream filename_out_Data;
  filename_out_Data<<param_->stat_dir()<<s<<".dat";
  std::string filename=filename_out_Data.str();
  file.open((char*)(filename.c_str()),std::ios::out|std::ios::app);
}

void grid::CopyBox(){
  /*	this function is written only for Nx=256, Ny=Nz =64, for 48 proc with layout 4*4*3
  	sending data is in x,y,z direction seperately: This script finds whether the processor 
	needs to recieve data from other processors or to send data to the others.
	(sending means the processor is inside the 2pi,2pi,2pi box). 
  	The processor is divided into two sections as ending and beginning.
  	beginning of each processor corresponds to ending part of 2PI.  
  	whereas ending portion of the receiving processors corresponds to beginning part of 2PI
  */
  //number of processors in x direction inside the box (2pi,2pi,2pi)
  int numProcxBox =(size_->Lx()/size_->Ly() + pc_->NX() - 1)/(size_->Lx()/size_->Ly());
  
  bool sender = (pc_->I()< numProcxBox)?true:false;
  bool receiver = (pc_->I() < pc_->NX()/(size_->Lx()/size_->Lz()))?false:true;
  MPI_Request request_send[1];
  MPI_Request request_recv[1];
  MPI_Status stat_send[1];
  MPI_Status stat_recv[1];
  MPI_Datatype midmem,lastmem,recvmem;
  int start_indices_l[3];
  int lsizes[3];
  int memsizes[3];
  int rank_sender;
  int rank_receiver;
  lsizes[0]=size_->Nz()-2*size_->bs();  lsizes[1]=size_->Ny()-2*size_->bs(); 
  memsizes[0]=size_->Nz();  memsizes[1]=size_->Ny();  memsizes[2]=size_->Nx();
  start_indices_l[0]=size_->bs();  start_indices_l[1]=size_->bs(); 
  

  if(receiver && !sender){
        //finding size of receiving data
  	lsizes[2] = size_->Nx() - 2*size_->bs();
        //finding the proc which is sending data
	rank_sender = pc_->RANK() - (pc_->RANK()%pc_->NX()) *numProcxBox;
        //memory layout to receive data for beginning of the proc
        start_indices_l[2]=size_->bs();
	MPI_Type_create_subarray(3,memsizes,lsizes,start_indices_l,MPI_ORDER_C,MPI_DOUBLE,&recvmem);
	MPI_Type_commit(&recvmem);
        }
  if(sender){
       //find the size of sending data to be beginning ofthe middle processor
       //memory layout for sending data for the beinning of the middle proc
       lsizes[2] = size_->Nx() - 2*size_->bs();
       start_indices_l[2]=size_->bs();
       MPI_Type_create_subarray(3,memsizes,lsizes,start_indices_l,MPI_ORDER_C,MPI_DOUBLE,&midmem);
       MPI_Type_commit(&midmem);
       }
  
  if(receiver && !sender){
	MPI_Irecv(&RU_np1.x(0,0,0),1,recvmem,rank_sender, 0,MPI_COMM_WORLD,&request_recv[0]);			
        MPI_Waitall(1,&request_recv[0],&stat_recv[0]);  
  }
  
  if(sender){
       //sending data to middle processor
       rank_receiver = pc_->RANK() +  numProcxBox;
       MPI_Isend(&RU_np1.x(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
       //sending data to far right processor in x direction 
       MPI_Waitall(1,&request_send[0],&stat_send[0]);     
       
       rank_receiver = pc_->RANK() +  2*numProcxBox ;
       MPI_Isend(&RU_np1.x(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
       MPI_Waitall(1,&request_send[0],&stat_send[0]);
 

       rank_receiver = pc_->RANK() +  3*numProcxBox ;
       MPI_Isend(&RU_np1.x(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
       MPI_Waitall(1,&request_send[0],&stat_send[0]);
 
  
 }

if(receiver && !sender){
        MPI_Irecv(&RU_np1.y(0,0,0),1,recvmem,rank_sender, 0,MPI_COMM_WORLD,&request_recv[0]);			
        MPI_Waitall(1,&request_recv[0],&stat_recv[0]);  
}
 if(sender){
       rank_receiver = pc_->RANK() +  numProcxBox; 
       MPI_Isend(&RU_np1.y(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
       MPI_Waitall(1,&request_send[0],&stat_send[0]);
     
       rank_receiver = pc_->RANK() +  2*numProcxBox ;
       MPI_Isend(&RU_np1.y(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);		
       MPI_Waitall(1,&request_send[0],&stat_send[0]);
       
       rank_receiver = pc_->RANK() +  3*numProcxBox ;
       MPI_Isend(&RU_np1.y(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);		
       MPI_Waitall(1,&request_send[0],&stat_send[0]);


   }
if(receiver && !sender){
        MPI_Irecv(&RU_np1.z(0,0,0),1,recvmem,rank_sender, 0,MPI_COMM_WORLD,&request_recv[0]);			
        MPI_Waitall(1,&request_recv[0],&stat_recv[0]);  
}
 if(sender){
       rank_receiver = pc_->RANK() +  numProcxBox; 
       MPI_Isend(&RU_np1.z(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
       MPI_Waitall(1,&request_send[0],&stat_send[0]); 
    
       rank_receiver = pc_->RANK() +  2*numProcxBox ;
       MPI_Isend(&RU_np1.z(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);		
       MPI_Waitall(1,&request_send[0],&stat_send[0]);
       
       rank_receiver = pc_->RANK() +  3*numProcxBox ;
       MPI_Isend(&RU_np1.z(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);		
       MPI_Waitall(1,&request_send[0],&stat_send[0]);
  

  }

  RU_np1.Update_Ghosts();

}


//void grid::CopyBox(){
//  //this function is written only for Nx=128, Ny=Nz =64, for 24 proc with layout 3*2*4
//  //sending data is in x,y,z direction seperately with divinding 2 receiving processor into the beginning and ending portion
//  //beginning of each processor corresponds to ending part of 2PI-So receiving data is disected not from beginning of 2PI- 
//  //whereas ending portion of the receiving processors corresponds to beginning part of 2PI
//  int numProcxBox =(size_->Lx()/size_->Ly() + pc_->NX() - 1)/(size_->Lx()/size_->Ly());
//  bool sender = (pc_->I()< numProcxBox)?true:false;
//  bool receiver = (pc_->I() < pc_->NX()/(size_->Lx()/size_->Lz()))?false:true;
//  MPI_Request request_send[2];
//  MPI_Request request_recv[1];
//  MPI_Status stat_send[2];
//  MPI_Status stat_recv[1];
//  int batch_size[2];
//  int b_offset = size_->bs();
//  MPI_Datatype midmem,lastmem,recvmem;
//  int start_indices_l[3];
//  int lsizes[3];
//  int memsizes[3];
//  int rank_sender;
//  int rank_receiver;
//  int size_recv;
//  int N2PI = size_->Lz()/size_->dx();
//  lsizes[0]=size_->Nz()-2*size_->bs();  lsizes[1]=size_->Ny()-2*size_->bs(); 
//  memsizes[0]=size_->Nz();  memsizes[1]=size_->Ny();  memsizes[2]=size_->Nx();
//  start_indices_l[0]=size_->bs();  start_indices_l[1]=size_->bs(); 
//  
//
//  if(receiver && !sender){
//        //finding size of receiving data
//  	batch_size[0] = std::min(size_->ih() - (int)(size_->il()/N2PI+ 1 )*(int)(N2PI),
//				 size_->ih() - size_->il()) + 1; 
//	batch_size[1] = size_->Nx() - 2*size_->bs()  - batch_size[0];
//        lsizes[2] = batch_size[1];
//        //finding the proc which is sending data
//	if(pc_->RANK()%pc_->NX()==1) {
//		rank_sender = pc_->RANK() - numProcxBox; 
//		}
//	else {rank_sender = pc_->RANK() - 2*numProcxBox;}
//        size_recv = lsizes[2]*lsizes[0]*lsizes[1];
//	//memory layout to receive data for beginning of the proc
//        start_indices_l[2]=size_->bs();
//	MPI_Type_create_subarray(3,memsizes,lsizes,start_indices_l,MPI_ORDER_C,MPI_DOUBLE,&recvmem);
//	MPI_Type_commit(&recvmem);
//        //std::cout <<"my size: " << size_->Nx() - 2*size_->bs() << std::endl;	
//        
//        //std::cout << "my rank in receiving pool: "<< pc_->RANK() << " starting:  " << start_indices_l[2] << " data size:" << lsizes[2] << std::endl;
//  }
//  if(receiver && !sender){
//	MPI_Irecv(&RU.x(0,0,0),1,recvmem,rank_sender, 0,MPI_COMM_WORLD,&request_recv[0]);			
//        MPI_Waitall(1,&request_recv[0],&stat_recv[0]);  
//  }
//  if(sender){
//       //find the size of sending data to be beginning ofthe middle processor
//       batch_size[0] = size_->Nx() -2*size_->bs() -  N2PI; 
//       batch_size[1] = N2PI - batch_size[0];
//       //memory layout for sending data for the beinning of the middle proc
//       lsizes[2]=batch_size[1];
//       start_indices_l[2]=size_->bs() + batch_size[0];
//       MPI_Type_create_subarray(3,memsizes,lsizes,start_indices_l,MPI_ORDER_C,MPI_DOUBLE,&midmem);
//       MPI_Type_commit(&midmem);
//       //std::cout << "my rank in sending   pool: "<< pc_->RANK() << " starting:" << start_indices_l[2] << " data size: "<< lsizes[2] << std::endl;
//       //find the size of sending data to the beginning of the far right processor in x direction 
//       batch_size[0] = -2*(size_->Nx()-2*size_->bs()) + 3*N2PI;
//       batch_size[1] = (size_->Nx()-2*size_->bs())-batch_size[0];
//       //memory layout of sending data to the beginning of the far right in x direction 
//       lsizes[2] = batch_size[0];
//       start_indices_l[2] = size_->bs() + N2PI - batch_size[0];
//       MPI_Type_create_subarray(3,memsizes,lsizes,start_indices_l,MPI_ORDER_C,MPI_DOUBLE,&lastmem);
//       MPI_Type_commit(&lastmem);
//       //std::cout << " my_Nx: "<<  size_->Nx() -2*size_->bs() << std::endl;
//       //std::cout << "my rank in sending   pool: "<< pc_->RANK() << " starting:" << start_indices_l[2] << " data size: "<< lsizes[2] << std::endl; 
//  }
// if(sender){
//       //sending data to middle processor
//       rank_receiver = pc_->RANK() +  numProcxBox;
//       MPI_Isend(&RU.x(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
//       //sending data to far right processor in x direction 
//       MPI_Waitall(1,&request_send[0],&stat_send[0]);     
//       rank_receiver = pc_->RANK() +  2*numProcxBox ;
//       MPI_Isend(&RU.x(0,0,0),1,lastmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
//       //copy data within sender processor if size of sender processor is larger than 2PI		
//       if(receiver){
//	       batch_size[0]= size_->Nx()-2*size_->bs() - N2PI;
//	       batch_size[1]= N2PI ;
//	       for (int i(size_->bs()); i<batch_size[0]+size_->bs(); i++)
//		       for (int j(size_->bs()); j<size_->Ny()-size_->bs(); j++)
//			       for (int k(size_->bs()); k<size_->Nz()-size_->bs(); k++){
//						RU.x(i+batch_size[1],j,k) = RU.x(i,j,k);
//						RU.y(i+batch_size[1],j,k) = RU.y(i,j,k);
//						RU.z(i+batch_size[1],j,k) = RU.z(i,j,k);
//				}
//        
//       MPI_Waitall(1,&request_send[0],&stat_send[0]);
// 
//  }
// }
//
//if(receiver && !sender){
//        MPI_Irecv(&RU.y(0,0,0),1,recvmem,rank_sender, 0,MPI_COMM_WORLD,&request_recv[0]);			
//        MPI_Waitall(1,&request_recv[0],&stat_recv[0]);  
//}
// if(sender){
//       rank_receiver = pc_->RANK() +  numProcxBox; 
//       MPI_Isend(&RU.y(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
//       MPI_Waitall(1,&request_send[0],&stat_send[0]);
//     
//       rank_receiver = pc_->RANK() +  2*numProcxBox ;
//       MPI_Isend(&RU.y(0,0,0),1,lastmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);		
//       MPI_Waitall(1,&request_send[0],&stat_send[0]);
//   }
//if(receiver && !sender){
//        MPI_Irecv(&RU.z(0,0,0),1,recvmem,rank_sender, 0,MPI_COMM_WORLD,&request_recv[0]);			
//        MPI_Waitall(1,&request_recv[0],&stat_recv[0]);  
//}
// if(sender){
//       rank_receiver = pc_->RANK() +  numProcxBox; 
//       MPI_Isend(&RU.z(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
//       MPI_Waitall(1,&request_send[0],&stat_send[0]); 
//    
//       rank_receiver = pc_->RANK() +  2*numProcxBox ;
//       MPI_Isend(&RU.z(0,0,0),1,lastmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);		
//       MPI_Waitall(1,&request_send[0],&stat_send[0]);
//  
//  }
//if(receiver && !sender){
//	//find receiving size for ending section of the processor
//	batch_size[0] = std::min(size_->ih() - (int)(size_->il()/N2PI+ 1 )*(int)(N2PI),
//				 size_->ih() - size_->il()) + 1; 
//
//        batch_size[1] = size_->Nx() - 2*size_->bs() - batch_size[0] ;
//	size_recv = (batch_size[0])*lsizes[0]*lsizes[1];
//        //memory layout for ending section of the processor
//	lsizes[2] = batch_size[0];
//        start_indices_l[2]=size_->bs() + batch_size[1];
//        
//        MPI_Type_free(&recvmem);
//	MPI_Type_create_subarray(3,memsizes,lsizes,start_indices_l,MPI_ORDER_C,MPI_DOUBLE,&recvmem);
//        MPI_Type_commit(&recvmem);
//	//rank_sender = pc_->RANK() - 2*numProcxBox; 
//        //std::cout << "my rank in receiving pool: "<< pc_->RANK() << " starting: " << start_indices_l[2] << " data size: "<< batch_size[0] << std::endl;
//   
//}
//if(receiver && !sender){
//	MPI_Irecv(&RU.x(0,0,0),1,recvmem,rank_sender, 0,MPI_COMM_WORLD,&request_recv[0]);
//        MPI_Waitall(1,&request_recv[0],&stat_recv[0]);  
//}
// if(sender){
//
//       int MidPCNx=(((pc_->I() + 1)<(size_->Nx_tot()%pc_->NX()))?size_->Nx_tot()/pc_->NX()+1:size_->Nx_tot()/pc_->NX());
//       batch_size[0] = size_->Nx()-2*size_->bs() + MidPCNx - 2*(N2PI); 
//       MPI_Type_free(&midmem);
//       //memory layout for ending part of middle processor
//       lsizes[2]=batch_size[0];
//       start_indices_l[2]=size_->bs();
//       MPI_Type_create_subarray(3,memsizes,lsizes,start_indices_l,MPI_ORDER_C,MPI_DOUBLE,&midmem);
//       MPI_Type_commit(&midmem);
//       //std::cout << "my rank in sending   pool: "<< pc_->RANK() << " starting:" << start_indices_l[2] << " data size: "<< lsizes[2] << std::endl;
//   
//       //memory layout for ending part of far right processor in x direction
//       batch_size[0] = N2PI;
//       MPI_Type_free(&lastmem);
//       lsizes[2]=batch_size[0];
//       MPI_Type_create_subarray(3,memsizes,lsizes,start_indices_l,MPI_ORDER_C,MPI_DOUBLE,&lastmem);
//       MPI_Type_commit(&lastmem);
//       //std::cout << "my rank in sending   pool: "<< pc_->RANK() << " starting:" << start_indices_l[2] << " data size: "<< lsizes[2] << std::endl;
//   
// 
//       }
// if(sender){
//       rank_receiver = pc_->RANK() +  numProcxBox; 
//       MPI_Isend(&RU.x(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
//       MPI_Waitall(1,&request_send[0],&stat_send[0]);
//
//       rank_receiver = pc_->RANK() +  2*numProcxBox ;
//       MPI_Isend(&RU.x(0,0,0),1,lastmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
//       MPI_Waitall(1,&request_send[0],&stat_send[0]);
//  }
//if(receiver && !sender){
//       MPI_Irecv(&RU.y(0,0,0),1,recvmem,rank_sender, 0,MPI_COMM_WORLD,&request_recv[0]);
//       MPI_Waitall(1,&request_recv[0],&stat_recv[0]);  
//}
// if(sender){
//      
//       rank_receiver = pc_->RANK() +  numProcxBox; 
//       MPI_Isend(&RU.y(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
//       MPI_Waitall(1,&request_send[0],&stat_send[0]); 
// 
//       rank_receiver = pc_->RANK() +  2*numProcxBox ;
//       MPI_Isend(&RU.y(0,0,0),1,lastmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
//       MPI_Waitall(1,&request_send[0],&stat_send[0]);
//     
//   }
//if(receiver && !sender){
////if(pc_->RANK() %pc_->NX() == (pc_->NX() -1)){
//        MPI_Irecv(&RU.z(0,0,0),1,recvmem,rank_sender,0 ,MPI_COMM_WORLD,&request_recv[0]);
//	MPI_Waitall(1,&request_recv[0],&stat_recv[0]);  
//}
// if(sender){
//   
//	rank_receiver = pc_->RANK() +  numProcxBox; 
//        MPI_Isend(&RU.z(0,0,0),1,midmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);
//        MPI_Waitall(1,&request_send[0],&stat_send[0]); 
//       
// 
//       rank_receiver = pc_->RANK() +  2*numProcxBox ;
//       MPI_Isend(&RU.z(0,0,0),1,lastmem,rank_receiver, 0,MPI_COMM_WORLD,&request_send[0]);	
//       MPI_Waitall(1,&request_send[0],&stat_send[0]); 
//   }
//  RU.Update_Ghosts();
//
//}
