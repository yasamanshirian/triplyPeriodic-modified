#include <fstream>
#include <sstream>
#include <math.h>
#include "proc.h"
#include "params.h"
#include "gridsize.h"
#include "communicator.h"
#include "grid.h"

grid::grid(gridsize* s,params* p,proc *pc,communicator* com): RU(s,com),RU_int(s,com),RU_new(s,com),RU_np1(s,com),Scalar_Concentration_int(s,com),Scalar_Concentration_new(s,com),Scalar_Concentration_np1(s,com),Scalar_Concentration_face(s,com),Scalar_Concentration(s,com),RHS_Scalar_Concentration(s,com), g(s,com), RU_WP(s,com),RHS_RU(s,com),U(s,com),P(s,com),dP(s,com),RHS_Pois(s,com),C(s,com),Rho(s,com),Rho_int(s,com),Rho_new(s,com),Rho_np1(s,com),RHS_Rho(s,com),Rho_face(s,com),T(s,com),dummy(s,com),dummy2(s,com),divergence(s,com),RHS_Part_Temp(s,com),PS_(p,pc,s,com),part(p,pc,s)
{
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
  if (pc->IsRoot())
    {
      touch_check.open("touch.check",std::ios::out|std::ios::trunc);
      touch_check<<0;
      touch_check.close();
      if (p->Statistics())
	{
	  open_stat_file("Tg",stat_Tg);
	  open_stat_file("Tp",stat_Tp);
	  open_stat_file("HT",stat_HT);
	  open_stat_file("RU",stat_TKE2);
	  open_stat_file("TKE_U",stat_TKE_U);
	  open_stat_file("TKE_V",stat_TKE_V);
	  open_stat_file("TKE_W",stat_TKE_W);
          open_stat_file("Scalar_Concentration_rms",stat_Scalar_Concentration_rms);
	  open_stat_file("P0",stat_P0);
	  open_stat_file("C_Max",stat_CMax);
	  open_stat_file("C_Min",stat_CMin);
	  open_stat_file("C_Mean",stat_CMean);
	  open_stat_file("Rho_Max",stat_RhoMax);
	  open_stat_file("Rho_Min",stat_RhoMin);
	  open_stat_file("Rho_Mean",stat_RhoMean);
	  open_stat_file("Max_CFL_Vp",stat_ParticleMaxCFL);
	  open_stat_file("Max_CFL_U",stat_GasMaxCFL);
	  open_stat_file("Num_Iteration",stat_NumIteration);
	  open_stat_file("Balance_Index",stat_BalanceIndex);
	}
    }
}

grid::~grid()
{
  if (pc_->IsRoot())
    {
      if (param_->Statistics())
	{
	  stat_TKE2.close();
	  stat_TKE.close();
	  stat_TKE_U.close();
	  stat_TKE_V.close();
	  stat_TKE_W.close();
          stat_Scalar_Concentration_rms.close();
	  stat_P0.close();
	  stat_CMax.close();
	  stat_CMin.close();
	  stat_CMean.close();
	  stat_RhoMax.close();
	  stat_RhoMin.close();
	  stat_RhoMean.close();
	  stat_ParticleMaxCFL.close();
	  stat_GasMaxCFL.close();
	  stat_GasMaxDiffCFL.close();
	  stat_NumIteration.close();
	  stat_BalanceIndex.close();
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
      part.load_random();
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
      part.load_random();
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
      part.load_random();
      P=0;
      P0=param_->P0();
      T_cur=0;
      num_timestep=0;
    }

  if (param_->Initial()==2) //user-designed initial condition
    {
      RU=0;
      Rho=param_->Rho0();
      part.load_random();
      P=0;
      P0=param_->P0();
      T_cur=0;
      num_timestep=0;
      Rho_face.Equal_I_C2F(Rho);
    }
  
  if (param_->Initial()==1)
    {
      //These files are initial condition for timestep=num_timestep
      com_->read(RU,"Restart_RU.bin");
      if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=RU LOADED*+=*+=*+=*+="<<std::endl;
      com_->read(Rho,"Restart_Rho.bin");
      if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=Rho LOADED*+=*+=*+=*+="<<std::endl;
      com_->read(P,"Restart_P.bin");  
      if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=P LOADED*+=*+=*+=*+="<<std::endl;
      part.Load_All();
      if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=Particles LOADED*+=*+=*+=*+="<<std::endl;
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
      //std::cout <<"U:" << RU.x.mean()<< std::endl;
      //std::cout <<"V:" << RU.y.mean()<< std::endl;
      //std::cout <<"W:" << RU.z.mean()<< std::endl;
      RU*=param_->Rho0();
      Rho=param_->Rho0();
      P=0;
      part.load_random();
      P0=param_->P0();
      T_cur=0;
      num_timestep=0;
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
	      Scalar_Concentration(I,J,K) = sin(Two_PI_Over_Lx*X) * sin(Two_PI_Over_Ly*Y);
	      
	    }
      Scalar_Concentration.Update_Ghosts();
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
	      Scalar_Concentration(I,J,K) = sin(Two_PI_Over_Lz * Z) * sin(Two_PI_Over_Lx * X);
	    }
      Scalar_Concentration.Update_Ghosts();
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
	      Scalar_Concentration(I,J,K) = sin(Two_PI_Over_Ly * Y) * sin(Two_PI_Over_Lz * Z);
	    }
      Scalar_Concentration.Update_Ghosts();
    }


    
    if (param_->Initial_C()==1)
    {
        //These files are initial condition for timestep=num_timestep
        com_->read(Scalar_Concentration,"Restart_Scalar_Concentration.bin");
        if (pc_->IsRoot()) std::cout<<"*+=*+=*+=*+=Scalar_Concentration LOADED*+=*+=*+=*+="<<std::endl;
    }
    
    if (param_->Initial_C()==0)
    {	
    	//std::cout<<"befor reading file"<<std::endl;
        com_->read(Scalar_Concentration,"Scalar_Concentration.bin");
        //std::cout<<"after reading file"<<std::endl;

    }
    
  //prepare variables for time integration loop
  RU_int=RU;
  RU_np1=RU;
  Scalar_Concentration_int = Scalar_Concentration;
  Scalar_Concentration_np1 = Scalar_Concentration;
  Rho_int=Rho;
  Rho_np1=Rho;
  P0_int=P0;
  P0_np1=P0;
  //Need to compute rho at faces once here, after this, Compute_RHS_Pois computes it
  Rho_face.Equal_I_C2F(Rho);
  //Need to compute Scalar_Concentration at faces once here, after this, Update_Scalar_Concentration computes it
  Scalar_Concentration_face.Equal_I_C2F(Scalar_Concentration);
  //particle part
  part.x_int=part.x; part.y_int=part.y; part.z_int=part.z; part.u_int=part.u; part.v_int=part.v; part.w_int=part.w; part.T_int=part.T;
  part.x_np1=part.x; part.y_np1=part.y; part.z_np1=part.z; part.u_np1=part.u; part.v_np1=part.v; part.w_np1=part.w; part.T_np1=part.T;
  //Need to compute RHS of particle energy equation before enetering RK4 loop
  T.Equal_Divide(P0_int / param_->R(),Rho_int);
  part.gas2part_Temp_int(T);
  part.Compute_RHS_Temp_int();
  part.part2gas_Temp_int(RHS_Part_Temp);
  
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
        
      filename_out_Data.str("");
      filename_out_Data.clear();
      filename_out_Data<<param_->data_dir()<<"g.bin";
      filename=filename_out_Data.str();
      com_->write(g,(char*)(filename.c_str()));
        
        
    }
  // MORE FREQUENT DATA STORING
  //std::cout<<"my number of steps" << num_timestep<<std::endl;
  if ((num_timestep%param_->data_freq_fast()==0)||(Is_touch_))
    { 
      //std::cout<<"before writing RU in file" <<std::endl;
      std::ostringstream filename_out_Data;
      filename_out_Data<<param_->data_dir()<<"RU"<<"_"<<num_timestep<<".bin";
      std::string filename=filename_out_Data.str();
      com_->write(RU,(char*)(filename.c_str()));
      //std::cout<<"after writing RU in file" <<std::endl;

      //std::cout<<"before writing C in file" <<std::endl; 
      filename_out_Data.str("");
      filename_out_Data.clear();
      filename_out_Data<<param_->data_dir()<<"C"<<"_"<<num_timestep<<".bin";
      filename=filename_out_Data.str();
      com_->write(Scalar_Concentration,(char*)(filename.c_str()));
      //std::cout<<"after writing C in file" <<std::endl;
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
      part.Store_All(num_timestep);
      std::ostringstream filename_out_Data;
      filename_out_Data<<param_->data_dir()<<"P"<<"_"<<num_timestep<<".bin";
      std::string filename=filename_out_Data.str();
      com_->write(P,(char*)(filename.c_str()));
      filename_out_Data.str("");
      filename_out_Data.clear();
      filename_out_Data<<param_->data_dir()<<"Rho"<<"_"<<num_timestep<<".bin";
      filename=filename_out_Data.str();
      com_->write(Rho,(char*)(filename.c_str()));
    }
}

void grid::TimeAdvance()
{
  Rho=Rho_np1;
  Scalar_Concentration = Scalar_Concentration_np1;
  RU=RU_np1;
  P0=P0_np1;
  part.x=part.x_np1; part.y=part.y_np1; part.z=part.z_np1; part.u=part.u_np1; part.v=part.v_np1; part.w=part.w_np1; part.T=part.T_np1;
  T_cur+=param_->dt();
  num_timestep++;
}

void grid::Update_Rho()
{
  RHS_Rho.Equal_Div_F2C(RU_int); //In fact, here we compute minus RHS_Rho, i.e. div(RU)
  Rho_np1.PlusEqual_Mult(-(param_->dt()*RK4_postCoeff[RK4_count]),RHS_Rho); //Update Rho_np1
  if (RK4_count!=3) Rho_new.Equal_LinComb(1,Rho,-param_->dt()*RK4_preCoeff[RK4_count],RHS_Rho); //update Rho_new
  else Rho_new=Rho_np1;
}

void grid::C_Source()
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
                //Y=size_->dy()/2.+j*size_->dy();
                //Z=size_->dz()/2.+k*size_->dz();
                g(I,J,K)=param_->A_g()*cos(param_->K_g()*X) + param_->B_g()*sin(param_->K_g()*X);//user can manually change this function to the desired one
            }
    g.Update_Ghosts();
    //g.Equal_Ix_F2C(g);
    
    
}

void grid::Update_Scalar_Concentration()
{
    U.Equal_Divide(RU_new,Rho_face); //Compute U_new at cell faces and store it in U
    //std::cout<<"before UC" <<std::endl;
    dummy.Equal_Mult(U,Scalar_Concentration_face);//computing u_int at faces, note: we have already computed Scalar_Concentration_face in previous RK4 substep
    //std::cout<<"after UC" <<std::endl;
    dummy2.x.Equal_Div_F2C(dummy);
    //std::cout<<"before Del2" <<std::endl;
    dummy2.y.Equal_Del2(Scalar_Concentration_int);//div(grad(C))
    //std::cout<<"before source" <<std::endl;
    grid::C_Source();
    //std::cout<<"before add source" <<std::endl;
    RHS_Scalar_Concentration.Equal_LinComb(param_->D_M(),dummy2.y,-1,dummy2.x);
    RHS_Scalar_Concentration+=g;
    U.x.Equal_Ix_F2C(U.x);
    RHS_Scalar_Concentration.PlusEqual_Mult(-1,U.x);
    //RHS_Scalar_Concentration=g;
    //std::cout<<"before diffusion source" <<std::endl;
    Scalar_Concentration_np1.PlusEqual_Mult((param_->dt()*RK4_postCoeff[RK4_count]),RHS_Scalar_Concentration); //Update Scalar_Concentration_np1
    //std::cout<<"after plusEqual_Mult" <<std::endl;
    if (RK4_count!=3) Scalar_Concentration_new.Equal_LinComb(1,Scalar_Concentration,param_->dt()*RK4_preCoeff[RK4_count],RHS_Scalar_Concentration); //update Scalar_Concentration_new
    else Scalar_Concentration_new=Scalar_Concentration_np1;
    
    Scalar_Concentration_face.Equal_I_C2F(Scalar_Concentration_new);
}



void grid::Update_RU_WOP()
{
  //at this point RHS_RU is equal to either zero or the values come from particle depends on TwoWayCoupling On or Off
  divergence.Equal_Div_F2C(U); //Divergence of u_int stored at cell center   Note: U at cell faces is already computed @Update_particle
  dummy2.Equal_Grad_C2F(divergence); // d/dx_i div(u)
  dummy.Equal_Del2(U); //compute div(grad(u_i)) and store it in the dummy variable
  RHS_RU.Equal_LinComb(-param_->mp()/size_->Vcell(),RHS_RU,param_->Mu0()/3.,dummy2,param_->Mu0(),dummy); //RHS = -mp/Vcell*RHS + mu/3*grad(div(U)) + mu*div(grad(U))
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
  //Add body force
  Rho_face -= param_->Rho0(); //make density zero mean! (be careful if you want to reuse it later somewhere else!)
  dummy.x.Equal_Mult(Rho_face.x,param_->gx());
  dummy.y.Equal_Mult(Rho_face.y,param_->gy());
  dummy.z.Equal_Mult(Rho_face.z,param_->gz());
  RHS_RU += dummy;
    
  RU_np1.PlusEqual_Mult(param_->dt()*RK4_postCoeff[RK4_count],RHS_RU); //Update RU_np1
  if (RK4_count!=3) RU_new.Equal_LinComb(1,RU,param_->dt()*RK4_preCoeff[RK4_count],RHS_RU); //update RU_new
  else RU_new = RU_np1;
}

void grid::Update_P0()
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
}

void grid::Compute_Div_U_new()
{
  dummy.x.Equal_Divide(1.,Rho_new);
  dummy.y.Equal_Del2(dummy.x); //compute del2(1/rho_new) and store it in dummy.y
  dummy.z=RHS_Part_Temp;
  if (param_->Is_Cooling()) dummy.z.make_mean_zero();
  divergence.Equal_LinComb( param_->k()/param_->Cp() , dummy.y , -param_->R()/(param_->Cp()*P0_new*size_->Vcell()) , dummy.z );
}

void grid::Compute_RHS_Pois()
{
  Rho_face.Equal_I_C2F(Rho_new); //Interpolate Rho_new to the faces of cell. Note: we reuse this at next RK4 step at Update_RU_WOP() 
  U.Equal_Divide(RU_new,Rho_face); //Compute U_new at cell faces and store it in U
  RHS_Pois.Equal_Div_F2C(U); //Compute div(u_new_wop) and store it in RHS_Pois
  RHS_Pois -= divergence;
  RHS_Pois *= (1./(param_->dt()*RK4_preCoeff[RK4_count]));
  RHS_Pois.make_mean_zero(); //make RHS_Pois zero mean (theoritically we do not need this if dP0/dt term is included, but not computationally!)
}

void grid::Solve_Poisson()
{
  dummy.Equal_Divide(1.,Rho_face); //compute coefficients
  PS_.Solve(dummy,P,RHS_Pois);
}

void grid::Update_RU_WP()
{
  dummy.Equal_Grad_C2F(P); //Compute gradient of hydrodynamic pressure
  RU_np1.PlusEqual_Mult(-(param_->dt()*RK4_postCoeff[RK4_count]),dummy); //Update RU_np1 with pressure 
  if (RK4_count!=3) RU_new.PlusEqual_Mult(-(param_->dt()*RK4_preCoeff[RK4_count]),dummy); //update RU_new with pressure
  else RU_new=RU_np1;
}

void grid::Update_Particle()
{
  //interpolation
  U.Equal_Divide(RU_int,Rho_face); //comupte u_int at faces (note: Rho_face is already computed from previous sub-step @ Compute_RHS_Pois)
  part.gas2part_velocity(U);

  //time integration
  part.update_position(RK4_count,RK4_preCoeff[RK4_count],RK4_postCoeff[RK4_count]);
  part.update_velocity(RK4_count,RK4_preCoeff[RK4_count],RK4_postCoeff[RK4_count]); 
  part.update_Temp(RK4_count,RK4_preCoeff[RK4_count],RK4_postCoeff[RK4_count]); //the RHS of temperature equation is claculated in previous substep
  //projection
  part.part2gas_concentration(C);
  if (param_->TWC()) part.part2gas_velocity(RHS_RU); else RHS_RU=0;
    
  part.Send_Recv();  

  //compute R66HS_Part_Temp at next substep
  T.Equal_Divide(P0_new / param_->R() ,Rho_new); 

  part.gas2part_Temp_new(T);
  part.Compute_RHS_Temp_new();
  part.part2gas_Temp_new(RHS_Part_Temp);
  
}

void grid::TimeAdvance_RK4()
{
  Rho_int=Rho_new;
  RU_int=RU_new;
  P0_int=P0_new;
  Scalar_Concentration_int = Scalar_Concentration_new;
  //particle part
  part.x_int=part.x_new; part.y_int=part.y_new; part.z_int=part.z_new; part.u_int=part.u_new; part.v_int=part.v_new; part.w_int=part.w_new; part.T_int=part.T_new;
}

void grid::Statistics()
{
  if (!param_->Statistics()) return;
  double Rho_mean=Rho.mean();
  double Rho_max=Rho.max();
  double Rho_min=Rho.min();
  double C_max=C.max();
  double C_min=C.min();
  double C_mean=C.mean();
  double TKE2=U.mean_squares();
  double Scalar_Concentration_rms = Scalar_Concentration.rms();
  double Particle_CFL_Max=0;
  double TKE,TKE_U,TKE_V,TKE_W;
  U.Equal_Divide(RU_np1,Rho_face);
  double Gas_CFL_Max=U.max_cfl(param_->dt());
  double u_max=U.max();
  double Vp_max=part.max(part.u);
  double ug_max=part.max(part.ug);
  double Tp_max=part.max(part.T);
  U*=RU_np1;
  TKE_U=U.x.mean();
  TKE_V=U.y.mean();
  TKE_W=U.z.mean();
  TKE=TKE_U+TKE_V+TKE_W;
  double Gas_Max_Diff_CFL = param_->dt() / ( size_->dx() * size_->dx() * Rho_min / param_->Mu0()) * 6.; //when dx=dy=dz
  double Load_Balance=part.Balance_Index();
  part.trajectory(T_cur); //store particle trajectory
  double Tp_mean = part.mean(part.T);
  double HT_mean = mean_energy_transferred;
  dummy.x.Equal_Divide( P0/param_->R(), Rho );
  double Tg_mean = dummy.x.mean();
  if (!pc_->IsRoot()) return;
  stat_Tg<<T_cur<<" "<<Tg_mean<<std::endl;
  stat_Tp<<T_cur<<" "<<Tp_mean<<std::endl;
  stat_HT<<T_cur<<" "<<HT_mean<<std::endl;
  stat_TKE2<<T_cur<<" "<<TKE2<<std::endl;
  stat_TKE<<T_cur<<" "<<TKE<<std::endl;
  stat_TKE_U<<T_cur<<" "<<TKE_U<<std::endl;
  stat_TKE_V<<T_cur<<" "<<TKE_V<<std::endl;
  stat_TKE_W<<T_cur<<" "<<TKE_W<<std::endl;
  stat_Scalar_Concentration_rms<<T_cur<< "" <<Scalar_Concentration_rms<<std::endl;
  stat_P0<<T_cur<<" "<<P0<<std::endl;
  stat_CMax<<T_cur<<" "<<C_max<<std::endl;
  stat_CMin<<T_cur<<" "<<C_min<<std::endl;
  stat_CMean<<T_cur<<" "<<C_mean<<std::endl;
  stat_RhoMax<<T_cur<<" "<<Rho_max<<std::endl;
  stat_RhoMin<<T_cur<<" "<<Rho_min<<std::endl;
  stat_RhoMean<<T_cur<<" "<<Rho_mean<<std::endl;
  stat_ParticleMaxCFL<<T_cur<<" "<<Particle_CFL_Max<<std::endl;    
  stat_GasMaxCFL<<T_cur<<" "<<Gas_CFL_Max<<std::endl;    
  stat_GasMaxDiffCFL<<T_cur<<" "<<Gas_Max_Diff_CFL<<std::endl;
  stat_NumIteration<<T_cur<<" "<<PS_.num_iteration()<<std::endl;
  stat_BalanceIndex<<T_cur<<" "<<Load_Balance<<std::endl;
  if (!param_->Stat_print()) return;
  std::cout<<std::endl<<"::::::::::TIME="<<T_cur<<"::::::::::STEP="<<num_timestep<<"::::::::::"<<std::endl;
  std::cout<<"*** Rho_min="<<Rho_min<<"  ,  Rho_max="<<Rho_max<<"  ,  Rho_mean="<<Rho_mean<<std::endl;
  std::cout<<"*** C_min="<<C_min<<"  ,  C_max="<<C_max<<"  ,  C_mean="<<C_mean<<std::endl;
  std::cout<<"*** Particle Maximum CFL="<<Particle_CFL_Max<<"  ,  Gas Maximum CFL="<<Gas_CFL_Max<<"  ,  Gas Maximum diffusive CFL="<<Gas_Max_Diff_CFL<<std::endl;
  std::cout<<"*** P0="<<P0<<"   ,   Number of Poisson solve iterations="<<PS_.num_iteration()<<std::endl;
  std::cout<<"*** Twice TKE_U ="<<TKE_U<<"  ,  TKE_V ="<<TKE_V<<"  ,  TKE_W ="<<TKE_W<<"  , Twice TKE ="<<TKE<<"  , Twice TKE2="<<TKE2<<std::endl;
  std::cout<<"*** Scalar_Concentration_rms ="<<Scalar_Concentration_rms<<std::endl;
  std::cout<<"*** Particle u_max="<<Vp_max<<"  ,  Gas interpolated u_max="<<ug_max<<"  ,  Gas u_max="<<u_max<<std::endl;
  std::cout<<"Mean energy transferred from particle to gas="<<-mean_energy_transferred<<"  ,  Balance_Index="<<Load_Balance<<std::endl;
  
  std::cout<<"*** Particle Tmax ="<<Tp_max<<std::endl;
 
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
    info<<" I have initially "<<part.Np<<" particles."<<std::endl<<std::endl;
    info.close();
    com_->Sequential_End(x_temp);
  }
  //critical section ends
}

void grid::Test_Poisson()
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
}


void grid::open_stat_file(char *name,std::ofstream &file)
{
  std::string s(name);
  std::ostringstream filename_out_Data;
  filename_out_Data<<param_->stat_dir()<<s<<".dat";
  std::string filename=filename_out_Data.str();
  file.open((char*)(filename.c_str()),std::ios::out|std::ios::app);
}
