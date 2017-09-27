#include "particle.h"
#include "gridsize.h"
#include "proc.h"
#include "params.h"
#include "tensor0.h"
#include "tensor1.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"
#include <iostream>
#include <sstream>

particle::particle(params *p,proc *pc,gridsize *s)
{
  param_=p;
  pc_=pc;
  size_=s;
  idx=1./size_->dx();
  idy=1./size_->dy();
  idz=1./size_->dz();
  
  int particle_density=(int)round(((double)(param_->N0()))/((double)(size_->size_tot())))+1;
  int N_max = ( s->Nx() > s->Ny() ) ? s->Nx() : s->Ny();
  N_max = ( N_max > s->Nz() ) ? N_max : s->Nz();
  Buffer_max_size = 3*N_max*N_max*N_max*particle_density*30; //this needs to be changed.... only for the case we change number of process and wanna load from restart files we need big buffers
  Sbuf0=new double[Buffer_max_size];
  Sbuf1=new double[Buffer_max_size];
  Rbuf0=new double[Buffer_max_size];
  Rbuf1=new double[Buffer_max_size];
  if (pc_->IsRoot())
    {
      Nps=new int[pc_->TOT()];
      if (param_->Statistics())
	{
	  std::ostringstream filename_out_Data;
	  filename_out_Data<<param_->stat_dir()<<"Part_trajectory"<<".dat";
	  std::string filename=filename_out_Data.str();
	  stat_trajectory.open((char*)(filename.c_str()),std::ios::out|std::ios::app);
	}
    }
}

particle::~particle()
{
  delete []Sbuf0;
  delete []Sbuf1;
  delete []Rbuf0;
  delete []Rbuf1;
  if (pc_->IsRoot())
    {
      delete []Nps;
      if (param_->Statistics())
	stat_trajectory.close();
    }
}

void particle::MODE_Lx(int i)
{
  if ( x_new[i] > size_->Lx() )
    {
      x_new[i] -= size_->Lx();
      x[i] -= size_->Lx();
      x_np1[i] -= size_->Lx();
    }
  else if( x_new[i] < 0 )
    {
      x_new[i] += size_->Lx();
      x[i] += size_->Lx();
      x_np1[i] += size_->Lx();
    }
}

void particle::MODE_Ly(int i)
{
  if (y_new[i] > size_->Ly())
    {
      y_new[i] -= size_->Ly();
      y[i] -= size_->Ly();
      y_np1[i] -= size_->Ly();
    }
  else if( y_new[i] < 0 )
    {
      y_new[i] += size_->Ly();
      y[i] += size_->Ly();
      y_np1[i] += size_->Ly();
    }
}

void particle::MODE_Lz(int i)
{
  if (z_new[i] > size_->Lz())
    {
      z_new[i] -= size_->Lz();
      z[i] -= size_->Lz();
      z_np1[i] -= size_->Lz();
    }
  else if(z_new[i] < 0)
    {
      z_new[i] += size_->Lz();
      z[i] += size_->Lz();
      z_np1[i] += size_->Lz();
    }
}

int particle::x2i(double xx)
{
  return (int) (xx*idx) - size_->il() + size_->bs();
}

int particle::y2j(double yy)
{
  return (int) (yy*idy) - size_->jl() + size_->bs();
}

int particle::z2k(double zz)
{
  return (int) (zz*idz) - size_->kl() + size_->bs();
}

int particle::x2i_round(double xx)
{
  return (int)( round( xx*idx ) );
}

int particle::y2j_round(double yy)
{
  return (int)( round( yy*idy ) );
}

int particle::z2k_round(double zz)
{
  return (int)( round( zz*idz ) );
}

void particle::RESIZE()
{
  x.resize(Np); x_int.resize(Np); x_new.resize(Np); x_np1.resize(Np);
  y.resize(Np); y_int.resize(Np); y_new.resize(Np); y_np1.resize(Np);
  z.resize(Np); z_int.resize(Np); z_new.resize(Np); z_np1.resize(Np);
  u.resize(Np); u_int.resize(Np); u_new.resize(Np); u_np1.resize(Np);
  v.resize(Np); v_int.resize(Np); v_new.resize(Np); v_np1.resize(Np);
  w.resize(Np); w_int.resize(Np); w_new.resize(Np); w_np1.resize(Np);
  T.resize(Np); T_int.resize(Np); T_new.resize(Np); T_np1.resize(Np);
  flag.resize(Np);
  RHS_u.resize(Np); RHS_v.resize(Np); RHS_w.resize(Np); RHS_T.resize(Np);
  ug.resize(Np); vg.resize(Np); wg.reserve(Np); Tg.reserve(Np);
}

void particle::SortValidX()
{
  double xl=size_->xl();
  double xh=size_->xh();
  int i(0);
  Snum0=0;
  Snum1=0;
  while (i<Np)
    {
      if (x_new[i]>xh)
	{
	  if (Snum0+29>=Buffer_max_size) std::cout<<"WARNING: Process "<<pc_->RANK()<<" reached its maximum buffersize"<<std::endl;
	  MODE_Lx(i);
	  ToBuf(i,&Sbuf0[Snum0]);
	  Snum0+=29;
	  SWAP(i,Np-1);
	  Np--;
	  i--;
	}
      else if (x_new[i]<xl)
	{
	  if (Snum1+29>=Buffer_max_size) std::cout<<"WARNING: Process "<<pc_->RANK()<<" reached its maximum buffersize"<<std::endl;
	  MODE_Lx(i);
	  ToBuf(i,&Sbuf1[Snum1]);
	  Snum1+=29;
	  SWAP(i,Np-1);
	  Np--;
	  i--;
	}
      i++;
    }
  RESIZE();
}

void particle::SortValidY()
{
  double yl=size_->yl();
  double yh=size_->yh();
  int i(0);
  Snum0=0;
  Snum1=0;
  while (i<Np)
    {
      if (y_new[i]>yh)
	{
	  if (Snum0+29>=Buffer_max_size) std::cout<<"WARNING: Process "<<pc_->RANK()<<" reached its maximum buffersize"<<std::endl;
	  MODE_Ly(i);
	  ToBuf(i,&Sbuf0[Snum0]);
	  Snum0+=29;
	  SWAP(i,Np-1);
	  Np--;
	  i--;
	}
      else if (y_new[i]<yl)
	{
	  if (Snum1+29>=Buffer_max_size) std::cout<<"WARNING: Process "<<pc_->RANK()<<" reached its maximum buffersize"<<std::endl;
	  MODE_Ly(i);
	  ToBuf(i,&Sbuf1[Snum1]);
	  Snum1+=29;
	  SWAP(i,Np-1);
	  Np--;
	  i--;
	}
      i++;
    }
  RESIZE();
}

void particle::SortValidZ()
{
  double zl=size_->zl();
  double zh=size_->zh();
  int i(0);
  Snum0=0;
  Snum1=0;
  while (i<Np)
    {
      if (z_new[i]>zh)
	{
	  if (Snum0+29>=Buffer_max_size) { std::cout<<"WARNING: Process "<<pc_->RANK()<<" reached its maximum buffersize"<<std::endl; return; }
	  MODE_Lz(i);
	  ToBuf(i,&Sbuf0[Snum0]);
	  Snum0+=29;
	  SWAP(i,Np-1);
	  Np--;
	  i--;
	}
      else if (z_new[i]<zl)
	{
	  if (Snum1+29>=Buffer_max_size) std::cout<<"WARNING: Process "<<pc_->RANK()<<" reached its maximum buffersize"<<std::endl;
	  MODE_Lz(i);
	  ToBuf(i,&Sbuf1[Snum1]);
	  Snum1+=29;
	  SWAP(i,Np-1);
	  Np--;
	  i--;
	}
      i++;
    }
  RESIZE();
}

void particle::ToBuf(int i,double *B)
{
  B[0]=x[i];  B[1]=x_int[i];  B[2]=x_new[i];  B[3]=x_np1[i];
  B[4]=y[i];  B[5]=y_int[i];  B[6]=y_new[i];  B[7]=y_np1[i];
  B[8]=z[i];  B[9]=z_int[i];  B[10]=z_new[i]; B[11]=z_np1[i];
  B[12]=u[i]; B[13]=u_int[i]; B[14]=u_new[i]; B[15]=u_np1[i];
  B[16]=v[i]; B[17]=v_int[i]; B[18]=v_new[i]; B[19]=v_np1[i];
  B[20]=w[i]; B[21]=w_int[i]; B[22]=w_new[i]; B[23]=w_np1[i];
  B[24]=T[i]; B[25]=T_int[i]; B[26]=T_new[i]; B[27]=T_np1[i];
  B[28]=(double)(flag[i]); //Cast flag integral number to double precision for sake of communication
}


void particle::FromBuf()
{
  if (Rnum0%29!=0) {std::cout<<"Bad data transfer @ process "<<pc_->RANK()<<std::endl; return;}
  int i(0);
  while (i<Rnum0)
    {
      x.push_back(Rbuf0[i++]); x_int.push_back(Rbuf0[i++]); x_new.push_back(Rbuf0[i++]); x_np1.push_back(Rbuf0[i++]);
      y.push_back(Rbuf0[i++]); y_int.push_back(Rbuf0[i++]); y_new.push_back(Rbuf0[i++]); y_np1.push_back(Rbuf0[i++]);
      z.push_back(Rbuf0[i++]); z_int.push_back(Rbuf0[i++]); z_new.push_back(Rbuf0[i++]); z_np1.push_back(Rbuf0[i++]);
      u.push_back(Rbuf0[i++]); u_int.push_back(Rbuf0[i++]); u_new.push_back(Rbuf0[i++]); u_np1.push_back(Rbuf0[i++]);
      v.push_back(Rbuf0[i++]); v_int.push_back(Rbuf0[i++]); v_new.push_back(Rbuf0[i++]); v_np1.push_back(Rbuf0[i++]);
      w.push_back(Rbuf0[i++]); w_int.push_back(Rbuf0[i++]); w_new.push_back(Rbuf0[i++]); w_np1.push_back(Rbuf0[i++]);
      T.push_back(Rbuf0[i++]); T_int.push_back(Rbuf0[i++]); T_new.push_back(Rbuf0[i++]); T_np1.push_back(Rbuf0[i++]);
      flag.push_back((int)(Rbuf0[i++]));
      Np++;
    }

  if (Rnum1%29!=0) {std::cout<<"Bad data transfer @ process "<<pc_->RANK()<<std::endl; return;}
  i=0;
  while (i<Rnum1)
    {
      x.push_back(Rbuf1[i++]); x_int.push_back(Rbuf1[i++]); x_new.push_back(Rbuf1[i++]); x_np1.push_back(Rbuf1[i++]);
      y.push_back(Rbuf1[i++]); y_int.push_back(Rbuf1[i++]); y_new.push_back(Rbuf1[i++]); y_np1.push_back(Rbuf1[i++]);
      z.push_back(Rbuf1[i++]); z_int.push_back(Rbuf1[i++]); z_new.push_back(Rbuf1[i++]); z_np1.push_back(Rbuf1[i++]);
      u.push_back(Rbuf1[i++]); u_int.push_back(Rbuf1[i++]); u_new.push_back(Rbuf1[i++]); u_np1.push_back(Rbuf1[i++]);
      v.push_back(Rbuf1[i++]); v_int.push_back(Rbuf1[i++]); v_new.push_back(Rbuf1[i++]); v_np1.push_back(Rbuf1[i++]);
      w.push_back(Rbuf1[i++]); w_int.push_back(Rbuf1[i++]); w_new.push_back(Rbuf1[i++]); w_np1.push_back(Rbuf1[i++]);
      T.push_back(Rbuf1[i++]); T_int.push_back(Rbuf1[i++]); T_new.push_back(Rbuf1[i++]); T_np1.push_back(Rbuf1[i++]);
      flag.push_back((int)(Rbuf1[i++]));
      Np++;
    }
  RESIZE();
}

void particle::SWAP(int i, int j)
{
  double temp;
  temp=x[i];  x[i]=x[j];  x[j]=temp;  temp=x_int[i];  x_int[i]=x_int[j];  x_int[j]=temp;  temp=x_new[i];  x_new[i]=x_new[j];  x_new[j]=temp;  temp=x_np1[i];  x_np1[i]=x_np1[j];  x_np1[j]=temp;
  temp=y[i];  y[i]=y[j];  y[j]=temp;  temp=y_int[i];  y_int[i]=y_int[j];  y_int[j]=temp;  temp=y_new[i];  y_new[i]=y_new[j];  y_new[j]=temp;  temp=y_np1[i];  y_np1[i]=y_np1[j];  y_np1[j]=temp;
  temp=z[i];  z[i]=z[j];  z[j]=temp;  temp=z_int[i];  z_int[i]=z_int[j];  z_int[j]=temp;  temp=z_new[i];  z_new[i]=z_new[j];  z_new[j]=temp;  temp=z_np1[i];  z_np1[i]=z_np1[j];  z_np1[j]=temp;
  temp=u[i];  u[i]=u[j];  u[j]=temp;  temp=u_int[i];  u_int[i]=u_int[j];  u_int[j]=temp;  temp=u_new[i];  u_new[i]=u_new[j];  u_new[j]=temp;  temp=u_np1[i];  u_np1[i]=u_np1[j];  u_np1[j]=temp;
  temp=v[i];  v[i]=v[j];  v[j]=temp;  temp=v_int[i];  v_int[i]=v_int[j];  v_int[j]=temp;  temp=v_new[i];  v_new[i]=v_new[j];  v_new[j]=temp;  temp=v_np1[i];  v_np1[i]=v_np1[j];  v_np1[j]=temp;
  temp=w[i];  w[i]=w[j];  w[j]=temp;  temp=w_int[i];  w_int[i]=w_int[j];  w_int[j]=temp;  temp=w_new[i];  w_new[i]=w_new[j];  w_new[j]=temp;  temp=w_np1[i];  w_np1[i]=w_np1[j];  w_np1[j]=temp;   
  temp=T[i];  T[i]=T[j];  T[j]=temp;  temp=T_int[i];  T_int[i]=T_int[j];  T_int[j]=temp;  temp=T_new[i];  T_new[i]=T_new[j];  T_new[j]=temp;  temp=T_np1[i];  T_np1[i]=T_np1[j];  T_np1[j]=temp;
  int temp1;
  temp1=flag[i]; flag[i]=flag[j]; flag[j]=temp1;
}


void particle::update_position(int k,double RK4_pre,double RK4_post)
{
  //the RHS at intermediate time is u_int
  double dt=param_->dt();
  for (int i(0);i<Np;i++)
    {
      x_np1[i] += dt*RK4_post*u_int[i];
      y_np1[i] += dt*RK4_post*v_int[i];
      z_np1[i] += dt*RK4_post*w_int[i];
   }
  if (k!=3)
    for (int i(0);i<Np;i++)
      {
	x_new[i] = x[i]+dt*RK4_pre*u_int[i];
	y_new[i] = y[i]+dt*RK4_pre*v_int[i];
	z_new[i] = z[i]+dt*RK4_pre*w_int[i];
      }
  else
    for (int i(0);i<Np;i++)
      {
	x_new[i] = x_np1[i];
	y_new[i] = y_np1[i];
	z_new[i] = z_np1[i];
      }
}

void particle::gas2part_velocity(tensor1& u)
{
  double x_l,y_l,z_l; //relative position in a gas computational cell
  int ii,jj,kk; //local index notation 
  for (int i(0);i<Np;i++)
    {
      ii = x2i(x_int[i]);
      jj = y2j(y_int[i]);
      kk = z2k(z_int[i]);

      x_l = (x_int[i]*idx-((int)(x_int[i]*idx)));
      y_l = (y_int[i]*idy-((int)(y_int[i]*idy)));
      z_l = (z_int[i]*idz-((int)(z_int[i]*idz)));

      //linear interpolation
      ug[i] = x_l*u.x(ii+1,jj,kk)+(1-x_l)*u.x(ii,jj,kk);
      vg[i] = y_l*u.y(ii,jj+1,kk)+(1-y_l)*u.y(ii,jj,kk);
      wg[i] = z_l*u.z(ii,jj,kk+1)+(1-z_l)*u.z(ii,jj,kk);
      
    }
}

void particle::update_velocity(int k,double RK4_pre,double RK4_post)
{
  double iTp; //one over particle momentum relaxation time
  if (param_->Tp()!=0) iTp=1./param_->Tp();
  //compute RHS
  for (int i(0);i<Np;i++)
    {
      RHS_u[i] = ( ug[i]-u_int[i] )*iTp;
      RHS_v[i] = ( vg[i]-v_int[i] )*iTp;
      RHS_w[i] = ( wg[i]-w_int[i] )*iTp;
    }
  
  // update np1
  double dt=param_->dt();
  double gx=param_->ParticleGravity()*param_->gx();
  double gy=param_->ParticleGravity()*param_->gy();
  double gz=param_->ParticleGravity()*param_->gz();
  for (int i(0);i<Np;i++)
    {
      u_np1[i] += dt*RK4_post*( RHS_u[i]+gx );
      v_np1[i] += dt*RK4_post*( RHS_v[i]+gy );
      w_np1[i] += dt*RK4_post*( RHS_w[i]+gz );
    }
  if (k!=3)
    for (int i(0);i<Np;i++)
      {
	u_new[i] = u[i]+dt*RK4_pre*( RHS_u[i]+gx );
	v_new[i] = v[i]+dt*RK4_pre*( RHS_v[i]+gy );
	w_new[i] = w[i]+dt*RK4_pre*( RHS_w[i]+gz );
      }
  else
    for (int i(0);i<Np;i++)
      {
	u_new[i] = u_np1[i];
	v_new[i] = v_np1[i];
	w_new[i] = w_np1[i];
      }
}

//It is only used before entering the RK4 loop
void particle::gas2part_Temp_int(tensor0& T)
{
  double x_l,y_l,z_l; //relative position in a gas computational cell
  int ii,jj,kk; //local index notation 
  for (int i(0);i<Np;i++)
    {
      //get global index of the up-right-front cell center point
      ii = x2i_round(x_int[i]);
      jj = y2j_round(y_int[i]);
      kk = z2k_round(z_int[i]);

      x_l = x_int[i]*idx-ii+0.5;
      y_l = y_int[i]*idy-jj+0.5;
      z_l = z_int[i]*idz-kk+0.5;

      //make indecies local
      ii += size_->bs()-size_->il();
      jj += size_->bs()-size_->jl();
      kk += size_->bs()-size_->kl();

      //linear interpolation
      Tg[i] = x_l*y_l*z_l*T(ii,jj,kk)+
	(1-x_l)*y_l*z_l*T(ii-1,jj,kk)+x_l*(1-y_l)*z_l*T(ii,jj-1,kk)+x_l*y_l*(1-z_l)*T(ii,jj,kk-1)+
	(1-x_l)*(1-y_l)*z_l*T(ii-1,jj-1,kk)+(1-x_l)*y_l*(1-z_l)*T(ii-1,jj,kk-1)+x_l*(y_l-1)*(z_l-1)*T(ii,jj-1,kk-1)+
	(1-x_l)*(1-y_l)*(1-z_l)*T(ii-1,jj-1,kk-1);
    }
}

void particle::gas2part_Temp_new(tensor0& T)
{
  double x_l,y_l,z_l; //relative position in a gas computational cell
  int ii,jj,kk; //local index notation 
  for (int i(0);i<Np;i++)
    {
      //get global index of the up-right-front cell center point
      ii = x2i_round(x_new[i]);
      jj = y2j_round(y_new[i]);
      kk = z2k_round(z_new[i]);

      x_l = x_new[i]*idx-ii+0.5;
      y_l = y_new[i]*idy-jj+0.5;
      z_l = z_new[i]*idz-kk+0.5;

      //make indecies local
      ii += size_->bs()-size_->il();
      jj += size_->bs()-size_->jl();
      kk += size_->bs()-size_->kl();

      //linear interpolation
      Tg[i] = x_l*y_l*z_l*T(ii,jj,kk)+
	(1-x_l)*y_l*z_l*T(ii-1,jj,kk)+x_l*(1-y_l)*z_l*T(ii,jj-1,kk)+x_l*y_l*(1-z_l)*T(ii,jj,kk-1)+
	(1-x_l)*(1-y_l)*z_l*T(ii-1,jj-1,kk)+(1-x_l)*y_l*(1-z_l)*T(ii-1,jj,kk-1)+x_l*(y_l-1)*(z_l-1)*T(ii,jj-1,kk-1)+
	(1-x_l)*(1-y_l)*(1-z_l)*T(ii-1,jj-1,kk-1);
    }
}

//use gas interm. value to compute RHS for temperature
//It is only used before entering the RK4 loop
void particle::Compute_RHS_Temp_int()
{
  double iTth = param_->Nu() * PI * param_->Dp() * param_->k();
  //compute RHS without the radiation term (unit: watt)
  for (int i(0);i<Np;i++)
    RHS_T[i] = ( Tg[i] - T_int[i] ) * iTth;
 }

void particle::Compute_RHS_Temp_new()
{
  double iTth = param_->Nu() * PI * param_->Dp() * param_->k();
  //compute RHS without the radiation term (unit: watt)
  for (int i(0);i<Np;i++)
    RHS_T[i] = ( Tg[i] - T_new[i] ) * iTth;
}


void particle::update_Temp(int k,double RK4_pre,double RK4_post)
{
  //RHS is already computed at this point @ previous sub-step
  // update np1
  double alpha = 0.25 * param_->epsilonp() * PI * param_->Dp() * param_->Dp() * param_->I0();
  double dt = param_->dt();
  double iS = 1. / ( param_->mp() * param_->Cvp() );
  for (int i(0); i<Np; i++)
    T_np1[i] += dt * RK4_post * iS * ( RHS_T[i] + alpha );
  if (k!=3)
    for (int i(0);i<Np;i++)
      T_new[i] = T[i] + dt * RK4_pre * iS * ( RHS_T[i] + alpha );
  else
    for (int i(0);i<Np;i++)
	T_new[i] = T_np1[i];
}

void particle::part2gas_velocity(tensor1 &A)
{
  double x_l,y_l,z_l; //relative position in a gas computational cell
  int ii,jj,kk; //local index notation 
  A=0;
  for (int i(0);i<Np;i++)
    {
      ii = x2i(x_int[i]);
      jj = y2j(y_int[i]);
      kk = z2k(z_int[i]);
      
      x_l = (x_int[i]*idx-((int)(x_int[i]*idx)));
      y_l = (y_int[i]*idy-((int)(y_int[i]*idy)));
      z_l = (z_int[i]*idz-((int)(z_int[i]*idz)));

      A.x(ii+1,jj,kk) += x_l*RHS_u[i];
      A.x(ii,jj,kk) += (1-x_l)*RHS_u[i];
      
      A.y(ii,jj+1,kk) += y_l*RHS_v[i];
      A.y(ii,jj,kk) += (1-y_l)*RHS_v[i];
      
      A.z(ii,jj,kk+1) += z_l*RHS_w[i];
      A.z(ii,jj,kk) += (1-z_l)*RHS_w[i];
    }
  A.Update_Ghosts_CUM();
  A.Update_Ghosts();
}

void particle::part2gas_concentration(tensor0 &A)
{
  double x_l,y_l,z_l; //relative position in a gas computational cell
  int ii,jj,kk; //local index notation 

  A=0;

  for (int i(0);i<Np;i++)
    {
      //get global index of the up-right-front cell center point
      ii = x2i_round(x_int[i]);
      jj = y2j_round(y_int[i]);
      kk = z2k_round(z_int[i]);

      x_l = x_int[i]*idx-ii+0.5;
      y_l = y_int[i]*idy-jj+0.5;
      z_l = z_int[i]*idz-kk+0.5;

      //make indecies local
      ii += (size_->bs()-size_->il());
      jj += (size_->bs()-size_->jl());
      kk += (size_->bs()-size_->kl());
      
      A(ii,jj,kk) += (x_l*y_l*z_l); 
      A(ii-1,jj,kk) += (1-x_l)*y_l*z_l;
      A(ii,jj-1,kk) += x_l*(1-y_l)*z_l;
      A(ii,jj,kk-1) += x_l*y_l*(1-z_l);      
      A(ii-1,jj-1,kk) += (1-x_l)*(1-y_l)*z_l;
      A(ii-1,jj,kk-1) += (1-x_l)*y_l*(1-z_l);
      A(ii,jj-1,kk-1) += x_l*(1-y_l)*(1-z_l);
      A(ii-1,jj-1,kk-1) += (1-x_l)*(1-y_l)*(1-z_l);
    }
  A.Update_Ghosts_CUM();
}

void particle::part2gas_concentration2(tensor0 &A)
{
  double x_l,y_l,z_l; //relative position in a gas computational cell
  int ii,jj,kk; //local index notation 

  A=0;
  
  for (int i(0);i<Np;i++)
    {
      //get global index of the up-right-front cell center point
      ii = x2i_round(x_int[i]);
      jj = y2j_round(y_int[i]);
      kk = z2k_round(z_int[i]);
      //make indecies local
      ii += (size_->bs()-size_->il());
      jj += (size_->bs()-size_->jl());
      kk += (size_->bs()-size_->kl());
      
      A(ii,jj,kk) += 1;
    }
  A.Update_Ghosts_CUM();
}

//projection based on particle interm. position
//It is only used before entering the RK4 loop
void particle::part2gas_Temp_int(tensor0 &A)
{
  double x_l,y_l,z_l; //relative position in a gas computational cell
  int ii,jj,kk; //local index notation 

  A=0;
  
  for (int i(0);i<Np;i++)
    {
      //get global index of the up-right-front cell center point
      ii = x2i_round(x_int[i]);
      jj = y2j_round(y_int[i]);
      kk = z2k_round(z_int[i]);

      x_l = x_int[i]*idx-ii+0.5;
      y_l = y_int[i]*idy-jj+0.5;
      z_l = z_int[i]*idz-kk+0.5;

      //make indecies local
      ii += size_->bs()-size_->il();
      jj += size_->bs()-size_->jl();
      kk += size_->bs()-size_->kl();
      
      A(ii,jj,kk) += x_l*y_l*z_l*RHS_T[i];      
      A(ii-1,jj,kk) += (1-x_l)*y_l*z_l*RHS_T[i];
      A(ii,jj-1,kk) += x_l*(1-y_l)*z_l*RHS_T[i];
      A(ii,jj,kk-1) += x_l*y_l*(1-z_l)*RHS_T[i];      
      A(ii-1,jj-1,kk) += (1-x_l)*(1-y_l)*z_l*RHS_T[i];
      A(ii-1,jj,kk-1) += (1-x_l)*y_l*(1-z_l)*RHS_T[i];
      A(ii,jj-1,kk-1) += x_l*(1-y_l)*(1-z_l)*RHS_T[i];
      A(ii-1,jj-1,kk-1) += (1-x_l)*(1-y_l)*(1-z_l)*RHS_T[i];
    }
  A.Update_Ghosts_CUM();
  A.Update_Ghosts();
}

//projection based on particle new position

void particle::part2gas_Temp_new(tensor0 &A)
{
  double x_l,y_l,z_l; //relative position in a gas computational cell
  int ii,jj,kk; //local index notation 

  A=0;
  
  for (int i(0);i<Np;i++)
    {
      //get global index of the up-right-front cell center point
      ii = x2i_round(x_new[i]);
      jj = y2j_round(y_new[i]);
      kk = z2k_round(z_new[i]);

      x_l = x_new[i]*idx-ii+0.5;
      y_l = y_new[i]*idy-jj+0.5;
      z_l = z_new[i]*idz-kk+0.5;

      //make indecies local
      ii += size_->bs()-size_->il();
      jj += size_->bs()-size_->jl();
      kk += size_->bs()-size_->kl();
      
      A(ii,jj,kk) += x_l*y_l*z_l*RHS_T[i];      
      A(ii-1,jj,kk) += (1-x_l)*y_l*z_l*RHS_T[i];
      A(ii,jj-1,kk) += x_l*(1-y_l)*z_l*RHS_T[i];
      A(ii,jj,kk-1) += x_l*y_l*(1-z_l)*RHS_T[i];      
      A(ii-1,jj-1,kk) += (1-x_l)*(1-y_l)*z_l*RHS_T[i];
      A(ii-1,jj,kk-1) += (1-x_l)*y_l*(1-z_l)*RHS_T[i];
      A(ii,jj-1,kk-1) += x_l*(1-y_l)*(1-z_l)*RHS_T[i];
      A(ii-1,jj-1,kk-1) += (1-x_l)*(1-y_l)*(1-z_l)*RHS_T[i];
    }
  A.Update_Ghosts_CUM();
  A.Update_Ghosts();
}


/////////////////////////////////Statistics////////////////////////////////////

int particle::NP_TOT()
{
  int LocalSum=Np;
  int GlobalSum;
  MPI_Reduce(&LocalSum,&GlobalSum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Bcast(&GlobalSum,1,MPI_INT,0,MPI_COMM_WORLD);
  return GlobalSum;
}

double particle::max(std::vector<double> &V)
{
  double LocalMax;
  if (Np==0) LocalMax=-1e9; 
  else
    {
      LocalMax=V[0];
      for (int i(1); i<Np;i++)
	if (V[i]>LocalMax) LocalMax=V[i];
    }
  double GlobalMax;
  MPI_Reduce(&LocalMax,&GlobalMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Bcast(&GlobalMax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  return GlobalMax;
}

double particle::min(std::vector<double> &V)
{
  double LocalMin;
  if (Np==0) LocalMin=1e9; 
  else
    {
      LocalMin=V[0];
      for (int i(1); i<Np;i++)
	if (V[i]<LocalMin) LocalMin=V[i];
    }
  double GlobalMin;
  MPI_Reduce(&LocalMin,&GlobalMin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Bcast(&GlobalMin,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  return GlobalMin;
}

double particle::mean(std::vector<double> &V)
{
  double np_tot = NP_TOT();

  double LocalSum = 0;
  if ( Np > 0 )
    {
      for (int i(0); i<Np;i++)
        LocalSum += V[i];
    }
  double GlobalSum;
  MPI_Reduce(&LocalSum,&GlobalSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Bcast(&GlobalSum,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

  return GlobalSum/np_tot;
}

void particle::trajectory(double t)
{
  MPI_Request request_send[1];
  MPI_Request request_recv[1];
  MPI_Status stat[1];
  if (param_->Np_track()==0) return;
  int K = ( param_->N0() + param_->Np_track() - 1 ) / param_->Np_track();
  double Sbuf[3];
  double Rbuf[3];
  if (pc_->IsRoot()) stat_trajectory<<t<<" ";
  for (int i(0); i<param_->N0(); i += K)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      if (pc_->IsRoot())
	MPI_Irecv(Rbuf,3,MPI_DOUBLE,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&request_recv[0]);
      if (find_tag(i,Sbuf[0],Sbuf[1],Sbuf[2]))
	{
	  MPI_Isend(Sbuf,3,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&request_send[0]);
	}
      if (pc_->IsRoot())
	{
	  MPI_Waitall(1,&request_recv[0],&stat[0]);
	  stat_trajectory<<Rbuf[0]<<" "<<Rbuf[1]<<" "<<Rbuf[2]<<" ";
	}
      if (find_tag(i,Sbuf[0],Sbuf[1],Sbuf[2]))
	MPI_Waitall(1,&request_send[0],&stat[0]);
    }
  stat_trajectory<<std::endl;
}
  
bool particle::find_tag(int tag,double &xx,double &yy,double &zz)
{
  for (int i(0); i<Np; i++)
    if ( flag[i] == tag )
      {
	xx = x[i];
	yy = y[i];
	zz = z[i];
	return true;
      }
  return false;
}

double particle::Balance_Index()
{
  double LocalNp = Np;
  double GlobalMax;
  MPI_Reduce(&LocalNp,&GlobalMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Bcast(&GlobalMax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  double Average_Np = ( (double) param_->N0() ) / ( (double) pc_->TOT() );
  return GlobalMax/Average_Np;
}

/////////////////////////////////////////////////////////////////////COMMUNICATOIN///////////////////

void particle::Send_Recv_X()
{
  MPI_Request request_send[2];
  MPI_Request request_recv[2];
  MPI_Status stat[2];
  //First get message length
  MPI_Irecv(&Rnum0,1,MPI_INT,pc_->RIGHT(),21,MPI_COMM_WORLD,&request_recv[0]);
  MPI_Irecv(&Rnum1,1,MPI_INT,pc_->LEFT(),20,MPI_COMM_WORLD,&request_recv[1]);
  SortValidX();
  MPI_Isend(&Snum0,1,MPI_INT,pc_->RIGHT(),20,MPI_COMM_WORLD,&request_send[0]);
  MPI_Isend(&Snum1,1,MPI_INT,pc_->LEFT(),21,MPI_COMM_WORLD,&request_send[1]);
  MPI_Waitall(2,&request_recv[0],&stat[0]);
  MPI_Waitall(2,&request_send[0],&stat[0]);
  // Now send actual data
  MPI_Irecv(Rbuf0,Rnum0,MPI_DOUBLE,pc_->RIGHT(),31,MPI_COMM_WORLD,&request_recv[0]);
  MPI_Irecv(Rbuf1,Rnum1,MPI_DOUBLE,pc_->LEFT(),30,MPI_COMM_WORLD,&request_recv[1]);
  MPI_Isend(Sbuf0,Snum0,MPI_DOUBLE,pc_->RIGHT(),30,MPI_COMM_WORLD,&request_send[0]);
  MPI_Isend(Sbuf1,Snum1,MPI_DOUBLE,pc_->LEFT(),31,MPI_COMM_WORLD,&request_send[1]);
  MPI_Waitall(2,&request_recv[0],&stat[0]);
  FromBuf();
  MPI_Waitall(2,&request_send[0],&stat[0]);
}

void particle::Send_Recv_Y()
{
  MPI_Request request_send[2];
  MPI_Request request_recv[2];
  MPI_Status stat[2];
  //First get message length
  MPI_Irecv(&Rnum0,1,MPI_INT,pc_->TOP(),23,MPI_COMM_WORLD,&request_recv[0]);
  MPI_Irecv(&Rnum1,1,MPI_INT,pc_->BOT(),22,MPI_COMM_WORLD,&request_recv[1]);
  SortValidY();
  MPI_Isend(&Snum0,1,MPI_INT,pc_->TOP(),22,MPI_COMM_WORLD,&request_send[0]);
  MPI_Isend(&Snum1,1,MPI_INT,pc_->BOT(),23,MPI_COMM_WORLD,&request_send[1]);
  MPI_Waitall(2,&request_recv[0],&stat[0]);
  MPI_Waitall(2,&request_send[0],&stat[0]);
  // Now send actual data
  MPI_Irecv(Rbuf0,Rnum0,MPI_DOUBLE,pc_->TOP(),33,MPI_COMM_WORLD,&request_recv[0]);
  MPI_Irecv(Rbuf1,Rnum1,MPI_DOUBLE,pc_->BOT(),32,MPI_COMM_WORLD,&request_recv[1]);
  MPI_Isend(Sbuf0,Snum0,MPI_DOUBLE,pc_->TOP(),32,MPI_COMM_WORLD,&request_send[0]);
  MPI_Isend(Sbuf1,Snum1,MPI_DOUBLE,pc_->BOT(),33,MPI_COMM_WORLD,&request_send[1]);
  MPI_Waitall(2,&request_recv[0],&stat[0]);
  FromBuf();
  MPI_Waitall(2,&request_send[0],&stat[0]);
}

void particle::Send_Recv_Z()
{
  MPI_Request request_send[2];
  MPI_Request request_recv[2];
  MPI_Status stat[2];
  //First get message length
  MPI_Irecv(&Rnum0,1,MPI_INT,pc_->FRONT(),25,MPI_COMM_WORLD,&request_recv[0]);
  MPI_Irecv(&Rnum1,1,MPI_INT,pc_->REAR(),24,MPI_COMM_WORLD,&request_recv[1]);
  SortValidZ();
  MPI_Isend(&Snum0,1,MPI_INT,pc_->FRONT(),24,MPI_COMM_WORLD,&request_send[0]);
  MPI_Isend(&Snum1,1,MPI_INT,pc_->REAR(),25,MPI_COMM_WORLD,&request_send[1]);
  MPI_Waitall(2,&request_recv[0],&stat[0]);
  MPI_Waitall(2,&request_send[0],&stat[0]);
  // Now send actual data
  MPI_Irecv(Rbuf0,Rnum0,MPI_DOUBLE,pc_->FRONT(),35,MPI_COMM_WORLD,&request_recv[0]);
  MPI_Irecv(Rbuf1,Rnum1,MPI_DOUBLE,pc_->REAR(),34,MPI_COMM_WORLD,&request_recv[1]);
  MPI_Isend(Sbuf0,Snum0,MPI_DOUBLE,pc_->FRONT(),34,MPI_COMM_WORLD,&request_send[0]);
  MPI_Isend(Sbuf1,Snum1,MPI_DOUBLE,pc_->REAR(),35,MPI_COMM_WORLD,&request_send[1]);
  MPI_Waitall(2,&request_recv[0],&stat[0]);
  FromBuf();
  MPI_Waitall(2,&request_send[0],&stat[0]);
}

void particle::Send_Recv()
{
  Send_Recv_X(); //first transfer outlying particles in x direction
  Send_Recv_Y(); //then y
  Send_Recv_Z(); //then z
}

void particle::Store_All(int num_ts)
{
  ///////////////// //first store number of particles per process////////////////
  MPI_Gather(&Np,1,MPI_INT,Nps,1,MPI_INT,pc_->ROOT(),MPI_COMM_WORLD); //get number of particles per process and store in root process
    //compute how many particles owned by proccesses with less rank
  if (pc_->IsRoot())
    {
      for (int i(1);i<pc_->TOT();i++)
	Nps[i]+=Nps[i-1];
      for (int i(pc_->TOT()-1);i>0;i--)
	Nps[i]=Nps[i-1];
      Nps[0] = 0;
    }
  MPI_Scatter(Nps,1,MPI_INT,&OFF_SET,1,MPI_INT,pc_->ROOT(),MPI_COMM_WORLD);
  ////////////////////////////Then store particle X,U,T,flags///////////////////////////
  Store("Part_x",num_ts,&x[0]);
  Store("Part_y",num_ts,&y[0]);
  Store("Part_z",num_ts,&z[0]);
  Store("Part_u",num_ts,&u[0]);
  Store("Part_v",num_ts,&v[0]);
  Store("Part_w",num_ts,&w[0]);
  Store("Part_T",num_ts,&T[0]);
  Store("Part_flag",num_ts,&flag[0]);
}

void particle::Store(char *name,int num_ts,double *p)
{
  std::string s(name);
  std::ostringstream filename_out_Data;
  filename_out_Data<<param_->data_dir()<<s<<"_"<<num_ts<<".bin";
  std::string filename=filename_out_Data.str();
  MPI_File Myfile;
  MPI_File_open(MPI_COMM_WORLD,(char*)(filename.c_str()),MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&Myfile);
  MPI_Offset offset=OFF_SET*sizeof(double);
  MPI_File_write_at_all(Myfile,offset,p,Np,MPI_DOUBLE,MPI_STATUS_IGNORE);
  MPI_File_close(&Myfile);
}

void particle::Store(char *name,int num_ts,int *p)
{
  std::string s(name);
  std::ostringstream filename_out_Data;
  filename_out_Data<<param_->data_dir()<<s<<"_"<<num_ts<<".bin";
  std::string filename=filename_out_Data.str();
  MPI_File Myfile;
  MPI_File_open(MPI_COMM_WORLD,(char*)(filename.c_str()),MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&Myfile);
  MPI_Offset offset=OFF_SET*sizeof(int);
  MPI_File_write_at_all(Myfile,offset,p,Np,MPI_INT,MPI_STATUS_IGNORE);
  MPI_File_close(&Myfile);
}

void particle::load_random()
{
  double xl=size_->xl();
  double xh=size_->xh();
  double yl=size_->yl();
  double yh=size_->yh();
  double zl=size_->zl();
  double zh=size_->zh();

  double T0 = param_->T0();

  if ( pc_->IsRoot() ) Np = param_->N0() / pc_->TOT() + param_->N0() % pc_->TOT();
  else Np = param_->N0() / pc_->TOT();

  int flag_number = ( pc_->IsRoot() ) ? 0 : pc_->RANK() * ( param_->N0() / pc_->TOT() ) + param_->N0() % pc_->TOT(); 

  double rnd;
  //srand(pc_->RANK());
  for (int i(0); i<Np; i++)
    {
      rnd = ( (double) ( rand() ) / (double) (RAND_MAX) ) * ( size_->xh() - size_->xl() ) + size_->xl();
      x.push_back(rnd);
      u.push_back(0);
      rnd = ( (double) ( rand() ) / (double) (RAND_MAX) ) * ( size_->yh() - size_->yl() ) + size_->yl();
      y.push_back(rnd);
      v.push_back(0);
      rnd = ( (double) ( rand() ) / (double) (RAND_MAX) ) * ( size_->zh() - size_->zl() ) + size_->zl();
      z.push_back(rnd);
      w.push_back(0);
      if ( ( x.back() > xh ) || ( x.back() < xl ) ) std::cout<<"RANK="<<pc_->RANK()<<" X ERROR"<<std::endl;
      if ( ( y.back() > yh ) || ( y.back() < yl ) ) std::cout<<"RANK="<<pc_->RANK()<<" Y ERROR"<<std::endl;
      if ( ( z.back() > zh ) || ( z.back() < zl ) ) std::cout<<"RANK="<<pc_->RANK()<<" Z ERROR"<<std::endl;
      T.push_back(T0);
      flag.push_back(flag_number++);
    }
  //make sure all other fileds have enough memory (i.e RHS's and gas interpolated arrays)
  RESIZE();
}


void particle::Load_All()
{
  if ( pc_->IsRoot() ) Np = param_->N0() / pc_->TOT() + param_->N0() % pc_->TOT();
  else Np = param_->N0() / pc_->TOT();
  RESIZE();
  OFF_SET = ( param_->N0() / pc_->TOT() ) * pc_->RANK() + ( param_->N0() % pc_->TOT() ) * ( 1 - pc_->IsRoot() );
  Load("Part_x",&x[0]);
  Load("Part_y",&y[0]);
  Load("Part_z",&z[0]);
  Load("Part_u",&u[0]);
  Load("Part_v",&v[0]);
  Load("Part_w",&w[0]);
  Load("Part_T",&T[0]);
  Load("Part_flag",&flag[0]);
  
  int N_logical_max = ( pc_->NX() > pc_->NY() ) ? pc_->NX() : pc_->NY(); //the maximum number of process in 3 directions
  N_logical_max = ( N_logical_max > pc_->NZ() ) ? N_logical_max : pc_->NZ();

  //transfer all loaded locations to new filed since process-correction transfomration is based on x_new !
  x_new = x; y_new = y; z_new = z;

  //make sure all loaded particles reside in propper process
  for ( int i(0); i < N_logical_max; i++)
    Send_Recv();
}

void particle::Load(char *name,double *p)
{
  std::string s(name);
  std::ostringstream filename_out_Data;
  filename_out_Data<<"Restart_"<<s<<".bin";
  std::string filename = filename_out_Data.str();
  MPI_File Myfile;
  MPI_File_open(MPI_COMM_WORLD,(char*)(filename.c_str()),MPI_MODE_RDONLY,MPI_INFO_NULL,&Myfile);
  MPI_Offset offset=OFF_SET*sizeof(double);
  MPI_File_read_at_all(Myfile,offset,p,Np,MPI_DOUBLE,MPI_STATUS_IGNORE);
  MPI_File_close(&Myfile);
}

void particle::Load(char *name,int *p)
{
  std::string s(name);
  std::ostringstream filename_out_Data;
  filename_out_Data<<"Restart_"<<s<<".bin";
  std::string filename=filename_out_Data.str();
  MPI_File Myfile;
  MPI_File_open(MPI_COMM_WORLD,(char*)(filename.c_str()),MPI_MODE_RDONLY,MPI_INFO_NULL,&Myfile);
  MPI_Offset offset=OFF_SET*sizeof(int);
  MPI_File_read_at_all(Myfile,offset,p,Np,MPI_INT,MPI_STATUS_IGNORE);
  MPI_File_close(&Myfile);
}

void particle::isOverlap()
{
  bool found = false;
  double xl=size_->xl();
  double xh=size_->xh();
  for ( int i = 0; i<Np; i++ )
    {
      if ((x[i]>xh)|| (x[i]<xl))
	{
	  std::cout<<"Out of bound particle found!"<<std::endl;
	  found = true;
	}
    }
  double yl=size_->yl();
  double yh=size_->yh();
  for ( int i = 0; i<Np; i++ )
    {
      if ((y[i]>yh)|| (y[i]<yl))
	{
	  std::cout<<"Out of bound particle found!"<<std::endl;
	  found = true;
	}
    }
  double zl=size_->zl();
  double zh=size_->zh();
  for ( int i = 0; i<Np; i++ )
    {
      if ((z[i]>zh)|| (z[i]<zl))
	{
	  std::cout<<"Out of bound particle found!"<<std::endl;
	  found = true;
	}
    }

  for ( int i(0); i<Np; i++ )
    {
      for ( int j(i+1); j<Np; j++)
	{
	  if ( ( (x[i] == x[j]) ) )// && (y[i]==y[j]) ) && (z[i]==z[j]) )
	    {
	      std::cout<<"overlap particles ( by x )found by Proc "<<pc_->RANK()<<" !\n";
	      std::cout<<" particle 1: ID="<<flag[i]<<"  x= "<<x[i]<<"  y= "<<y[i]<<"  z= "<<z[i]<<"  u_np1= "<<u_np1[i]<<"  v_np1= "<<v_np1[i]<<"  w_np1= "<<w_np1[i];
	      std::cout<<"  xnp1= "<<x_np1[i]<<"  ynp1= "<<y_np1[i]<<"  znp1= "<<z_np1[i]<<std::endl;
	      std::cout<<" particle 2: ID="<<flag[j]<<"  x= "<<x[j]<<"  y= "<<y[j]<<"  z= "<<z[j]<<"  u_np1= "<<u_np1[j]<<"  v_np1= "<<v_np1[j]<<"  w_np1= "<<w_np1[j];
	      std::cout<<"  xnp1= "<<x_np1[j]<<"  ynp1= "<<y_np1[j]<<"  znp1= "<<z_np1[j]<<std::endl;
	      found = true;
	    }
	  if ( ( (x_np1[i] == x_np1[j]) && (y_np1[i]==y_np1[j]) ) && (z_np1[i]==z_np1[j]) )
	    {
	      std::cout<<"overlap particles ( by x_np1 )found!\n";
	      std::cout<<" particle ID = "<<flag[i]<<" and "<<flag[j]<<std::endl;
	      found = true;
	    }
	  if ( found )	
	    {
	      exit(0);
	    }
	}
    }
  std::cout<<"I am Proc. "<<pc_->RANK()<<" and passed the overLap test\n";  
}
